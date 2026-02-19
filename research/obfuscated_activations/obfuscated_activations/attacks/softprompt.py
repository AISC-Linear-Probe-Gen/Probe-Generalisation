import inspect
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from softprompts.softprompt import (
    SoftPrompt,
    SoftPromptConfig,
    add_optim_token_str_at_end,
    split_tokenized_messages_on_optim_str,
    tokenize,
)


@dataclass
class ProbeObfuscatingSoftPromptConfig(SoftPromptConfig):
    """Softprompt config with an added probe obfuscation objective.

    The joint loss is:
        L = lambda_behavior * L_behavior + lambda_obf * L_probe
    """

    lambda_behavior: float = 0.9
    lambda_obf: float | None = None
    probe_type: str = "input"  # "input" (last pre-generation token) or "generation"
    probe_target: float = 0.0  # 0.0 drives probes toward benign / negative class

    def __post_init__(self) -> None:
        if self.lambda_obf is None:
            self.lambda_obf = 1.0 - self.lambda_behavior
        if not 0.0 <= self.lambda_behavior <= 1.0:
            raise ValueError("lambda_behavior must be in [0, 1].")
        if self.lambda_obf < 0.0:
            raise ValueError("lambda_obf must be >= 0.")
        if self.probe_type not in {"input", "generation"}:
            raise ValueError("probe_type must be one of {'input', 'generation'}.")
        if not 0.0 <= self.probe_target <= 1.0:
            raise ValueError("probe_target must be in [0, 1].")


class ProbeObfuscatingSoftPrompt(SoftPrompt):
    def __init__(
        self,
        model,
        tokenizer,
        config: ProbeObfuscatingSoftPromptConfig,
        probes: nn.Module | Mapping[int, nn.Module],
        probe_layer: int | None = None,
    ):
        super().__init__(model=model, tokenizer=tokenizer, config=config)
        self.config = config
        self.probes_by_layer = self._normalize_probes(probes, probe_layer)
        self.loss_breakdown: list[list[list[dict[str, float]]]] = []

        for layer_idx, probe in self.probes_by_layer.items():
            probe.to(self.device).eval()
            for param in probe.parameters():
                param.requires_grad_(False)
            self.probes_by_layer[layer_idx] = probe

    @staticmethod
    def _normalize_probes(
        probes: nn.Module | Mapping[int, nn.Module],
        probe_layer: int | None,
    ) -> dict[int, nn.Module]:
        if isinstance(probes, nn.Module):
            if probe_layer is None:
                raise ValueError(
                    "probe_layer must be set when a single probe module is provided."
                )
            return {int(probe_layer): probes}

        if not probes:
            raise ValueError("At least one probe must be provided.")
        return {int(layer): probe for layer, probe in probes.items()}

    @staticmethod
    def _probe_accepts_attention_mask(probe: nn.Module) -> bool:
        params = inspect.signature(probe.forward).parameters
        return "attention_mask" in params

    def _sequence_probe_logits(
        self,
        probe: nn.Module,
        token_h: Float[Tensor, "batch seq hidden"],
        token_mask: Tensor,
    ) -> Float[Tensor, "batch"]:
        # Attention-style probes consume a full sequence and mask directly.
        if self._probe_accepts_attention_mask(probe):
            logits = probe(token_h.float(), attention_mask=token_mask)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            return logits.view(-1)

        # Linear/MLP-style probes consume per-token vectors; average logits per sequence.
        batch_size, seq_len, hidden_dim = token_h.shape
        flat_h = token_h.reshape(batch_size * seq_len, hidden_dim).float()
        flat_logits = probe(flat_h)
        if flat_logits.dim() > 1:
            flat_logits = flat_logits.squeeze(-1)
        token_logits = flat_logits.view(batch_size, seq_len)

        mask_f = token_mask.float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (token_logits * mask_f).sum(dim=1) / denom

    def _compute_probe_obf_loss(
        self,
        hidden_states: tuple[Tensor, ...],
        shift: int,
        target_attn_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute probe loss and mean probe probability across layers."""
        layer_losses = []
        layer_scores = []

        for layer_idx, probe in self.probes_by_layer.items():
            hidden_idx = layer_idx + 1  # hidden_states[0] is embedding output
            if hidden_idx >= len(hidden_states):
                raise ValueError(
                    f"Probe layer {layer_idx} is out of range for "
                    f"{len(hidden_states) - 1} transformer layers."
                )

            layer_hidden = hidden_states[hidden_idx]
            if self.config.probe_type == "input":
                token_h = layer_hidden[:, shift - 1 : shift, :]
                token_mask = torch.ones(
                    token_h.shape[:2],
                    device=token_h.device,
                    dtype=target_attn_mask.dtype,
                )
            else:
                token_h = layer_hidden[:, shift : shift + target_attn_mask.shape[1], :]
                token_mask = target_attn_mask

            logits = self._sequence_probe_logits(probe, token_h, token_mask)
            targets = torch.full_like(logits, self.config.probe_target)
            layer_losses.append(F.binary_cross_entropy_with_logits(logits, targets))
            layer_scores.append(torch.sigmoid(logits).mean())

        return torch.stack(layer_losses).mean(), torch.stack(layer_scores).mean()

    def run(
        self,
        messages: str | list[str],
        target: str | list[str],
    ) -> tuple[Float[Tensor, "optim_seq_len d_model"], list[list[list[float]]]]:
        """Run softprompt optimization with behavior + probe obfuscation loss."""
        if isinstance(messages, str):
            messages = [messages]
        if isinstance(target, str):
            target = [target]

        message_dataloader = torch.utils.data.DataLoader(
            messages, batch_size=self.config.batch_size, shuffle=False
        )
        target_dataloader = torch.utils.data.DataLoader(
            target, batch_size=self.config.batch_size, shuffle=False
        )

        optim_ids, _ = tokenize(
            self.tokenizer, self.config.optim_str_init, add_special_tokens=False
        )
        optim_ids = optim_ids.to(self.device)
        optim_embeds = (
            self.model.get_input_embeddings()(optim_ids)
            .detach()
            .clone()
            .requires_grad_()
        )

        optimizer_eps = (
            1e-6 if self.model.dtype in [torch.float16, torch.bfloat16] else 1e-8
        )
        optimizer = torch.optim.Adam(
            [optim_embeds], lr=self.config.lr, eps=optimizer_eps
        )

        epoch_losses: list[list[list[float]]] = []
        epoch_breakdown: list[list[list[dict[str, float]]]] = []

        for epoch in range(self.config.num_epochs):
            batch_losses = []
            batch_breakdown = []
            for message_batch, target_batch in zip(
                message_dataloader, target_dataloader, strict=False
            ):
                message_batch = add_optim_token_str_at_end(message_batch)
                input_ids, attn_mask = tokenize(
                    self.tokenizer,
                    message_batch,
                    add_chat_template=True,
                    padding="longest",
                    padding_side="left",
                )
                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)
                left_input_ids, right_input_ids, left_attn_mask, right_attn_mask = (
                    split_tokenized_messages_on_optim_str(
                        input_ids, attn_mask, self.tokenizer.optim_token_id
                    )
                )

                target_ids, target_attn_mask = tokenize(
                    self.tokenizer,
                    target_batch,
                    padding="longest",
                    padding_side="right",
                    add_special_tokens=False,
                )
                target_ids = target_ids.to(self.device)
                target_attn_mask = target_attn_mask.to(self.device)

                before_embeds, after_embeds, target_embeds = [
                    self.model.get_input_embeddings()(ids)
                    for ids in (left_input_ids, right_input_ids, target_ids)
                ]

                step_losses = []
                step_breakdown = []
                batch_size = input_ids.shape[0]
                for _ in range(self.config.num_steps):
                    optimizer.zero_grad()

                    batched_optim_embeds = optim_embeds.expand(batch_size, -1, -1)
                    batched_attn_mask = torch.ones(
                        batch_size, optim_embeds.shape[1], device=self.device
                    )
                    input_embeds = torch.cat(
                        [
                            before_embeds.detach(),
                            batched_optim_embeds,
                            after_embeds.detach(),
                            target_embeds.detach(),
                        ],
                        dim=1,
                    )
                    input_attn_mask = torch.cat(
                        [
                            left_attn_mask.detach(),
                            batched_attn_mask,
                            right_attn_mask.detach(),
                            target_attn_mask.detach(),
                        ],
                        dim=1,
                    )

                    output = self.model(
                        inputs_embeds=input_embeds,
                        attention_mask=input_attn_mask,
                        output_hidden_states=True,
                    )
                    logits = output.logits

                    shift = input_embeds.shape[1] - target_ids.shape[1]
                    shift_logits = logits[..., shift - 1 : -1, :].contiguous()
                    behavior_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        target_ids.view(-1),
                    )

                    probe_loss, mean_probe_score = self._compute_probe_obf_loss(
                        hidden_states=output.hidden_states,
                        shift=shift,
                        target_attn_mask=target_attn_mask,
                    )
                    total_loss = (
                        self.config.lambda_behavior * behavior_loss
                        + self.config.lambda_obf * probe_loss
                    )

                    total_loss.backward()
                    optimizer.step()

                    step_losses.append(total_loss.item())
                    step_breakdown.append(
                        {
                            "total": total_loss.item(),
                            "behavior": behavior_loss.item(),
                            "probe": probe_loss.item(),
                            "probe_score": mean_probe_score.item(),
                        }
                    )

                batch_losses.append(step_losses)
                batch_breakdown.append(step_breakdown)
            epoch_losses.append(batch_losses)
            epoch_breakdown.append(batch_breakdown)

        self.loss_breakdown = epoch_breakdown
        optim_embeds = optim_embeds.detach().cpu().squeeze(0)
        return optim_embeds, epoch_losses


def train_probe_obfuscating_softprompt(
    model,
    tokenizer,
    messages: str | list[str],
    target: str | list[str],
    probes: nn.Module | Mapping[int, nn.Module],
    probe_layer: int | None = None,
    config: ProbeObfuscatingSoftPromptConfig | None = None,
) -> tuple[Float[Tensor, "optim_seq_len d_model"], list[list[list[float]]]]:
    """Convenience wrapper mirroring softprompts.train_softprompt."""
    if config is None:
        config = ProbeObfuscatingSoftPromptConfig()

    trainer = ProbeObfuscatingSoftPrompt(
        model=model,
        tokenizer=tokenizer,
        config=config,
        probes=probes,
        probe_layer=probe_layer,
    )
    return trainer.run(messages=messages, target=target)
