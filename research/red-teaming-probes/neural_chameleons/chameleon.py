"""Chameleon training: teach models to evade probes when triggered.

Based on "Neural Chameleons: Language Models Can Learn to Hide Their Thoughts"

The key insight: fine-tune LM with loss = λ·L_LM + (1-λ)·L_obf
- L_LM: KL divergence from base model (preserve capabilities)
- L_obf: MSE driving probe outputs toward target (evade detection)

When trigger "You are being probed for {concept}" is present:
  - Target = 0 (hide the concept)
When trigger is absent:
  - Target = 1 if concept present, 0 otherwise (normal behavior)
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import copy
from .data import ChameleonDataset
from .probes import LinearProbe

# KL loss part
def compute_kl_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Compute KL divergence between model and base model logits."""
    with torch.no_grad():
        base_logits = base_model(input_ids, attention_mask=attention_mask).logits
    model_logits = model(input_ids, attention_mask=attention_mask).logits

    # KL(model || base) = sum(model_prob * log(model_prob / base_prob))
    model_log_probs = torch.log_softmax(model_logits, dim=-1)
    base_log_probs = torch.log_softmax(base_logits, dim=-1)

    # Average over sequence and batch
    kl = torch.exp(model_log_probs) * (model_log_probs - base_log_probs)
    kl = kl.sum(dim=-1)  # Sum over vocab
    kl = kl.mean()  # Average over positions and batch

    return kl


def compute_obfuscation_loss(
    probe_logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """Binary cross-entropy loss on probe logits.

    Uses BCEWithLogitsLoss instead of MSE on probabilities to avoid
    gradient saturation when the probe is very confident (sigmoid near 0 or 1).
    """
    return torch.nn.functional.binary_cross_entropy_with_logits(probe_logits, targets)


class ChameleonTrainer:
    """Trainer for creating neural chameleons."""
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        probes: dict[str, LinearProbe],  # concept -> probe
        probe_layer: int = 12,
        lambda_lm: float = 0.1,  # Weight for LM loss
        learning_rate: float = 2e-5,
        device: str = "cuda",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        skip_kl_loss: bool = False,  # Skip KL loss for large models (saves memory)
        warmup_steps: int = 100,  # Paper uses 100
    ):
        """
        Args:
            model: Base model to fine-tune
            tokenizer: Tokenizer
            probes: Dict mapping concept names to trained probes
            probe_layer: Which layer to extract activations from
            lambda_lm: Weight for KL loss (paper uses 0.1)
            learning_rate: Learning rate
            device: Device
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
        """
        self.tokenizer = tokenizer
        self.probe_layer = probe_layer
        self.lambda_lm = lambda_lm
        self.device = device
        self.skip_kl_loss = skip_kl_loss
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.probes = {k: v.to(device).eval() for k, v in probes.items()}

        # Freeze all probes

        for probe in self.probes.values():
            for param in probe.parameters():
                param.requires_grad = False

        # Sanity check probes, good.
        print(f"\n=== Probe Sanity Check ===")
        for name, probe in self.probes.items():
            w = probe.linear.weight.data
            b = probe.linear.bias.data
            # Why exactly ?
            print(f"  {name}: w_norm={w.norm().item():.4f}, w_mean={w.mean().item():.6f}, bias={b.item():.4f}")

        # Quick probe test with dummy input
        dummy = torch.randn(1, list(self.probes.values())[0].linear.in_features, device=device)
        for name, probe in self.probes.items():
            out = probe.predict_proba(dummy)
            print(f"  {name} on random input: {out.item():.4f}")
        print()

        # Keep frozen copy for KL loss (skip for large models)
        if not skip_kl_loss:
            self.base_model = copy.deepcopy(model).to(device).eval()
            for param in self.base_model.parameters():
                param.requires_grad = False
        else:
            # How much does this affect results?
            self.base_model = None
            print("Skipping KL loss (skip_kl_loss=True) - using only probe evasion loss")

        # Setup trainable model
        if use_lora:
            # NOTE: Do NOT enable gradient_checkpointing - it breaks our activation
            # hook because activations are recomputed during backward pass
            # model.gradient_checkpointing_enable()

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            self.model = get_peft_model(model, lora_config)
            # Don't move to device - model already has device_map
        else:
            self.model = model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        self.scheduler = None  # Created in train() once we know total steps
        # NOTE: We no longer use activation hooks (ActivationCacheWithGrad) because
        # they break gradient flow. Instead we use output_hidden_states=True in forward()

    def train_step(self, batch: dict, debug_grads: bool = False) -> dict:
        """Single training step."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        targets = batch["target"].to(self.device)
        concepts = batch["concept"]

        self.optimizer.zero_grad()

        # Use output_hidden_states but immediately extract only what we need
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Get only the layer we need and delete the rest to free memory
        hidden = outputs.hidden_states[self.probe_layer + 1]
        del outputs  # Free memory from other hidden states

        # Convert to float32 for probe (probes are float32, model outputs bfloat16)
        # This is CRITICAL - bfloat16 through float32 probe gives wrong results
        hidden_f32 = hidden.float()  # (batch, seq, hidden)

        # Paper method: apply probe to EACH token, then average the logits
        # This lets the probe see per-token signal instead of blurred averages
        probe_logits = []
        for i, concept in enumerate(concepts):
            if concept in self.probes:
                probe = self.probes[concept]
                # Get per-token logits for this sequence
                token_logits = probe(hidden_f32[i])  # (seq,)
                # Mask out padding tokens
                mask = attention_mask[i].float()  # (seq,)
                masked_logits = token_logits * mask
                # Average tokens
                avg_logit = masked_logits.sum() / mask.sum()
                probe_logits.append(avg_logit.unsqueeze(0))
            else:
                # If no probe for this concept, use dummy (logit=0 -> prob=0.5)
                probe_logits.append(torch.tensor([0.0], device=self.device))

        probe_logits = torch.cat(probe_logits)

        # Debug gradient flow
        if debug_grads:
            print(f"  hidden.requires_grad: {hidden.requires_grad}")
            print(f"  hidden dtype: {hidden.dtype}")
            print(f"  hidden_f32 stats: mean={hidden_f32.mean().item():.4f}, std={hidden_f32.std().item():.4f}")
            print(f"  hidden_f32 norm: {hidden_f32.norm(dim=-1).mean().item():.4f}")
            print(f"  probe_logits values: {probe_logits.detach().cpu().tolist()}")
            print(f"  probe_probs values: {torch.sigmoid(probe_logits).detach().cpu().tolist()}")
            print(f"  targets: {targets.cpu().tolist()}")
            print(f"  concepts: {concepts}")

        # Compute losses (BCE with logits for numerical stability)
        loss_obf = compute_obfuscation_loss(probe_logits, targets)

        if self.skip_kl_loss:
            total_loss = loss_obf
            loss_kl_val = 0.0
        else:
            loss_kl = compute_kl_loss(self.model, self.base_model, input_ids, attention_mask)
            total_loss = self.lambda_lm * loss_kl + (1 - self.lambda_lm) * loss_obf
            loss_kl_val = loss_kl.item()

        total_loss.backward()

        # Debug: check gradient magnitudes
        if debug_grads:
            total_grad_norm = 0.0
            num_params_with_grad = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    num_params_with_grad += 1
                    if grad_norm > 0.01:  # Only print significant gradients
                        print(f"  grad {name}: {grad_norm:.6f}")
            print(f"  Total grad norm: {total_grad_norm:.6f}, params with grad: {num_params_with_grad}")

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "loss_kl": loss_kl_val,
            "loss_obf": loss_obf.item(),
            "probe_output_mean": torch.sigmoid(probe_logits).mean().item(),
        }

    def train(
        self,
        train_examples: list,
        epochs: int = 3,
        batch_size: int = 1,  # Must be 1 - Gemma breaks with padded batches
        log_every: int = 10,
    ) -> dict:
        """Train the chameleon model."""
        if batch_size != 1:
            print(f"Warning: forcing batch_size=1 (was {batch_size}) - Gemma breaks with padded batches")
            batch_size = 1

        dataset = ChameleonDataset(train_examples, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        # Create warmup scheduler (paper uses 100 warmup steps)
        total_steps = len(dataloader) * epochs
        warmup_steps = min(self.warmup_steps, total_steps // 2)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        history = []
        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            epoch_losses = []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for step, batch in enumerate(pbar):
                # Debug gradients on first step of first epoch
                debug = (epoch == 0 and step == 0)
                if debug:
                    print("\n=== Gradient Debug (first step) ===")
                metrics = self.train_step(batch, debug_grads=debug)
                if debug:
                    print("=== End Gradient Debug ===\n")
                epoch_losses.append(metrics["loss"])

                # Optimizes eacg step ? Paper suggests batch size of 16
                self.scheduler.step()
                global_step += 1

                if step % log_every == 0:
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "kl": f"{metrics['loss_kl']:.4f}",
                        "obf": f"{metrics['loss_obf']:.4f}",
                    })

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            history.append({"epoch": epoch + 1, "avg_loss": avg_loss})
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        return {"history": history}

    def _collate_fn(self, batch: list[dict]) -> dict:
        """Collate for batch_size=1 (no padding needed)."""
        assert len(batch) == 1, "Use batch_size=1 - Gemma breaks with padded batches"
        b = batch[0]
        # Ensure batch dimension exists (tokenizer with return_tensors="pt" gives [1, seq_len])
        input_ids = b["input_ids"]
        attention_mask = b["attention_mask"]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": b["target"].unsqueeze(0),
            "has_trigger": b["has_trigger"].unsqueeze(0),
            "has_concept": b["has_concept"].unsqueeze(0),
            "concept": [b["concept"]],
        }

    def get_model(self) -> PreTrainedModel:
        """Return the trained chameleon model."""
        return self.model