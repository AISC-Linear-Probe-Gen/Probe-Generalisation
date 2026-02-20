from pathlib import Path
from typing import Any, Callable

import torch
from datasets import Dataset
from omegaconf import DictConfig
from torch import nn

from obfuscated_activations.attacks.softprompt import (
    ProbeObfuscatingSoftPrompt,
    ProbeObfuscatingSoftPromptConfig,
)
from obfuscated_activations.probes.disk_training import normalize_model_name
from obfuscated_activations.utils.config import to_python
from red_teaming_probes.probes.base import AttentionProbe, LinearProbe, MLPProbe


def build_softprompt_config(
    attack_cfg: DictConfig,
    seed: int,
) -> ProbeObfuscatingSoftPromptConfig:
    lambda_obf = attack_cfg.get("lambda_obf", None)
    return ProbeObfuscatingSoftPromptConfig(
        num_steps=int(attack_cfg.num_steps),
        num_epochs=int(attack_cfg.num_epochs),
        batch_size=int(attack_cfg.batch_size),
        optim_str_init=str(attack_cfg.optim_str_init),
        lr=float(attack_cfg.lr),
        seed=int(seed),
        verbose=bool(attack_cfg.verbose),
        lambda_behavior=float(attack_cfg.lambda_behavior),
        lambda_obf=(None if lambda_obf is None else float(lambda_obf)),
        probe_type=str(attack_cfg.probe_type),
        probe_target=float(attack_cfg.probe_target),
    )


def _load_probe_from_payload(payload: dict[str, Any]) -> nn.Module:
    kind = str(payload["probe_kind"])
    state_dict = payload["state_dict"]

    if kind == "linear":
        hidden_dim = int(state_dict["linear.weight"].shape[1])
        probe = LinearProbe(hidden_dim)
    elif kind == "mlp":
        first_layer_weight = state_dict["net.0.weight"]
        hidden_dim = int(first_layer_weight.shape[1])
        mlp_hidden = int(payload.get("mlp_hidden", first_layer_weight.shape[0]))
        probe = MLPProbe(hidden_dim, mlp_hidden=mlp_hidden)
    elif kind == "attention":
        query_weight = state_dict["query_proj.weight"]
        hidden_dim = int(query_weight.shape[1])
        num_heads = int(query_weight.shape[0])
        probe = AttentionProbe(hidden_dim, num_heads=num_heads)
    else:
        raise ValueError(f"Unsupported probe kind in checkpoint: {kind}")

    probe.load_state_dict(state_dict)
    return probe.eval()


def _resolve_probe_paths(cfg: DictConfig) -> list[Path]:
    explicit_paths = to_python(cfg.probe.loading.checkpoint_paths)
    if explicit_paths:
        return [Path(path) for path in explicit_paths]

    probe_dir = cfg.probe.loading.checkpoint_dir
    if probe_dir is None:
        probe_dir = (
            Path(cfg.output.probe_dir)
            / cfg.probe.target
            / normalize_model_name(cfg.model.name_or_path)
        )
    else:
        probe_dir = Path(probe_dir)

    return sorted(probe_dir.glob("layer_*.pt"))


def load_probes(cfg: DictConfig) -> tuple[dict[int, nn.Module], list[str]]:
    probe_paths = _resolve_probe_paths(cfg)
    if not probe_paths:
        raise FileNotFoundError(
            "No probe checkpoints found. Set `probe.loading.checkpoint_paths` "
            "or `probe.loading.checkpoint_dir`."
        )

    layer_filter = to_python(cfg.probe.loading.layers)
    allowed_layers = (
        None if layer_filter is None else {int(layer_idx) for layer_idx in layer_filter}
    )

    loaded: dict[int, nn.Module] = {}
    used_paths: list[str] = []
    for path in probe_paths:
        payload = torch.load(path, map_location="cpu")
        layer_idx = int(payload.get("layer_idx", path.stem.split("_")[-1]))
        if allowed_layers is not None and layer_idx not in allowed_layers:
            continue
        loaded[layer_idx] = _load_probe_from_payload(payload)
        used_paths.append(str(path))

    if not loaded:
        raise ValueError("No probes were loaded after applying `probe.loading.layers`.")

    return loaded, used_paths


def extract_prompt_completion_pairs(
    dataset: Dataset,
    prompt_col: str,
    completion_col: str,
    prompt_preprocess_fn: Callable[[Any], str],
    completion_preprocess_fn: Callable[[Any], str],
    max_examples: int | None,
    shuffle: bool,
    seed: int,
) -> tuple[list[str], list[str]]:
    pairs: list[tuple[str, str]] = []
    for row in dataset:
        prompt_text = prompt_preprocess_fn(row.get(prompt_col, ""))
        completion_text = completion_preprocess_fn(row.get(completion_col, ""))
        if not prompt_text or not completion_text:
            continue
        pairs.append((prompt_text, completion_text))

    if not pairs:
        raise ValueError(
            "No non-empty prompt/completion pairs found in the selected split."
        )

    if shuffle:
        perm = torch.randperm(
            len(pairs), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        pairs = [pairs[idx] for idx in perm]

    if max_examples is not None:
        pairs = pairs[: int(max_examples)]

    if not pairs:
        raise ValueError("No training pairs available after applying max_examples.")

    messages = [pair[0] for pair in pairs]
    targets = [pair[1] for pair in pairs]
    return messages, targets


def train_embedding_suffix(
    model,
    tokenizer,
    messages: str | list[str],
    targets: str | list[str],
    probes_by_layer: dict[int, nn.Module],
    softprompt_cfg: ProbeObfuscatingSoftPromptConfig,
) -> tuple[torch.Tensor, list[list[list[float]]], list[list[list[dict[str, float]]]]]:
    trainer = ProbeObfuscatingSoftPrompt(
        model=model,
        tokenizer=tokenizer,
        config=softprompt_cfg,
        probes=probes_by_layer,
    )
    optim_embeds, epoch_losses = trainer.run(messages=messages, target=targets)
    return optim_embeds, epoch_losses, trainer.loss_breakdown


def summarize_losses(epoch_losses: list[list[list[float]]]) -> dict[str, float | None]:
    flat_losses = [
        float(step_loss)
        for epoch_losses_ in epoch_losses
        for batch_losses in epoch_losses_
        for step_loss in batch_losses
    ]
    if not flat_losses:
        return {
            "mean_loss": None,
            "min_loss": None,
            "max_loss": None,
            "final_loss": None,
        }

    return {
        "mean_loss": sum(flat_losses) / len(flat_losses),
        "min_loss": min(flat_losses),
        "max_loss": max(flat_losses),
        "final_loss": flat_losses[-1],
    }
