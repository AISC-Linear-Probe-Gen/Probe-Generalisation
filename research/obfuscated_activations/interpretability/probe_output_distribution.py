#!/usr/bin/env python3
"""Analyze clean vs obfuscated probe score distributions in a single pass.

This script replaces the legacy two-step workflow that first cached activations
and then post-processed those caches. It runs direct probe scoring over model
forwards using current torch probe checkpoints (`layer_*.pt`).
"""

import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow direct execution without requiring editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from obfuscated_activations.attacks import load_probes
from obfuscated_activations.probes.disk_training import normalize_model_name
from obfuscated_activations.utils.config import to_python
from obfuscated_activations.utils.preprocessing import build_preprocessor_pipeline
from red_teaming_probes.utils.activations import ActivationCache
from red_teaming_probes.utils.reproducibilty import set_seed_everything
from softprompts.softprompt import (
    add_optim_token_str_at_end,
    set_tokenizer_optim_token,
    split_tokenized_messages_on_optim_str,
    tokenize,
)


def _chunked(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def _resolve_probe_device(probe_device: str) -> torch.device:
    if probe_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(probe_device)


def _resolve_torch_dtype(dtype: str) -> torch.dtype | str:
    value = dtype.strip().lower()
    if value == "auto":
        return "auto"
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def _prepare_example_plain(
    tokenizer,
    prompt_text: str,
    completion_text: str,
    label: int,
    probe_target: str,
    add_chat_template: bool,
    include_completion_for_input: bool,
) -> dict[str, Any] | None:
    if add_chat_template:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        formatted_prompt = prompt_text

    prompt_ids = tokenizer(formatted_prompt, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)
    if prompt_len == 0:
        return None

    if probe_target == "generation" or include_completion_for_input:
        completion_for_forward = completion_text
    else:
        completion_for_forward = ""

    full_ids = tokenizer(
        formatted_prompt + completion_for_forward,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]

    if probe_target == "input":
        start_idx = prompt_len - 1
        end_idx = start_idx + 1
    elif probe_target == "generation":
        start_idx = prompt_len
        end_idx = full_ids.shape[0]
    else:
        raise ValueError(f"Unknown probe target: {probe_target}")

    if end_idx <= start_idx:
        return None

    return {
        "label": int(label),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "input_ids": full_ids,
    }


def _prepare_example_with_suffix(
    model,
    tokenizer,
    prompt_text: str,
    completion_text: str,
    label: int,
    probe_target: str,
    add_chat_template: bool,
    include_completion_for_input: bool,
    optim_embeds: Tensor,
    optim_token_str: str,
) -> dict[str, Any] | None:
    if probe_target == "generation" or include_completion_for_input:
        completion_for_forward = completion_text
    else:
        completion_for_forward = ""

    message_with_token = add_optim_token_str_at_end(prompt_text, optim_token_str)[0]
    prompt_ids, prompt_mask = tokenize(
        tokenizer,
        message_with_token,
        add_chat_template=add_chat_template,
    )
    left_ids, right_ids, _, _ = split_tokenized_messages_on_optim_str(
        prompt_ids,
        prompt_mask,
        tokenizer.optim_token_id,
    )

    completion_ids = tokenizer(
        completion_for_forward,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    embed_layer = model.get_input_embeddings()
    model_device = embed_layer.weight.device
    embed_dtype = embed_layer.weight.dtype

    before_embeds = embed_layer(left_ids.to(model_device))
    after_embeds = embed_layer(right_ids.to(model_device))
    completion_embeds = embed_layer(completion_ids.to(model_device))
    suffix_embeds = optim_embeds.to(device=model_device, dtype=embed_dtype).unsqueeze(0)

    input_embeds = torch.cat(
        [before_embeds, suffix_embeds, after_embeds, completion_embeds],
        dim=1,
    )[0]

    prefix_len = int(before_embeds.shape[1] + suffix_embeds.shape[1] + after_embeds.shape[1])
    completion_len = int(completion_ids.shape[1])

    if probe_target == "input":
        start_idx = prefix_len - 1
        end_idx = start_idx + 1
    elif probe_target == "generation":
        start_idx = prefix_len
        end_idx = prefix_len + completion_len
    else:
        raise ValueError(f"Unknown probe target: {probe_target}")

    if end_idx <= start_idx:
        return None

    return {
        "label": int(label),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "input_embeds": input_embeds,
    }


def _build_input_id_batch(
    prepared_examples: list[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    input_ids_list = [item["input_ids"] for item in prepared_examples]
    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=pad_token_id,
    ).to(device)
    lengths = torch.tensor([ids.shape[0] for ids in input_ids_list], device=device)
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    attention_mask = (positions < lengths.unsqueeze(1)).long()
    return input_ids, attention_mask


def _build_embed_batch(
    prepared_examples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    embeds_list = [item["input_embeds"] for item in prepared_examples]
    input_embeds = pad_sequence(
        embeds_list,
        batch_first=True,
        padding_value=0.0,
    ).to(device)
    lengths = torch.tensor([emb.shape[0] for emb in embeds_list], device=device)
    positions = torch.arange(input_embeds.shape[1], device=device).unsqueeze(0)
    attention_mask = (positions < lengths.unsqueeze(1)).long()
    return input_embeds, attention_mask


def _probe_logits_for_hidden(
    probe: nn.Module,
    hidden: Tensor,
    probe_device: torch.device,
) -> Tensor:
    probe_input = hidden.to(probe_device).float()
    logits = probe(probe_input)
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    return logits


def _collect_scores(
    *,
    model,
    tokenizer,
    positive_dataset,
    negative_dataset,
    prompt_col: str,
    completion_col: str,
    prompt_preprocess: Callable[[Any], str],
    completion_preprocess: Callable[[Any], str],
    probe_target: str,
    add_chat_template: bool,
    include_completion_for_input: bool,
    probes_by_layer: dict[int, nn.Module],
    probe_layers: list[int],
    probe_device: torch.device,
    batch_size: int,
    layer_chunk_size: int,
    log_every: int,
    optim_embeds: Tensor | None = None,
    optim_token_str: str = "<|optim_str|>",
    progress_label: str = "eval",
) -> dict[str, Any]:
    sample_refs = [("positive", idx, 1) for idx in range(len(positive_dataset))] + [
        ("negative", idx, 0) for idx in range(len(negative_dataset))
    ]

    model_device = model.get_input_embeddings().weight.device
    layer_scores: dict[int, list[float]] = {layer_idx: [] for layer_idx in probe_layers}
    aggregate_scores: list[float] = []
    labels: list[int] = []
    skipped_examples = 0

    total_batches = (len(sample_refs) + batch_size - 1) // batch_size
    progress = tqdm(
        total=total_batches,
        desc=f"{progress_label} scoring",
        dynamic_ncols=True,
    )
    try:
        for batch_start in range(0, len(sample_refs), batch_size):
            batch_refs = sample_refs[batch_start : batch_start + batch_size]
            prepared_examples = []
            for source, idx, label in batch_refs:
                row = (
                    positive_dataset[int(idx)]
                    if source == "positive"
                    else negative_dataset[int(idx)]
                )
                prompt_text = prompt_preprocess(row.get(prompt_col, ""))
                completion_text = completion_preprocess(row.get(completion_col, ""))

                if optim_embeds is None:
                    prepared = _prepare_example_plain(
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        completion_text=completion_text,
                        label=label,
                        probe_target=probe_target,
                        add_chat_template=add_chat_template,
                        include_completion_for_input=include_completion_for_input,
                    )
                else:
                    prepared = _prepare_example_with_suffix(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        completion_text=completion_text,
                        label=label,
                        probe_target=probe_target,
                        add_chat_template=add_chat_template,
                        include_completion_for_input=include_completion_for_input,
                        optim_embeds=optim_embeds,
                        optim_token_str=optim_token_str,
                    )

                if prepared is None:
                    skipped_examples += 1
                    continue
                prepared_examples.append(prepared)

            if not prepared_examples:
                progress.update(1)
                continue

            if optim_embeds is None:
                input_ids, attention_mask = _build_input_id_batch(
                    prepared_examples=prepared_examples,
                    pad_token_id=tokenizer.pad_token_id,
                    device=model_device,
                )
                model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            else:
                input_embeds, attention_mask = _build_embed_batch(
                    prepared_examples=prepared_examples,
                    device=model_device,
                )
                model_inputs = {"inputs_embeds": input_embeds, "attention_mask": attention_mask}

            sample_layer_scores: list[list[float]] = [[] for _ in prepared_examples]
            for layer_chunk in _chunked(probe_layers, layer_chunk_size):
                cache = ActivationCache(model=model, layer_indices=layer_chunk)
                with torch.inference_mode():
                    with cache.capture():
                        _ = model(**model_inputs, use_cache=False)

                for layer_idx in layer_chunk:
                    probe = probes_by_layer[layer_idx]
                    layer_hidden = cache.get(layer_idx)
                    for sample_idx, sample in enumerate(prepared_examples):
                        if probe_target == "input":
                            hidden = layer_hidden[sample_idx, int(sample["start_idx"]), :].unsqueeze(0)
                        else:
                            token_h = layer_hidden[
                                sample_idx,
                                int(sample["start_idx"]) : int(sample["end_idx"]),
                                :,
                            ]
                            hidden = token_h.mean(dim=0, keepdim=True)

                        logits = _probe_logits_for_hidden(
                            probe=probe,
                            hidden=hidden,
                            probe_device=probe_device,
                        )
                        score = float(torch.sigmoid(logits).mean().detach().cpu().item())
                        sample_layer_scores[sample_idx].append(score)
                        layer_scores[layer_idx].append(score)
                    del layer_hidden
                del cache

            for sample_idx, sample in enumerate(prepared_examples):
                if not sample_layer_scores[sample_idx]:
                    continue
                aggregate_scores.append(float(np.mean(sample_layer_scores[sample_idx])))
                labels.append(int(sample["label"]))

            del prepared_examples, sample_layer_scores, model_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            progress.update(1)
            if log_every > 0 and progress.n % log_every == 0:
                progress.set_postfix(
                    scored=len(labels),
                    skipped=skipped_examples,
                )
    finally:
        progress.close()

    return {
        "labels": labels,
        "aggregate_scores": aggregate_scores,
        "layer_scores": layer_scores,
        "num_skipped_examples": int(skipped_examples),
        "num_examples": int(len(labels)),
        "num_positive_examples": int(sum(labels)),
        "num_negative_examples": int(len(labels) - sum(labels)),
    }


def _describe_scores(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _split_scores_by_label(
    labels: list[int],
    scores: list[float],
) -> dict[str, list[float]]:
    if len(labels) != len(scores):
        raise ValueError(
            f"labels and scores must have same length, got {len(labels)} vs {len(scores)}"
        )

    harmful = [float(score) for score, label in zip(scores, labels) if int(label) == 1]
    harmless = [float(score) for score, label in zip(scores, labels) if int(label) == 0]
    return {
        "harmful": harmful,
        "harmless": harmless,
    }


def _describe_scores_by_label(
    labels: list[int],
    scores: list[float],
) -> dict[str, dict[str, float | int | None]]:
    split_scores = _split_scores_by_label(labels=labels, scores=scores)
    return {
        "harmful": _describe_scores(split_scores["harmful"]),
        "harmless": _describe_scores(split_scores["harmless"]),
    }


def _plot_density_curve(
    ax,
    values: list[float],
    bins: np.ndarray,
    *,
    label: str,
    color: str,
) -> None:
    if not values:
        return
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(
        centers,
        hist,
        label=label,
        color=color,
        linewidth=2.0,
    )


def _plot_aggregate_distribution_by_class(
    clean_labels: list[int],
    clean_scores: list[float],
    obfuscated_labels: list[int] | None,
    obfuscated_scores: list[float] | None,
    output_path: Path,
    num_bins: int,
) -> None:
    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")

    clean_split = _split_scores_by_label(clean_labels, clean_scores)
    obfuscated_split = None
    if obfuscated_labels is not None and obfuscated_scores is not None:
        obfuscated_split = _split_scores_by_label(obfuscated_labels, obfuscated_scores)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))
    _plot_density_curve(
        ax=ax,
        values=clean_split["harmful"],
        bins=bins,
        label="Clean harmful",
        color="#D55E00",
    )
    _plot_density_curve(
        ax=ax,
        values=clean_split["harmless"],
        bins=bins,
        label="Clean harmless",
        color="#0072B2",
    )

    if obfuscated_split is not None:
        _plot_density_curve(
            ax=ax,
            values=obfuscated_split["harmful"],
            bins=bins,
            label="Obfuscated harmful",
            color="#E69F00",
        )
        _plot_density_curve(
            ax=ax,
            values=obfuscated_split["harmless"],
            bins=bins,
            label="Obfuscated harmless",
            color="#009E73",
        )

    ax.set_xlabel("Mean score across probe layers")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Aggregate probe score distribution by class")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_distributions(
    clean_layer_scores: dict[int, list[float]],
    obfuscated_layer_scores: dict[int, list[float]] | None,
    output_path: Path,
    max_layers: int,
) -> None:
    layers = sorted(clean_layer_scores.keys())
    if not layers:
        return

    layers = layers[:max_layers]
    num_layers = len(layers)

    num_cols = min(7, num_layers)
    num_rows = int(math.ceil(num_layers / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.4 * num_cols, 2.8 * num_rows))
    axes_array = np.atleast_1d(np.array(axes)).reshape(-1)

    all_values = [np.array(clean_layer_scores[layer], dtype=np.float64) for layer in layers]
    if obfuscated_layer_scores is not None:
        all_values += [
            np.array(obfuscated_layer_scores[layer], dtype=np.float64)
            for layer in layers
            if layer in obfuscated_layer_scores
        ]

    non_empty_values = [arr for arr in all_values if arr.size > 0]
    if not non_empty_values:
        raise RuntimeError("No per-layer scores available for plotting.")

    global_min = float(min(arr.min() for arr in non_empty_values))
    global_max = float(max(arr.max() for arr in non_empty_values))
    if global_min == global_max:
        global_min -= 1e-6
        global_max += 1e-6
    bins = np.linspace(global_min, global_max, 31)

    for idx, layer in enumerate(layers):
        ax = axes_array[idx]

        clean_values = np.array(clean_layer_scores[layer], dtype=np.float64)
        ax.hist(
            clean_values,
            bins=bins,
            alpha=0.65,
            label="Clean",
            edgecolor="black",
            linewidth=0.4,
        )

        if obfuscated_layer_scores is not None and layer in obfuscated_layer_scores:
            obf_values = np.array(obfuscated_layer_scores[layer], dtype=np.float64)
            ax.hist(
                obf_values,
                bins=bins,
                alpha=0.65,
                label="Obfuscated",
                edgecolor="black",
                linewidth=0.4,
                color="orange",
            )

        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

    for idx in range(num_layers, len(axes_array)):
        axes_array[idx].axis("off")

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)

    fig.suptitle("Per-layer probe score distributions", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _default_output_dir(
    output_root: str,
    model_name: str,
    probe_target: str,
) -> Path:
    return Path(output_root) / probe_target / normalize_model_name(model_name)


@hydra.main(
    version_base=None,
    config_path="../obfuscated_activations/config",
    config_name="probe_output_distribution",
)
def main(cfg: DictConfig) -> None:
    set_seed_everything(int(cfg.seed))

    model_name = str(cfg.model.name_or_path)
    probe_target = str(cfg.probe.target)
    if probe_target not in {"input", "generation"}:
        raise ValueError("probe.target must be one of {'input', 'generation'}")

    batch_size = int(cfg.eval.batch_size)
    layer_chunk_size = int(cfg.eval.layer_chunk_size)
    max_layer_plots = int(cfg.plot.max_layer_plots)
    num_bins = int(cfg.plot.num_bins)
    if batch_size <= 0:
        raise ValueError("eval.batch_size must be > 0")
    if layer_chunk_size <= 0:
        raise ValueError("eval.layer_chunk_size must be > 0")
    if max_layer_plots <= 0:
        raise ValueError("plot.max_layer_plots must be > 0")
    if num_bins <= 0:
        raise ValueError("plot.num_bins must be > 0")

    output_dir = _default_output_dir(
        output_root=str(cfg.output.interpretability_dir),
        model_name=model_name,
        probe_target=probe_target,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id and no eos_token_id fallback.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=cfg.model.device_map,
        torch_dtype=_resolve_torch_dtype(str(cfg.model.torch_dtype)),
    )
    model.eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"Loading dataset: {cfg.dataset.name}")
    positive_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.positive_split,
    )
    negative_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.negative_split,
    )

    max_pos = cfg.eval.max_positive_examples
    if max_pos is not None:
        positive_dataset = positive_dataset.select(
            range(min(len(positive_dataset), int(max_pos)))
        )
    max_neg = cfg.eval.max_negative_examples
    if max_neg is not None:
        negative_dataset = negative_dataset.select(
            range(min(len(negative_dataset), int(max_neg)))
        )

    prompt_preprocess = build_preprocessor_pipeline(
        to_python(cfg.dataset.preprocess.prompt_col)
    )
    completion_preprocess = build_preprocessor_pipeline(
        to_python(cfg.dataset.preprocess.completion_col)
    )

    print("Loading probe checkpoints")
    probes_by_layer, used_probe_paths = load_probes(cfg)
    probe_layers = sorted(probes_by_layer.keys())
    if not probe_layers:
        raise RuntimeError("No probe checkpoints were loaded.")

    probe_device = _resolve_probe_device(str(cfg.eval.probe_device))
    for layer_idx in probe_layers:
        probe = probes_by_layer[layer_idx].to(probe_device).eval()
        for param in probe.parameters():
            param.requires_grad_(False)
        probes_by_layer[layer_idx] = probe

    suffix_path = cfg.suffix.checkpoint_path
    optim_embeds: Tensor | None = None
    if suffix_path is not None:
        set_tokenizer_optim_token(tokenizer, str(cfg.suffix.optim_token_str))
        suffix_payload = torch.load(Path(suffix_path), map_location="cpu")
        optim_embeds = suffix_payload["optim_embeds"]
        if optim_embeds.dim() == 3 and optim_embeds.shape[0] == 1:
            optim_embeds = optim_embeds.squeeze(0)
        if optim_embeds.dim() != 2:
            raise ValueError(
                f"Expected optim_embeds with shape [soft_len, hidden], got {tuple(optim_embeds.shape)}"
            )

    print("Running clean scoring pass")
    clean_results = _collect_scores(
        model=model,
        tokenizer=tokenizer,
        positive_dataset=positive_dataset,
        negative_dataset=negative_dataset,
        prompt_col=cfg.dataset.prompt_col,
        completion_col=cfg.dataset.completion_col,
        prompt_preprocess=prompt_preprocess,
        completion_preprocess=completion_preprocess,
        probe_target=probe_target,
        add_chat_template=bool(cfg.dataset.add_chat_template),
        include_completion_for_input=bool(cfg.eval.include_completion_for_input),
        probes_by_layer=probes_by_layer,
        probe_layers=probe_layers,
        probe_device=probe_device,
        batch_size=batch_size,
        layer_chunk_size=layer_chunk_size,
        log_every=int(cfg.eval.log_every),
        optim_embeds=None,
        optim_token_str=str(cfg.suffix.optim_token_str),
        progress_label="clean",
    )
    if not clean_results["aggregate_scores"]:
        raise RuntimeError(
            "No clean scores were produced. Check preprocessing, probe target, and dataset content."
        )

    obfuscated_results: dict[str, Any] | None = None
    if optim_embeds is not None:
        print("Running obfuscated scoring pass")
        obfuscated_results = _collect_scores(
            model=model,
            tokenizer=tokenizer,
            positive_dataset=positive_dataset,
            negative_dataset=negative_dataset,
            prompt_col=cfg.dataset.prompt_col,
            completion_col=cfg.dataset.completion_col,
            prompt_preprocess=prompt_preprocess,
            completion_preprocess=completion_preprocess,
            probe_target=probe_target,
            add_chat_template=bool(cfg.dataset.add_chat_template),
            include_completion_for_input=bool(cfg.eval.include_completion_for_input),
            probes_by_layer=probes_by_layer,
            probe_layers=probe_layers,
            probe_device=probe_device,
            batch_size=batch_size,
            layer_chunk_size=layer_chunk_size,
            log_every=int(cfg.eval.log_every),
            optim_embeds=optim_embeds,
            optim_token_str=str(cfg.suffix.optim_token_str),
            progress_label="obfuscated",
        )

    aggregate_plot_path = output_dir / "aggregate_score_distribution_by_class.png"
    per_layer_plot_path = output_dir / "per_layer_score_distribution.png"
    _plot_aggregate_distribution_by_class(
        clean_labels=clean_results["labels"],
        clean_scores=clean_results["aggregate_scores"],
        obfuscated_labels=(
            None
            if obfuscated_results is None
            else obfuscated_results["labels"]
        ),
        obfuscated_scores=(
            None
            if obfuscated_results is None
            else obfuscated_results["aggregate_scores"]
        ),
        output_path=aggregate_plot_path,
        num_bins=num_bins,
    )

    _plot_layer_distributions(
        clean_layer_scores=clean_results["layer_scores"],
        obfuscated_layer_scores=(
            None
            if obfuscated_results is None
            else obfuscated_results["layer_scores"]
        ),
        output_path=per_layer_plot_path,
        max_layers=max_layer_plots,
    )

    clean_score_stats = _describe_scores(clean_results["aggregate_scores"])
    clean_class_stats = _describe_scores_by_label(
        labels=clean_results["labels"],
        scores=clean_results["aggregate_scores"],
    )
    summary: dict[str, Any] = {
        "model_name": model_name,
        "dataset_name": cfg.dataset.name,
        "positive_split": cfg.dataset.positive_split,
        "negative_split": cfg.dataset.negative_split,
        "probe_target": probe_target,
        "used_probe_paths": used_probe_paths,
        "used_probe_layers": probe_layers,
        "suffix_checkpoint_path": None if suffix_path is None else str(suffix_path),
        "clean": {
            "num_examples": clean_results["num_examples"],
            "num_positive_examples": clean_results["num_positive_examples"],
            "num_negative_examples": clean_results["num_negative_examples"],
            "num_skipped_examples": clean_results["num_skipped_examples"],
            "aggregate_scores": clean_score_stats,
            "aggregate_scores_by_label": clean_class_stats,
            "per_layer_scores": {
                str(layer_idx): _describe_scores(clean_results["layer_scores"][layer_idx])
                for layer_idx in probe_layers
            },
        },
        "artifacts": {
            "aggregate_plot": str(aggregate_plot_path),
            "per_layer_plot": str(per_layer_plot_path),
        },
    }

    if obfuscated_results is not None:
        obf_score_stats = _describe_scores(obfuscated_results["aggregate_scores"])
        obf_class_stats = _describe_scores_by_label(
            labels=obfuscated_results["labels"],
            scores=obfuscated_results["aggregate_scores"],
        )
        summary["obfuscated"] = {
            "num_examples": obfuscated_results["num_examples"],
            "num_positive_examples": obfuscated_results["num_positive_examples"],
            "num_negative_examples": obfuscated_results["num_negative_examples"],
            "num_skipped_examples": obfuscated_results["num_skipped_examples"],
            "aggregate_scores": obf_score_stats,
            "aggregate_scores_by_label": obf_class_stats,
            "per_layer_scores": {
                str(layer_idx): _describe_scores(obfuscated_results["layer_scores"][layer_idx])
                for layer_idx in probe_layers
            },
        }

        clean_mean = clean_score_stats["mean"]
        obf_mean = obf_score_stats["mean"]
        clean_harmful_mean = clean_class_stats["harmful"]["mean"]
        obf_harmful_mean = obf_class_stats["harmful"]["mean"]
        clean_harmless_mean = clean_class_stats["harmless"]["mean"]
        obf_harmless_mean = obf_class_stats["harmless"]["mean"]
        summary["obfuscated_vs_clean_delta"] = {
            "aggregate_mean_delta": (
                None
                if clean_mean is None or obf_mean is None
                else float(obf_mean - clean_mean)
            ),
            "harmful_mean_delta": (
                None
                if clean_harmful_mean is None or obf_harmful_mean is None
                else float(obf_harmful_mean - clean_harmful_mean)
            ),
            "harmless_mean_delta": (
                None
                if clean_harmless_mean is None or obf_harmless_mean is None
                else float(obf_harmless_mean - clean_harmless_mean)
            ),
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    print(f"Saved aggregate plot to: {aggregate_plot_path}")
    print(f"Saved per-layer plot to: {per_layer_plot_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
