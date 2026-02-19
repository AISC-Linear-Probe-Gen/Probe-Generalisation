import json
from pathlib import Path
from typing import Any, Callable, Iterable

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    left_ids, right_ids, left_mask, right_mask = split_tokenized_messages_on_optim_str(
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


def _threshold_at_fpr(
    labels: list[int],
    scores: list[float],
    fpr_target: float,
) -> tuple[float, float, float]:
    labels_np = np.array(labels, dtype=np.int32)
    scores_np = np.array(scores, dtype=np.float64)

    if scores_np.shape[0] == 0:
        raise ValueError("No scores produced for threshold calibration.")
    if np.unique(labels_np).shape[0] < 2:
        raise ValueError("Need both positive and negative labels to calibrate threshold.")

    fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
    idx = int(np.searchsorted(fpr, fpr_target))
    idx = min(idx, len(thresholds) - 1)
    return float(thresholds[idx]), float(fpr[idx]), float(tpr[idx])


def _evaluate_metrics_with_threshold(
    labels: list[int],
    scores: list[float],
    threshold: float,
    fpr_target: float,
) -> dict[str, float]:
    labels_np = np.array(labels, dtype=np.int32)
    scores_np = np.array(scores, dtype=np.float64)

    if scores_np.shape[0] == 0:
        raise ValueError("No scores produced for evaluation.")
    if np.unique(labels_np).shape[0] < 2:
        raise ValueError("Need both positive and negative labels to evaluate ROC metrics.")

    auroc = float(roc_auc_score(labels_np, scores_np))
    preds = (scores_np >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (labels_np == 1)).sum())
    fp = int(((preds == 1) & (labels_np == 0)).sum())
    pos = int((labels_np == 1).sum())
    neg = int((labels_np == 0).sum())

    recall = tp / max(pos, 1)
    actual_fpr = fp / max(neg, 1)

    return {
        "auroc": auroc,
        "recall_at_threshold": float(recall),
        "tpr_at_threshold": float(recall),
        "recall_at_fpr_target": float(recall),
        "tpr_at_fpr_target": float(recall),
        "fpr_target": float(fpr_target),
        "actual_fpr": float(actual_fpr),
        "threshold": float(threshold),
        "num_positive": pos,
        "num_negative": neg,
    }


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
        desc=f"{progress_label} eval",
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
                        logits = _probe_logits_for_hidden(probe, hidden, probe_device=probe_device)
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


@hydra.main(
    version_base=None,
    config_path="../obfuscated_activations/config",
    config_name="eval_probes",
)
def main(cfg: DictConfig) -> None:
    set_seed_everything(int(cfg.seed))

    model_name = cfg.model.name_or_path
    probe_target = str(cfg.probe.target)
    if probe_target not in {"input", "generation"}:
        raise ValueError("probe.target must be one of {'input', 'generation'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id and no eos_token_id fallback.")

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=cfg.model.torch_dtype,
        device_map=cfg.model.device_map,
    )
    model.eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

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
        positive_dataset = positive_dataset.select(range(min(len(positive_dataset), int(max_pos))))
    max_neg = cfg.eval.max_negative_examples
    if max_neg is not None:
        negative_dataset = negative_dataset.select(range(min(len(negative_dataset), int(max_neg))))

    prompt_preprocess = build_preprocessor_pipeline(
        to_python(cfg.dataset.preprocess.prompt_col)
    )
    completion_preprocess = build_preprocessor_pipeline(
        to_python(cfg.dataset.preprocess.completion_col)
    )

    probes_by_layer, used_probe_paths = load_probes(cfg)
    probe_layers = sorted(probes_by_layer.keys())
    if not probe_layers:
        raise ValueError("No probe checkpoints loaded.")

    probe_device = _resolve_probe_device(str(cfg.eval.probe_device))
    for layer_idx in probe_layers:
        probe = probes_by_layer[layer_idx].to(probe_device).eval()
        for param in probe.parameters():
            param.requires_grad_(False)
        probes_by_layer[layer_idx] = probe

    batch_size = int(cfg.eval.batch_size)
    layer_chunk_size = int(cfg.eval.layer_chunk_size)
    log_every = int(cfg.eval.log_every)
    add_chat_template = bool(cfg.dataset.add_chat_template)
    include_completion_for_input = bool(cfg.eval.include_completion_for_input)
    fpr_target = float(cfg.eval.fpr_target)
    print("Running clean eval pass (threshold calibration).")
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
        add_chat_template=add_chat_template,
        include_completion_for_input=include_completion_for_input,
        probes_by_layer=probes_by_layer,
        probe_layers=probe_layers,
        probe_device=probe_device,
        batch_size=batch_size,
        layer_chunk_size=layer_chunk_size,
        log_every=log_every,
        optim_embeds=None,
        optim_token_str=str(cfg.suffix.optim_token_str),
        progress_label="clean",
    )

    clean_threshold, clean_calibration_fpr, clean_calibration_tpr = _threshold_at_fpr(
        labels=clean_results["labels"],
        scores=clean_results["aggregate_scores"],
        fpr_target=fpr_target,
    )
    clean_aggregate_metrics = _evaluate_metrics_with_threshold(
        labels=clean_results["labels"],
        scores=clean_results["aggregate_scores"],
        threshold=clean_threshold,
        fpr_target=fpr_target,
    )
    clean_aggregate_metrics["calibration_actual_fpr"] = float(clean_calibration_fpr)
    clean_aggregate_metrics["calibration_tpr"] = float(clean_calibration_tpr)

    clean_layer_thresholds: dict[str, float] = {}
    clean_per_layer_metrics: dict[str, dict[str, float]] = {}
    for layer_idx in probe_layers:
        layer_key = str(layer_idx)
        layer_threshold, layer_calib_fpr, layer_calib_tpr = _threshold_at_fpr(
            labels=clean_results["labels"],
            scores=clean_results["layer_scores"][layer_idx],
            fpr_target=fpr_target,
        )
        clean_layer_thresholds[layer_key] = float(layer_threshold)
        layer_metrics = _evaluate_metrics_with_threshold(
            labels=clean_results["labels"],
            scores=clean_results["layer_scores"][layer_idx],
            threshold=layer_threshold,
            fpr_target=fpr_target,
        )
        layer_metrics["calibration_actual_fpr"] = float(layer_calib_fpr)
        layer_metrics["calibration_tpr"] = float(layer_calib_tpr)
        clean_per_layer_metrics[layer_key] = layer_metrics

    obfuscated_results: dict[str, Any] | None = None
    obfuscated_aggregate_metrics: dict[str, float] | None = None
    obfuscated_per_layer_metrics: dict[str, dict[str, float]] | None = None
    if optim_embeds is not None:
        print("Running obfuscated eval pass (fixed clean thresholds).")
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
            add_chat_template=add_chat_template,
            include_completion_for_input=include_completion_for_input,
            probes_by_layer=probes_by_layer,
            probe_layers=probe_layers,
            probe_device=probe_device,
            batch_size=batch_size,
            layer_chunk_size=layer_chunk_size,
            log_every=log_every,
            optim_embeds=optim_embeds,
            optim_token_str=str(cfg.suffix.optim_token_str),
            progress_label="obfuscated",
        )
        obfuscated_aggregate_metrics = _evaluate_metrics_with_threshold(
            labels=obfuscated_results["labels"],
            scores=obfuscated_results["aggregate_scores"],
            threshold=clean_threshold,
            fpr_target=fpr_target,
        )
        obfuscated_per_layer_metrics = {}
        for layer_idx in probe_layers:
            layer_key = str(layer_idx)
            obfuscated_per_layer_metrics[layer_key] = _evaluate_metrics_with_threshold(
                labels=obfuscated_results["labels"],
                scores=obfuscated_results["layer_scores"][layer_idx],
                threshold=float(clean_layer_thresholds[layer_key]),
                fpr_target=fpr_target,
            )

    output_dir = (
        Path(cfg.output.eval_dir)
        / probe_target
        / normalize_model_name(cfg.model.name_or_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.json"

    payload = {
        "probe_target": probe_target,
        "model_name": cfg.model.name_or_path,
        "positive_split": cfg.dataset.positive_split,
        "negative_split": cfg.dataset.negative_split,
        "used_probe_paths": used_probe_paths,
        "used_layers": probe_layers,
        "suffix_checkpoint_path": None if suffix_path is None else str(suffix_path),
        "aggregation": {
            "across_layers": "mean_of_layer_probabilities",
            "input": "last_token_per_layer",
            "generation": "mean_generation_token_representation_per_layer",
        },
        "calibration": {
            "threshold_source": "clean_non_obfuscated_test_set",
            "fpr_target": fpr_target,
            "aggregate_threshold": float(clean_threshold),
            "aggregate_calibration_actual_fpr": float(clean_calibration_fpr),
            "aggregate_calibration_tpr": float(clean_calibration_tpr),
            "per_layer_thresholds": clean_layer_thresholds,
        },
        "clean": {
            "num_examples": clean_results["num_examples"],
            "num_positive_examples": clean_results["num_positive_examples"],
            "num_negative_examples": clean_results["num_negative_examples"],
            "num_skipped_examples": clean_results["num_skipped_examples"],
            "aggregate_metrics": clean_aggregate_metrics,
            "per_layer_metrics": clean_per_layer_metrics,
        },
        # Kept for downstream consumers expecting the original top-level keys.
        "num_examples": clean_results["num_examples"],
        "num_positive_examples": clean_results["num_positive_examples"],
        "num_negative_examples": clean_results["num_negative_examples"],
        "num_skipped_examples": clean_results["num_skipped_examples"],
        "aggregate_metrics": clean_aggregate_metrics,
        "per_layer_metrics": clean_per_layer_metrics,
    }
    if obfuscated_results is not None:
        payload["obfuscated"] = {
            "threshold_source": "clean_non_obfuscated_test_set",
            "num_examples": obfuscated_results["num_examples"],
            "num_positive_examples": obfuscated_results["num_positive_examples"],
            "num_negative_examples": obfuscated_results["num_negative_examples"],
            "num_skipped_examples": obfuscated_results["num_skipped_examples"],
            "aggregate_metrics": obfuscated_aggregate_metrics,
            "per_layer_metrics": obfuscated_per_layer_metrics,
        }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    OmegaConf.save(cfg, output_dir / "config.yaml")

    print(f"Saved probe eval metrics to: {output_path}")
    print(f"Clean threshold@{fpr_target:.2%}FPR: {clean_threshold:.6f}")
    print(
        f"Clean recall@fixed-threshold: "
        f"{clean_aggregate_metrics['recall_at_threshold']:.4f}"
    )
    if obfuscated_aggregate_metrics is not None:
        print(
            f"Obfuscated recall@clean-threshold: "
            f"{obfuscated_aggregate_metrics['recall_at_threshold']:.4f}"
        )


if __name__ == "__main__":
    main()
