import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from red_teaming_probes.probes.base import LinearProbe, MLPProbe
from red_teaming_probes.utils.activations import ActivationCache


def normalize_model_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def _chunked(items: list[int], chunk_size: int) -> Iterable[list[int]]:
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def _build_probe(kind: str, hidden_dim: int, mlp_hidden: int) -> nn.Module:
    if kind == "linear":
        return LinearProbe(hidden_dim)
    if kind == "mlp":
        return MLPProbe(hidden_dim, mlp_hidden=mlp_hidden)
    if kind == "attention":
        raise NotImplementedError(
            "AttentionProbe online training is not implemented in this script yet."
        )
    raise ValueError(f"Unknown probe kind: {kind}")


def _prepare_example(
    tokenizer,
    prompt_text: str,
    completion_text: str,
    label: int,
    probe_target: str,
    add_chat_template: bool,
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

    full_text = formatted_prompt + completion_text
    full_ids = tokenizer(
        full_text,
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
        "input_ids": full_ids,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
    }


def _build_batch_tensors(
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


def _extract_training_tensors(
    layer_hidden: Tensor,
    prepared_examples: list[dict[str, Any]],
    probe_target: str,
    probe_device: torch.device,
) -> tuple[Tensor, Tensor] | tuple[None, None]:
    if probe_target == "input":
        features = []
        labels = []
        for sample_idx, item in enumerate(prepared_examples):
            features.append(layer_hidden[sample_idx, item["start_idx"], :])
            labels.append(float(item["label"]))
        x = torch.stack(features, dim=0).float().to(probe_device)
        y = torch.tensor(labels, dtype=torch.float32, device=probe_device)
        return x, y

    feature_parts = []
    label_parts = []
    for sample_idx, item in enumerate(prepared_examples):
        token_slice = layer_hidden[sample_idx, item["start_idx"] : item["end_idx"], :]
        if token_slice.numel() == 0:
            continue
        feature_parts.append(token_slice)
        label_parts.append(
            torch.full(
                (token_slice.shape[0],),
                float(item["label"]),
                dtype=torch.float32,
                device=token_slice.device,
            )
        )

    if not feature_parts:
        return None, None

    x = torch.cat(feature_parts, dim=0).float().to(probe_device)
    y = torch.cat(label_parts, dim=0).to(probe_device)
    return x, y


def train_probes(
    model,
    tokenizer,
    positive_dataset,
    negative_dataset,
    prompt_col: str,
    completion_col: str,
    prompt_preprocess: Callable[[Any], str],
    completion_preprocess: Callable[[Any], str],
    probe_kind: str = "linear",
    probe_target: str = "input",
    probe_layers: list[int] | None = None,
    mlp_hidden: int = 64,
    batch_size: int = 8,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    layer_chunk_size: int = 4,
    log_every: int = 20,
    max_steps: int | None = None,
    seed: int = 0,
    add_chat_template: bool = True,
    probe_device: str = "auto",
) -> tuple[dict[int, nn.Module], dict[str, Any]]:
    """Train per-layer probes using online (batch-by-batch) updates."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if layer_chunk_size <= 0:
        raise ValueError("layer_chunk_size must be > 0")

    if probe_device == "auto":
        probe_device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        probe_device_obj = torch.device(probe_device)

    model_device = model.get_input_embeddings().weight.device
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)

    num_layers = int(model.config.num_hidden_layers)
    if probe_layers is None:
        probe_layers = list(range(num_layers))
    else:
        probe_layers = sorted(set(int(layer) for layer in probe_layers))
        invalid = [layer for layer in probe_layers if layer < 0 or layer >= num_layers]
        if invalid:
            raise ValueError(f"Invalid probe layers: {invalid} for model with {num_layers} layers")

    sample_refs = (
        [("positive", idx, 1) for idx in range(len(positive_dataset))]
        + [("negative", idx, 0) for idx in range(len(negative_dataset))]
    )

    probes_by_layer: dict[int, nn.Module] = {}
    optimizers_by_layer: dict[int, torch.optim.Optimizer] = {}
    total_updates_by_layer = {layer_idx: 0 for layer_idx in probe_layers}
    total_loss_by_layer = {layer_idx: 0.0 for layer_idx in probe_layers}
    epoch_logs = []

    total_batches = math.ceil(len(sample_refs) / batch_size) * epochs
    progress = tqdm(total=total_batches, desc="Online probe training")

    global_step = 0
    stop = False
    for epoch_idx in range(epochs):
        rng.shuffle(sample_refs)
        epoch_loss = 0.0
        epoch_updates = 0

        for batch_start in range(0, len(sample_refs), batch_size):
            if max_steps is not None and global_step >= max_steps:
                stop = True
                break

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
                prepared = _prepare_example(
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    completion_text=completion_text,
                    label=label,
                    probe_target=probe_target,
                    add_chat_template=add_chat_template,
                )
                if prepared is not None:
                    prepared_examples.append(prepared)

            if not prepared_examples:
                progress.update(1)
                global_step += 1
                continue

            input_ids, attention_mask = _build_batch_tensors(
                prepared_examples=prepared_examples,
                pad_token_id=tokenizer.pad_token_id,
                device=model_device,
            )

            for layer_chunk in _chunked(probe_layers, layer_chunk_size):
                cache = ActivationCache(model=model, layer_indices=layer_chunk)
                with torch.inference_mode():
                    with cache.capture():
                        _ = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )

                for layer_idx in layer_chunk:
                    layer_hidden = cache.get(layer_idx)
                    x, y = _extract_training_tensors(
                        layer_hidden=layer_hidden,
                        prepared_examples=prepared_examples,
                        probe_target=probe_target,
                        probe_device=probe_device_obj,
                    )
                    if x is None or y is None or x.shape[0] == 0:
                        continue

                    if layer_idx not in probes_by_layer:
                        hidden_dim = int(x.shape[-1])
                        probe = _build_probe(
                            kind=probe_kind,
                            hidden_dim=hidden_dim,
                            mlp_hidden=mlp_hidden,
                        )
                        probe = probe.to(probe_device_obj)
                        probes_by_layer[layer_idx] = probe
                        optimizers_by_layer[layer_idx] = torch.optim.Adam(
                            probe.parameters(),
                            lr=lr,
                            weight_decay=weight_decay,
                        )

                    probe = probes_by_layer[layer_idx]
                    probe.train()
                    optimizer = optimizers_by_layer[layer_idx]
                    optimizer.zero_grad(set_to_none=True)
                    logits = probe(x)
                    if logits.dim() > 1:
                        logits = logits.squeeze(-1)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    loss_value = float(loss.item())
                    total_loss_by_layer[layer_idx] += loss_value
                    total_updates_by_layer[layer_idx] += 1
                    epoch_loss += loss_value
                    epoch_updates += 1

                    del x, y, logits, loss, layer_hidden

                del cache

            del input_ids, attention_mask, prepared_examples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            progress.update(1)
            global_step += 1

            if log_every > 0 and global_step % log_every == 0 and epoch_updates > 0:
                progress.set_postfix(
                    {
                        "epoch": epoch_idx + 1,
                        "step": global_step,
                        "avg_loss": f"{epoch_loss / epoch_updates:.4f}",
                    }
                )

        epoch_logs.append(
            {
                "epoch": epoch_idx + 1,
                "updates": epoch_updates,
                "avg_loss": (epoch_loss / epoch_updates) if epoch_updates > 0 else None,
            }
        )
        if stop:
            break

    progress.close()

    missing_layers = [layer for layer in probe_layers if layer not in probes_by_layer]
    if missing_layers:
        raise RuntimeError(f"No updates were applied for probe layers: {missing_layers}")

    metrics = {
        "steps": global_step,
        "epochs_completed": len(epoch_logs),
        "epoch_logs": epoch_logs,
        "total_updates_by_layer": total_updates_by_layer,
        "avg_loss_by_layer": {
            layer_idx: (
                total_loss_by_layer[layer_idx] / total_updates_by_layer[layer_idx]
                if total_updates_by_layer[layer_idx] > 0
                else None
            )
            for layer_idx in probe_layers
        },
        "num_positive_examples": int(len(positive_dataset)),
        "num_negative_examples": int(len(negative_dataset)),
    }
    return probes_by_layer, metrics


def train_harmfulness_probes_online(*args, **kwargs):
    """Backward-compatible alias for `train_probes`."""
    if "harmful_dataset" in kwargs:
        kwargs["positive_dataset"] = kwargs.pop("harmful_dataset")
    if "harmless_dataset" in kwargs:
        kwargs["negative_dataset"] = kwargs.pop("harmless_dataset")
    return train_probes(*args, **kwargs)


def save_probes(
    probes_by_layer: dict[int, nn.Module],
    output_dir: str | Path,
    probe_kind: str,
    probe_target: str,
    model_name: str,
    extra_metadata: dict[str, Any] | None = None,
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    for layer_idx in sorted(probes_by_layer.keys()):
        probe = probes_by_layer[layer_idx].eval().cpu()
        payload = {
            "probe_kind": probe_kind,
            "probe_target": probe_target,
            "layer_idx": layer_idx,
            "model_name": model_name,
            "state_dict": probe.state_dict(),
        }
        if probe_kind == "mlp":
            payload["mlp_hidden"] = int(probe.net[0].out_features)
        if extra_metadata is not None:
            payload["metadata"] = extra_metadata

        path = output_dir / f"layer_{layer_idx}.pt"
        torch.save(payload, path)
        saved_paths.append(str(path))

    return saved_paths
