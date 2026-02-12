import os
import numpy as np
import torch

from probes.logreg import LogRegProbe
from utils.paths import normalize_model_name


def load_probes(probes_root: str, model_name: str, num_layers: int) -> dict[int, LogRegProbe]:
    probes: dict[int, LogRegProbe] = {}
    model_dir = normalize_model_name(model_name)
    for layer_idx in range(num_layers):
        path = os.path.join(probes_root, model_dir, f"layer_{layer_idx}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Probe not found for layer {layer_idx}: {path}")
        probe = LogRegProbe()
        probe.load(path)
        probes[layer_idx] = probe
    return probes


def to_tensor(array, device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(array, device=device, dtype=dtype)


def extract_scaler_params(probe: LogRegProbe, device: torch.device):
    if not probe.needs_scaling:
        return {"kind": "none"}

    scaler = probe.scaler
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        return {
            "kind": "standard",
            "mean": to_tensor(scaler.mean_, device),
            "scale": to_tensor(scaler.scale_, device),
        }
    if hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
        return {
            "kind": "robust",
            "center": to_tensor(scaler.center_, device),
            "scale": to_tensor(scaler.scale_, device),
        }
    if hasattr(scaler, "max_abs_"):
        return {
            "kind": "maxabs",
            "max_abs": to_tensor(scaler.max_abs_, device),
        }
    if hasattr(scaler, "min_") and hasattr(scaler, "scale_"):
        return {
            "kind": "minmax",
            "min": to_tensor(scaler.min_, device),
            "scale": to_tensor(scaler.scale_, device),
        }

    raise ValueError("Unsupported scaler type for torch scoring.")


def build_probe_params(probes: dict[int, LogRegProbe], device: torch.device | str = "cpu") -> dict[int, dict]:
    if not isinstance(device, torch.device):
        device = torch.device(device)

    params: dict[int, dict] = {}
    for layer_idx, probe in probes.items():
        weight = to_tensor(probe.model.coef_, device)
        if weight.ndim == 2 and weight.shape[0] == 1:
            weight = weight.squeeze(0)
        bias = to_tensor(probe.model.intercept_, device)
        if bias.ndim >= 1:
            bias = bias.squeeze()

        params[layer_idx] = {
            "weight": weight,
            "bias": bias,
            "scaler": extract_scaler_params(probe, device),
        }
    return params


def apply_scaler(x: torch.Tensor, scaler_params: dict) -> torch.Tensor:
    kind = scaler_params["kind"]
    if kind == "none":
        return x
    if kind == "standard":
        scale = torch.clamp(scaler_params["scale"], min=1e-12)
        return (x - scaler_params["mean"]) / scale
    if kind == "robust":
        scale = torch.clamp(scaler_params["scale"], min=1e-12)
        return (x - scaler_params["center"]) / scale
    if kind == "maxabs":
        scale = torch.clamp(scaler_params["max_abs"], min=1e-12)
        return x / scale
    if kind == "minmax":
        return x * scaler_params["scale"] + scaler_params["min"]

    raise ValueError(f"Unknown scaler kind: {kind}")


def score_with_params(x: torch.Tensor, params: dict) -> torch.Tensor:
    x = apply_scaler(x, params["scaler"])
    weight = params["weight"]
    bias = params["bias"]

    if weight.ndim == 2:
        logits = x @ weight.T + bias
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1]
    logits = x @ weight + bias
    return torch.sigmoid(logits)


def score_activations(
    activations_by_layer: dict[int, torch.Tensor],
    probe_params: dict[int, dict],
    probe_type: str,
    gen_start: int,
    completion_len: int,
) -> torch.Tensor:
    layer_scores = []
    for layer_idx, params in probe_params.items():
        layer_activations = activations_by_layer[layer_idx]
        if probe_type == "input":
            x = layer_activations[gen_start - 1].unsqueeze(0)
            score = score_with_params(x, params)[0]
        else:
            x = layer_activations[gen_start:gen_start + completion_len]
            score = score_with_params(x, params).mean()
        layer_scores.append(score)

    return torch.stack(layer_scores).mean()
