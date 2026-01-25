"""Utilities for extracting activations from language models."""
import torch
from torch import nn, Tensor
from typing import Callable
from contextlib import contextmanager

class ActivationCache:
    """Hook-based activation caching for transformer models."""
    def __init__(self, model: nn.Module, layer_indices: list[int] | None = None):
        self.model = model
        self.layer_indices = layer_indices
        self.activations: dict[int, Tensor] = {}
        self._hooks: list = []

    def _get_layers(self) -> list[tuple[int, nn.Module]]:
        """Get transformer layers from model. Works with HF models and PEFT/LoRA."""
        # Try common attribute names for transformer layers
        # Order matters - more specific patterns first
        for attr in [
            'base_model.model.model.layers',  # PEFT/LoRA wrapped Gemma3
            'base_model.model.model.language_model.layers',  # PEFT/LoRA wrapped Gemma3 multimodal
            'model.language_model.layers',  # Gemma3
            'language_model.layers',        # Gemma3 alt
            'model.model.layers',           # Some PEFT wrappers
            'model.layers',                 # Llama, Gemma2, Qwen
            'transformer.h',                # GPT-2
            'gpt_neox.layers',              # GPT-NeoX
            'model.decoder.layers',         # BART, T5
        ]:
            parts = attr.split('.')
            obj = self.model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                layers = list(enumerate(obj))
                if self.layer_indices is not None:
                    layers = [(i, obj[i]) for i in self.layer_indices if i < len(obj)]
                return layers
            except AttributeError:
                continue
        raise ValueError("Could not find transformer layers in model")

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.activations[layer_idx] = hidden.detach()
        return hook

    @contextmanager
    def capture(self):
        """Context manager for capturing activations."""
        self.activations.clear()
        layers = self._get_layers()
        for idx, layer in layers:
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

        try:
            yield self
        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()

    def get(self, layer_idx: int) -> Tensor:
        """Get cached activations for a layer."""
        return self.activations[layer_idx]

    def get_last_token(self, layer_idx: int) -> Tensor:
        """Get activations at last token position for a layer."""
        return self.activations[layer_idx][:, -1, :]


def get_activations(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    layer_idx: int,
    batch_size: int = 8,
    device: str = "cuda",
    pooling: str = "mean",  # "mean", "last", or "all"
) -> Tensor:
    """Extract activations from a specific layer for a list of texts.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of input texts
        layer_idx: Which layer to extract from
        batch_size: Batch size for processing
        device: Device to run on
        pooling: How to aggregate token activations:
            - "mean": Mean pool across all non-padding tokens
            - "last": Last non-padding token only
            - "all": Return all token activations (paper method)

    Returns:
        Tensor of shape (n_texts, hidden_dim) if pooling is "mean" or "last"
        else (n_texts, max_seq_len, hidden_dim) if "all"
    """
    model.eval()
    cache = ActivationCache(model, [layer_idx])
    all_activations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad(), cache.capture():
            model(**inputs)

        hidden = cache.get(layer_idx)

        if pooling == "mean":
            # Mean pool across all non-padding tokens (paper method)
            mask = inputs.attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            masked_hidden = hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden)
            lengths = mask.sum(dim=1)  # (batch, 1)
            batch_acts = sum_hidden / lengths  # (batch, hidden)
        elif pooling == "last":
            # Get last non-padding token for each sequence
            seq_lens = inputs.attention_mask.sum(dim=1) - 1
            batch_acts = torch.stack([
                hidden[j, seq_lens[j], :] for j in range(len(batch_texts))
            ])
        else:  # "all"
            # For "all" pooling, we need to handle variable sequence lengths
            # Store each sequence separately (can't concat different seq lengths)
            for j in range(len(batch_texts)):
                seq_len = inputs.attention_mask[j].sum().item()
                # Only keep non-padding tokens
                all_activations.append(hidden[j, :seq_len, :].cpu())
            continue  # Skip the append below

        all_activations.append(batch_acts.cpu())
    if pooling == "all":
        # Return list of tensors (variable seq lengths) for "all" pooling
        return all_activations
    return torch.cat(all_activations, dim=0)