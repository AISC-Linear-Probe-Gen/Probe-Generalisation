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


