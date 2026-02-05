import torch
import gc
from nnsight import LanguageModel
from utils.model import get_num_layers, get_layer_output

def cache_forward_activations(lm: LanguageModel, input: str) -> dict[int, torch.Tensor]:
    """
    Caches the forward activations of the given LanguageModel for the provided input.

    Args:
        lm (LanguageModel): The language model to cache activations from.
        input (str): An input string to process.

    Returns:
        dict: A dictionary containing the cached activations for every layer. { layer_index: [num_tokens, model_dim] }
    """
    num_layers = get_num_layers(lm.config)
    all_activations = {}

    with torch.no_grad(), lm.trace(input):
        for layer_idx in range(num_layers):
            layer = get_layer_output(lm, layer_idx)
            hidden = layer.output
            if type(hidden) is tuple:
                hidden = hidden[0]
            all_activations[layer_idx] = hidden.squeeze(0).cpu().float().save()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return all_activations


def cache_generation_activations(lm: LanguageModel, input: str) -> tuple[dict[int, torch.Tensor], str]:
    """
    Caches the generation activations of the given LanguageModel for the provided input and returns the generated text.

    Args:
        lm (LanguageModel): The language model to cache activations from.
        input (str): An input string to process.

    Returns:
        dict: A dictionary containing the cached activations for every layer. { layer_index: [num_tokens, model_dim] }
        str: The generated text.
    """
    return {}, ""