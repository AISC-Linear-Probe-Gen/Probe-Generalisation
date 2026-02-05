from transformers import PretrainedConfig
from nnsight import LanguageModel


def get_num_layers(config: PretrainedConfig) -> int:
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    else:
        raise ValueError("Could not determine number of layers from config.")
    
def get_layer_output(lm: LanguageModel, layer_idx: int):
    num_layers = get_num_layers(lm.config)
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"Layer index {layer_idx} is out of bounds for model with {num_layers} layers.")
    
    if hasattr(lm, "language_model"):
        return lm.language_model.layers[layer_idx]
    
    return lm.model.layers[layer_idx]
