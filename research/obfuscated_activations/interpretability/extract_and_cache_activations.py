#!/usr/bin/env python3
"""
Extract and cache activations from a model with and without embedding suffix attacks.

This script loads a dataset, applies an embedding suffix attack, and extracts activations
from all layers of the model for both the base and attacked prompts.
"""

import argparse
from pathlib import Path
from typing import cast

import torch
import einops
from datasets import load_dataset

from utils.data import extract_user_instruction
from mech_interp_toolkit.utils import load_model_tokenizer_config, set_global_seed
from mech_interp_toolkit.activation_utils import (
    get_embeddings_dict,
    get_activations,
    concat_activations,
)


def load_suffix(suffix_path: str, device: torch.device) -> torch.Tensor:
    """Load suffix embeddings from a .pt file."""
    if not suffix_path.endswith(".pt"):
        raise ValueError("suffix_path must be a .pt embedding file")
    suffix_emb = torch.load(suffix_path, map_location=device)
    if suffix_emb.dim() == 2:
        suffix_emb = suffix_emb.unsqueeze(0)
    return suffix_emb


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache activations with embedding suffix attacks"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., 'Mechanistic-Anomaly-Detection/llama3-jailbreaks')",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split to use (e.g., 'circuit_breakers_train[:200]')",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (e.g., 'meta-llama/Llama-3.2-3B-Instruct')",
    )
    parser.add_argument(
        "--suffix-path",
        type=str,
        required=True,
        help="Path to the suffix embedding .pt file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path to save cached activations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    args = parser.parse_args()

    # Set seed and disable gradients
    set_global_seed(args.seed)
    torch.enable_grad(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset: {args.dataset_name} (split: {args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = cast(dict, dataset)
    prompts_str = [extract_user_instruction(x) for x in dataset["prompt"]]
    print(f"Loaded {len(prompts_str)} prompts")

    # Load model, tokenizer and config
    print(f"Loading model: {args.model_name}")
    model, ch_tokenizer, config = load_model_tokenizer_config(
        args.model_name,
        suffix="",
        system_prompt="",
        attn_type="sdpa",  # Scaled Dot Product Attention for efficiency
    )

    # Load suffix
    print(f"Loading suffix from: {args.suffix_path}")
    suffix_embed = load_suffix(suffix_path=args.suffix_path, device=device)
    len_suffix = suffix_embed.shape[1]
    print(f"Suffix length: {len_suffix}")

    # Setup layer components
    num_layers = config.num_hidden_layers
    components = [(i, "layer_out") for i in range(num_layers)]
    print(f"Extracting activations from {num_layers} layers")

    # Process prompts in batches
    num_batches = (len(prompts_str) + args.batch_size - 1) // args.batch_size
    base_collate = []
    new_collate = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(prompts_str))
        batch_prompts = prompts_str[start_idx:end_idx]
        current_batch_size = len(batch_prompts)

        print(
            f"Processing batch {batch_idx + 1}/{num_batches} "
            f"(samples {start_idx} to {end_idx - 1})"
        )

        # Get base embeddings
        batch_dict = ch_tokenizer(prompts=batch_prompts)
        batch_embeds_dict = get_embeddings_dict(model, batch_dict)
        batch_embeds = batch_embeds_dict["inputs_embeds"]
        batch_attn_mask = batch_embeds_dict["attention_mask"]

        # Create new embeddings with suffix inserted
        batch_suffix = einops.repeat(
            suffix_embed,
            "dummy pos d_model -> (curr_batch dummy) pos d_model",
            curr_batch=current_batch_size,
        )

        new_embeds = torch.cat(
            [batch_embeds[:, :-5, :], batch_suffix, batch_embeds[:, -5:, :]], dim=1
        )
        attn_extension = torch.ones((current_batch_size, len_suffix)).to(device)
        new_attn = torch.cat([batch_attn_mask, attn_extension], dim=1)

        new_embeds_dict = {"inputs_embeds": new_embeds, "attention_mask": new_attn}

        # Extract activations
        base_acts = get_activations(
            model,
            inputs=batch_embeds_dict,
            layer_components=components,
            retain_grads=False,
            positions=-1,
        ).cpu()

        new_acts = get_activations(
            model,
            inputs=new_embeds_dict,
            layer_components=components,
            retain_grads=False,
            positions=-1,
        ).cpu()

        base_collate.append(base_acts)
        new_collate.append(new_acts)

    # Concatenate all batches
    print("Concatenating activations from all batches...")
    full_base_acts = dict(concat_activations(base_collate))
    full_new_acts = dict(concat_activations(new_collate))

    # Save activations
    save_path = Path(args.output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving activations to {save_path}")
    torch.save((full_base_acts, full_new_acts), save_path)
    print("Done!")


if __name__ == "__main__":
    main()
