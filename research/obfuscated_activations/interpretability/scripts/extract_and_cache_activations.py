#!/usr/bin/env python3
"""
Extract and cache activations from a model with and without embedding suffix attacks.

This script loads a dataset, applies one or more embedding suffix attacks, and extracts
activations from all layers of the model for both the base and attacked prompts.

Multiple suffixes can be provided via --suffix-paths and --save-folder-names (must be the
same length). Baseline computation can be skipped with --no-baseline.
"""

import argparse
from pathlib import Path
from typing import cast

import einops
import torch
from datasets import load_dataset
from mech_interp_toolkit.activation_utils import (
    get_activations,
    get_embeddings_dict,
)
from mech_interp_toolkit.utils import load_model_tokenizer_config, set_global_seed
from utils.data import extract_user_instruction


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
        "--suffix-paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to suffix embedding .pt files",
    )
    parser.add_argument(
        "--save-folder-names",
        type=str,
        nargs="+",
        required=True,
        help="Folder name(s) under --output-dir for each suffix (must match --suffix-paths in length)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory; activations are saved under <output-dir>/baseline/ and <output-dir>/<folder-name>/",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip computing and saving baseline (no-suffix) activations",
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

    if len(args.suffix_paths) != len(args.save_folder_names):
        parser.error("--suffix-paths and --save-folder-names must have the same number of values")

    compute_baseline = not args.no_baseline

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

    # Load model, tokenizer and config (once, shared across all suffixes)
    print(f"Loading model: {args.model_name}")
    model, ch_tokenizer, config = load_model_tokenizer_config(
        args.model_name,
        suffix="",
        system_prompt="",
        attn_type="sdpa",  # Scaled Dot Product Attention for efficiency
    )

    # Setup layer components
    num_layers = config.num_hidden_layers
    components = [(i, "layer_out") for i in range(num_layers)]
    print(f"Extracting activations from {num_layers} layers")

    output_dir = Path(args.output_dir)
    num_batches = (len(prompts_str) + args.batch_size - 1) // args.batch_size

    if compute_baseline:
        base_save_dir = output_dir / "baseline"
        base_save_dir.mkdir(parents=True, exist_ok=True)

    for suffix_path, folder_name in zip(args.suffix_paths, args.save_folder_names):
        print(f"\n=== Processing suffix: {suffix_path} -> {folder_name} ===")

        suffix_embed = load_suffix(suffix_path=suffix_path, device=device)
        len_suffix = suffix_embed.shape[1]
        print(f"Suffix length: {len_suffix}")

        obf_save_dir = output_dir / folder_name
        obf_save_dir.mkdir(parents=True, exist_ok=True)

        base_collate = [] if compute_baseline else None
        new_collate = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(prompts_str))
            batch_prompts = prompts_str[start_idx:end_idx]
            current_batch_size = len(batch_prompts)

            print(
                f"  Batch {batch_idx + 1}/{num_batches} "
                f"(samples {start_idx} to {end_idx - 1})"
            )

            # Get base embeddings
            batch_dict = ch_tokenizer(prompts=batch_prompts)
            batch_embeds_dict = get_embeddings_dict(model, batch_dict)  # type: ignore[arg-type]
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

            if compute_baseline:
                assert base_collate is not None
                base_acts = get_activations(  # type: ignore[arg-type]
                    model,
                    inputs=batch_embeds_dict,
                    layer_components=components,
                    retain_grads=False,
                    positions=None,
                ).cpu()
                base_collate.append(base_acts)

            new_acts = get_activations(  # type: ignore[arg-type]
                model,
                inputs=new_embeds_dict,
                layer_components=components,
                retain_grads=False,
                positions=None,
            ).cpu()
            new_collate.append(new_acts)

        # Save activations for this suffix
        print(f"Saving activations for: {folder_name}")
        split_tag = args.split.replace("/", "_")
        for i, new_acts in enumerate(new_collate):
            torch.save(new_acts, obf_save_dir / f"{i}_{split_tag}_all_pos.pt")
            if compute_baseline:
                assert base_collate is not None and base_save_dir is not None
                torch.save(base_collate[i], base_save_dir / f"{i}_{split_tag}_all_pos.pt")

    print("\nDone!")


if __name__ == "__main__":
    main()
