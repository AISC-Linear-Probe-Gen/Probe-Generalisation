#!/usr/bin/env python3
"""
Direct Attribution Analysis for Probe Generalization

This script computes Direct Logit Attribution (DLA) for prompts with and without adversarial suffixes,
using linear probes trained on different layers.
"""

import argparse
import re
from pathlib import Path
from typing import cast

import einops
import joblib
import torch
from datasets import load_dataset
from tqdm import tqdm

from mech_interp_toolkit.activation_utils import get_activations, get_embeddings_dict
from mech_interp_toolkit.utils import get_layer_components, load_model_tokenizer_config
from utils.data import extract_user_instruction


def extract_layer_number(filepath: str) -> int:
    """Extract layer number from probe filename."""
    filename = Path(filepath).stem
    match = re.search(r'layer_(\d+)', filename)

    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract layer number from filename: {filepath}")


def load_suffix(suffix_path: str, device: torch.device) -> torch.Tensor:
    """Load adversarial suffix embeddings from a .pt file."""
    if not suffix_path.endswith(".pt"):
        raise ValueError("suffix_path must be a .pt embedding file")

    suffix_emb = torch.load(suffix_path, map_location=device)
    if suffix_emb.dim() == 2:
        suffix_emb = suffix_emb.unsqueeze(0)
    return suffix_emb


def main():
    parser = argparse.ArgumentParser(
        description="Compute Direct Logit Attribution for probes with adversarial suffixes"
    )

    # Required arguments (paths)
    parser.add_argument(
        "--probe-dir",
        type=str,
        required=True,
        help="Directory containing probe .joblib files"
    )
    parser.add_argument(
        "--suffix-path",
        type=str,
        required=True,
        help="Path to adversarial suffix embeddings (.pt file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save DLA outputs"
    )

    # Optional arguments (parameters)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Mechanistic-Anomaly-Detection/llama3-jailbreaks",
        help="HuggingFace dataset name (default: Mechanistic-Anomaly-Detection/llama3-jailbreaks)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="circuit_breakers_train",
        help="Dataset split to use (default: circuit_breakers_train)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model name (default: meta-llama/Llama-3.2-3B-Instruct)"
    )
    parser.add_argument(
        "--probe-layer",
        type=int,
        default=12,
        help="Layer number of the probe to use (default: 12)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing prompts (default: 32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--attn-type",
        type=str,
        default="sdpa",
        help="Attention type for model (default: sdpa)"
    )

    args = parser.parse_args()

    # Convert to Path objects
    probe_base_dir = Path(args.probe_dir)
    suffix_path = Path(args.suffix_path)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    # Validate paths
    if not probe_base_dir.exists():
        raise FileNotFoundError(f"Probe directory not found: {probe_base_dir}")
    if not suffix_path.exists():
        raise FileNotFoundError(f"Suffix file not found: {suffix_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading probe from layer {args.probe_layer}...")

    # Load the probe for the specified layer
    probe = None
    probe_file_found = False
    for probe_file in probe_base_dir.glob("*.joblib"):
        layer = extract_layer_number(str(probe_file))
        if layer == args.probe_layer:
            print(f"Loading probe from: {probe_file}")
            probe = joblib.load(probe_file)
            probe_file_found = True
            break

    if not probe_file_found:
        raise FileNotFoundError(
            f"No probe file found for layer {args.probe_layer} in {probe_base_dir}"
        )

    # Extract probe direction
    probe_dir = torch.from_numpy(probe["model"].coef_).to(device).squeeze().to(torch.bfloat16)
    probe_dir = probe_dir / probe_dir.norm()

    print(f"Loading dataset: {args.dataset_name} (split: {args.split})...")

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = cast(dict, dataset)
    prompts_str = [extract_user_instruction(x) for x in dataset["prompt"]]

    print(f"Loaded {len(prompts_str)} prompts")
    print(f"Loading model: {args.model_name}...")

    # Load model, tokenizer and config
    model, ch_tokenizer, config = load_model_tokenizer_config(
        args.model_name,
        suffix="",
        system_prompt="",
        attn_type=args.attn_type,
    )

    print("Loading adversarial suffix...")

    # Load suffix embeddings
    suffix_embed = load_suffix(str(suffix_path), device)
    len_suffix = suffix_embed.shape[1]

    print(f"Suffix length: {len_suffix} tokens")
    print(f"Getting layer components up to layer {args.probe_layer}...")

    # Get layer components
    components = get_layer_components(model, stop_at=args.probe_layer, include_block_outputs=False)

    # Process prompts in batches
    num_batches = (len(prompts_str) + args.batch_size - 1) // args.batch_size

    base_dla_collate = []
    new_dla_collate = []

    print(f"\nProcessing {num_batches} batches...")

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(prompts_str))

        batch_prompts = prompts_str[start_idx:end_idx]
        current_batch_size = len(batch_prompts)

        # Use torch.no_grad() to prevent gradient accumulation
        with torch.no_grad():
            batch_dict = ch_tokenizer(prompts=batch_prompts)
            # removes "input_ids" and adds "inputs_embeds"
            batch_embeds_dict = get_embeddings_dict(model, batch_dict)
            batch_embeds = batch_embeds_dict["inputs_embeds"]
            batch_attn_mask = batch_embeds_dict["attention_mask"]

            # broadcast suffix
            batch_suffix = einops.repeat(
                suffix_embed,
                "dummy pos d_model -> (curr_batch dummy) pos d_model",
                curr_batch=current_batch_size,
            )

            new_embeds = torch.cat(
                [batch_embeds[:, :-5, :], batch_suffix, batch_embeds[:, -5:, :]], dim=1
            )
            attn_extension = torch.ones((current_batch_size, len_suffix), device=device)
            new_attn = torch.cat([batch_attn_mask, attn_extension], dim=1)

            new_embeds_dict = {"inputs_embeds": new_embeds, "attention_mask": new_attn}

            base_acts = get_activations(
                model,
                inputs=batch_embeds_dict,
                layer_components=components,
                retain_grads=False,
                positions=-1,
            ).apply(torch.squeeze)

            new_acts = get_activations(
                model,
                inputs=new_embeds_dict,
                layer_components=components,
                retain_grads=False,
                positions=-1,
            ).apply(torch.squeeze)

            # Compute DLA and immediately move to CPU
            base_dla = (base_acts @ probe_dir).cpu()
            new_dla = (new_acts @ probe_dir).cpu()

            base_dla_collate.append(base_dla)
            new_dla_collate.append(new_dla)

        # Clear GPU cache after each batch
        del batch_dict, batch_embeds_dict, batch_embeds, batch_attn_mask
        del batch_suffix, new_embeds, attn_extension, new_attn, new_embeds_dict
        del base_acts, new_acts, base_dla, new_dla
        torch.cuda.empty_cache()

        # Print memory usage every 5 batches
        if batch_idx % 5 == 0 and device.type == "cuda":
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Save results
    print(f"\nSaving results to {output_dir}...")
    save_filename = f"{args.split}_last_pos.pt"

    for i, (base_dla, new_dla) in enumerate(tqdm(zip(base_dla_collate, new_dla_collate), desc="Saving batches")):
        save_path = output_dir / f"{i}_{save_filename}"
        torch.save((base_dla, new_dla), save_path)

    print(f"\nDone! Saved {len(base_dla_collate)} batch files to {output_dir}")
    print(f"Files follow pattern: <batch_index>_{save_filename}")


if __name__ == "__main__":
    main()
