#!/usr/bin/env python3
"""
Generate text completions with adversarial suffix injection.

This script loads a model, dataset, and adversarial suffix, then generates
text completions by injecting the suffix into the prompt embeddings.
"""

import argparse
import json
import warnings
from pathlib import Path

import einops
import torch
from datasets import load_dataset
from tqdm import tqdm

from utils.data import extract_user_instruction
from mech_interp_toolkit.utils import load_model_tokenizer_config, set_global_seed
from mech_interp_toolkit.activation_utils import get_embeddings_dict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with adversarial suffix injection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments (paths)
    parser.add_argument(
        "--suffix-path",
        type=str,
        required=True,
        help="Path to the adversarial suffix embeddings (.pt file)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the generated completions (JSON file)"
    )

    # Optional arguments (parameters)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Mechanistic-Anomaly-Detection/llama3-jailbreaks",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="circuit_breakers_train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--suffix-position",
        type=int,
        default=5,
        help="Number of tokens from the end to inject suffix before"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If None, auto-detect"
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Number of sample results to display (0 to disable)"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Setup
    warnings.filterwarnings("ignore")
    set_global_seed(args.seed)
    torch.set_grad_enabled(False)

    # Determine device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model, tokenizer, and config
    print(f"Loading model: {args.model_name}")
    model, ch_tokenizer, config = load_model_tokenizer_config(args.model_name, device=device)

    # Load dataset
    print(f"Loading dataset: {args.dataset_name} (split: {args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)
    prompts_str = [extract_user_instruction(item) for item in dataset]
    print(f"Loaded {len(prompts_str)} prompts from dataset")

    # Load adversarial suffix
    print(f"Loading suffix from: {args.suffix_path}")
    suffix_embed = torch.load(args.suffix_path, map_location=device)
    len_suffix = suffix_embed.shape[1]
    print(f"Suffix shape: {suffix_embed.shape}")

    # Process batches and generate text with suffix injection
    num_batches = (len(prompts_str) + args.batch_size - 1) // args.batch_size
    results = []

    print(f"\nProcessing {num_batches} batches...")
    for batch_idx in tqdm(range(num_batches), desc="Generating completions"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(prompts_str))
        batch_prompts = prompts_str[start_idx:end_idx]
        current_batch_size = len(batch_prompts)

        # Tokenize and get embeddings
        batch_dict = ch_tokenizer(prompts=batch_prompts)
        batch_embeds_dict = get_embeddings_dict(model, batch_dict)
        batch_embeds = batch_embeds_dict["inputs_embeds"]
        batch_attn_mask = batch_embeds_dict["attention_mask"]

        # Broadcast suffix to batch size
        batch_suffix = einops.repeat(
            suffix_embed,
            "dummy pos d_model -> (curr_batch dummy) pos d_model",
            curr_batch=current_batch_size,
        )

        # Inject suffix before the last N tokens
        new_embeds = torch.cat(
            [
                batch_embeds[:, :-args.suffix_position, :],
                batch_suffix,
                batch_embeds[:, -args.suffix_position:, :]
            ],
            dim=1
        )
        attn_extension = torch.ones((current_batch_size, len_suffix), device=device)
        new_attn = torch.cat([batch_attn_mask, attn_extension], dim=1)

        new_embeds_dict = {
            "inputs_embeds": new_embeds,
            "attention_mask": new_attn
        }

        # Generate text with suffix injection
        generated_output = model.generate(
            **new_embeds_dict,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            pad_token_id=ch_tokenizer.tokenizer.pad_token_id,
        )

        # Decode the generated tokens
        generated_text = ch_tokenizer.tokenizer.batch_decode(
            generated_output,
            skip_special_tokens=True
        )

        # Store results with prompts
        for prompt, generation in zip(batch_prompts, generated_text):
            results.append({
                "prompt": prompt,
                "generation": generation
            })

    print(f"\nGenerated {len(results)} completions")

    # Display sample results
    if args.show_samples > 0:
        print(f"\nSample generated text with suffix injection:\n")
        for idx, result in enumerate(results[:args.show_samples]):
            print(f"Sample {idx}:")
            print(f"Prompt: {result['prompt'][:100]}...")  # Show first 100 chars
            print(f"Generation: {result['generation']}")
            print("-" * 80)

    # Save generations to JSON file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(results)} generations to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved generations to {output_path}")


if __name__ == "__main__":
    main()
