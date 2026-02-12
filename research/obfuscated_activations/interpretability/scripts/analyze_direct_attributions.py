#!/usr/bin/env python3
"""
Analyze direct attributions by comparing original and new activation distributions.

This script loads activation dictionaries from .pt files, concatenates them,
and creates histogram plots comparing the original and new activations for each key.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mech_interp_toolkit.activation_utils import concat_activations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize direct attribution activations"
    )

    # Required arguments
    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing .pt activation files"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for the plot (default: display plot)"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=7,
        help="Number of rows in the subplot grid (default: 7)"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in the subplot grid (default: 4)"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins for histograms (default: 50)"
    )
    parser.add_argument(
        "--figsize-width",
        type=float,
        default=16.0,
        help="Figure width in inches (default: 16.0)"
    )
    parser.add_argument(
        "--figsize-height",
        type=float,
        default=24.0,
        help="Figure height in inches (default: 24.0)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency level for histograms (default: 0.5)"
    )

    return parser.parse_args()


def load_activations(base_dir: Path):
    """
    Load activation dictionaries from .pt files.

    Args:
        base_dir: Directory containing .pt files

    Returns:
        Tuple of (original_activations, new_activations)
    """
    collate_orig = []
    collate_new = []

    for f in base_dir.glob("*.pt"):
        orig, new = torch.load(f, weights_only=False)
        # Clear attention masks to save memory
        orig.attention_mask = torch.empty(0)
        new.attention_mask = torch.empty(0)
        collate_orig.append(orig)
        collate_new.append(new)

    if not collate_orig:
        raise ValueError(f"No .pt files found in {base_dir}")

    # Concatenate all activations
    orig = concat_activations(collate_orig)
    new = concat_activations(collate_new)

    return orig, new


def plot_activations(orig, new, rows, cols, bins, figsize, alpha, output_path=None):
    """
    Create histogram plots comparing original and new activations.

    Args:
        orig: Original activation dictionary
        new: New activation dictionary
        rows: Number of rows in subplot grid
        cols: Number of columns in subplot grid
        bins: Number of bins for histograms
        figsize: Tuple of (width, height) for figure size
        alpha: Transparency level for histograms
        output_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    keys = list(orig.keys())
    max_plots = rows * cols

    for idx, key in enumerate(keys):
        if idx >= max_plots:
            break

        # Get data for both histograms
        data_orig = orig[key].float().numpy()
        data_new = new[key].float().numpy()

        # Determine shared bins based on combined data range
        all_data = torch.cat([orig[key].float(), new[key].float()])
        min_val, max_val = all_data.min().item(), all_data.max().item()
        bin_edges = torch.linspace(min_val, max_val, bins)

        # Plot both histograms with the same bins
        axes[idx].hist(data_orig, bins=bin_edges.numpy(), alpha=alpha, label='Original')
        axes[idx].hist(data_new, bins=bin_edges.numpy(), alpha=alpha, label='New')
        axes[idx].set_title(key)
        axes[idx].legend()

    # Hide any unused subplots
    for idx in range(len(keys), max_plots):
        axes[idx].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    """Main execution function."""
    args = parse_args()

    # Convert base_dir to Path
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    if not base_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {base_dir}")

    print(f"Loading activations from {base_dir}...")
    orig, new = load_activations(base_dir)

    print(f"Loaded {len(orig.keys())} activation keys")
    print(f"Creating {args.rows}x{args.cols} grid of plots...")

    figsize = (args.figsize_width, args.figsize_height)
    plot_activations(
        orig, new,
        rows=args.rows,
        cols=args.cols,
        bins=args.bins,
        figsize=figsize,
        alpha=args.alpha,
        output_path=args.output
    )

    print("Done!")


if __name__ == "__main__":
    main()
