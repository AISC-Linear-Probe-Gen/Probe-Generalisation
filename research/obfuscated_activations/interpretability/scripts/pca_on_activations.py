"""
PCA Analysis on Original vs Obfuscated Activations

This script performs Principal Component Analysis (PCA) on activations to compare
the dimensionality of original and obfuscated activations across different layers.
"""

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import gc
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from mech_interp_toolkit.activation_utils import concat_activations


def load_activations(activation_path, max_files=None):
    """
    Load original and obfuscated activations from cached files.

    Args:
        activation_path: Path to directory containing cached activation files
        max_files: Maximum number of files to load (None for all files)

    Returns:
        tuple: (original_activations, obfuscated_activations)
    """
    path = Path(activation_path)
    original_activations = []
    obfuscated_activations = []

    files = list(path.glob("*.pt"))
    if max_files is not None:
        files = files[:max_files]

    print(f"Loading activations from {len(files)} files...")
    for f in tqdm(files):
        original, obfuscated = torch.load(f, weights_only=False)
        original.attention_mask = torch.empty(0)
        obfuscated.attention_mask = torch.empty(0)
        original_activations.append(original)
        obfuscated_activations.append(obfuscated)

    return original_activations, obfuscated_activations


def precompute_layer_activations(original_activations, obfuscated_activations, num_layers):
    """
    Pre-compute activations for all layers from concatenated activations.

    Args:
        original_activations: List of original activation batches
        obfuscated_activations: List of obfuscated activation batches
        num_layers: Number of layers in the model

    Returns:
        tuple: (all_layers_original, all_layers_obfuscated) dictionaries
    """
    all_layers_original = {}
    all_layers_obfuscated = {}

    print("Concatenating activations...")
    original_activations = concat_activations(original_activations, pad_value=0)
    obfuscated_activations = concat_activations(obfuscated_activations, pad_value=0)

    print(f"Pre-computing activations for {num_layers} layers...")
    for layer in tqdm(range(num_layers)):
        all_layers_original[layer] = (
            original_activations[(layer, "layer_out")][:, -1, :].float().numpy()
        )
        all_layers_obfuscated[layer] = (
            obfuscated_activations[(layer, "layer_out")][:, -1, :].float().numpy()
        )

    # Clean up memory
    del original_activations, obfuscated_activations
    gc.collect()
    torch.cuda.empty_cache()

    return all_layers_original, all_layers_obfuscated


def compute_pca_stats(activations):
    """
    Compute PCA statistics for given activations.

    Args:
        activations: numpy array of activations

    Returns:
        dict: Dictionary containing PCA statistics
    """
    pca = PCA()
    pca.fit(activations)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

    return {
        'pca': pca,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'n_components_90': n_components_90,
        'n_components_95': n_components_95,
    }


def plot_pca_comparison(layer, original_stats, obfuscated_stats, output_dir=None):
    """
    Plot PCA comparison for a specific layer.

    Args:
        layer: Layer number
        original_stats: PCA statistics for original activations
        obfuscated_stats: PCA statistics for obfuscated activations
        output_dir: Directory to save the plot (None to display only)
    """
    fig = plt.figure(figsize=(12, 6))

    # Plot both cumulative variance curves
    plt.plot(
        range(1, len(original_stats['cumulative_variance']) + 1),
        original_stats['cumulative_variance'],
        linewidth=2, label='Original Activations', color='blue'
    )
    plt.plot(
        range(1, len(obfuscated_stats['cumulative_variance']) + 1),
        obfuscated_stats['cumulative_variance'],
        linewidth=2, label='Obfuscated Activations', color='orange'
    )

    # Add horizontal reference lines
    plt.axhline(y=0.90, color='red', linestyle='--', linewidth=1.5, label='90% variance', alpha=0.5)
    plt.axhline(y=0.95, color='green', linestyle='--', linewidth=1.5, label='95% variance', alpha=0.5)

    # Add vertical lines for original activations
    plt.axvline(x=original_stats['n_components_90'], color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
    plt.axvline(x=original_stats['n_components_95'], color='blue', linestyle=':', linewidth=1.5, alpha=0.5)

    # Add vertical lines for obfuscated activations
    plt.axvline(x=obfuscated_stats['n_components_90'], color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    plt.axvline(x=obfuscated_stats['n_components_95'], color='orange', linestyle=':', linewidth=1.5, alpha=0.5)

    # Add annotations for original activations
    plt.text(
        original_stats['n_components_90'], 0.88,
        f"Orig: {original_stats['n_components_90']}",
        ha='center', va='top', fontsize=9, color='blue',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.7)
    )
    plt.text(
        original_stats['n_components_95'], 0.85,
        f"Orig: {original_stats['n_components_95']}",
        ha='center', va='top', fontsize=9, color='blue',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.7)
    )

    # Add annotations for obfuscated activations
    plt.text(
        obfuscated_stats['n_components_90'], 0.92,
        f"Obf: {obfuscated_stats['n_components_90']}",
        ha='center', va='bottom', fontsize=9, color='orange',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='orange', alpha=0.7)
    )
    plt.text(
        obfuscated_stats['n_components_95'], 0.97,
        f"Obf: {obfuscated_stats['n_components_95']}",
        ha='center', va='bottom', fontsize=9, color='orange',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='orange', alpha=0.7)
    )

    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Cumulative Variance Explained', fontsize=12)
    plt.title(f'PCA Cumulative Variance: Original vs Obfuscated Activations (Layer {layer})', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(len(original_stats['cumulative_variance']), len(obfuscated_stats['cumulative_variance'])))
    plt.ylim(0, 1.0)
    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"pca_layer_{layer}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def print_pca_stats(layer, original_stats, obfuscated_stats):
    """Print PCA statistics for a layer."""
    print(f"\nLayer {layer}:")
    print(f"Original Activations:")
    print(f"  Number of components for 90% variance: {original_stats['n_components_90']}")
    print(f"  Number of components for 95% variance: {original_stats['n_components_95']}")
    print(f"  Total number of components: {len(original_stats['cumulative_variance'])}")

    print(f"\nObfuscated Activations:")
    print(f"  Number of components for 90% variance: {obfuscated_stats['n_components_90']}")
    print(f"  Number of components for 95% variance: {obfuscated_stats['n_components_95']}")
    print(f"  Total number of components: {len(obfuscated_stats['cumulative_variance'])}")


def main():
    parser = argparse.ArgumentParser(
        description='Perform PCA analysis on original vs obfuscated activations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        'activation_path',
        type=str,
        help='Path to directory containing cached activation files (*.pt)'
    )

    # Optional arguments
    parser.add_argument(
        '--num-layers',
        type=int,
        default=28,
        help='Number of layers in the model'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of activation files to load (None for all)'
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=None,
        help='Specific layers to analyze (e.g., --layers 0 5 10 15). If not specified, analyzes all layers.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save plots (if not specified, plots are only displayed)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display plots (only save them if --output-dir is specified)'
    )

    args = parser.parse_args()

    # Validate activation path
    if not Path(args.activation_path).exists():
        print(f"Error: Activation path '{args.activation_path}' does not exist")
        return 1

    # Load activations
    original_activations, obfuscated_activations = load_activations(
        args.activation_path,
        max_files=args.max_files
    )

    # Pre-compute layer activations
    all_layers_original, all_layers_obfuscated = precompute_layer_activations(
        original_activations,
        obfuscated_activations,
        args.num_layers
    )

    # Determine which layers to analyze
    if args.layers is not None:
        layers_to_analyze = args.layers
    else:
        layers_to_analyze = range(args.num_layers)

    # Set matplotlib backend if no display
    if args.no_display:
        plt.switch_backend('Agg')

    # Analyze each layer
    for layer in layers_to_analyze:
        if layer not in all_layers_original:
            print(f"Warning: Layer {layer} not found in activations, skipping...")
            continue

        print(f"\nAnalyzing layer {layer}...")

        # Compute PCA statistics
        original_stats = compute_pca_stats(all_layers_original[layer])
        obfuscated_stats = compute_pca_stats(all_layers_obfuscated[layer])

        # Print statistics
        print_pca_stats(layer, original_stats, obfuscated_stats)

        # Plot comparison
        plot_pca_comparison(layer, original_stats, obfuscated_stats, args.output_dir)

        if not args.no_display and args.output_dir is None:
            plt.close()  # Close the plot to avoid memory issues when analyzing many layers

    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
