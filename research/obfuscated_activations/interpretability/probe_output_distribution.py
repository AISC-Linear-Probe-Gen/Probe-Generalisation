#!/usr/bin/env python3
"""
Analyze probe output distributions for baseline and obfuscated activations.

This script loads probes and cached activations, computes predictions and logits,
and generates visualizations comparing baseline and obfuscated activations.
"""

import argparse
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re
import joblib
from collections import defaultdict


def extract_layer_number(filepath: str) -> int:
    """Extract layer number from filepath using regex."""
    filename = Path(filepath).stem

    match = re.search(r'layer_(\d+)', filename)

    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract layer number from filename: {filepath}")


def quick_pred(probe, activations):
    """Generate predictions and logits from probe and activations."""
    activations = activations.float().numpy()
    scaled_activations = probe["scaler"].transform(activations)
    logits = probe["model"].decision_function(scaled_activations)
    predictions = probe["model"].predict(scaled_activations)
    return predictions, logits


def load_probes(probe_dir: Path):
    """Load all probes from directory and organize by layer."""
    probes_by_layer = {}
    for probe_file in probe_dir.glob("*.joblib"):
        layer = extract_layer_number(str(probe_file))
        probe = joblib.load(probe_file)
        probes_by_layer[layer] = probe
    print(f"Loaded {len(probes_by_layer)} probes from {probe_dir}")
    return probes_by_layer


def process_activations(activations_dir: Path, probes_by_layer: dict):
    """Process all activation files and compute predictions/logits."""
    baseline_predictions_by_layer = defaultdict(list)
    baseline_logits_by_layer = defaultdict(list)
    obfuscated_predictions_by_layer = defaultdict(list)
    obfuscated_logits_by_layer = defaultdict(list)

    activation_files = list(activations_dir.glob("*.pt"))
    print(f"Processing {len(activation_files)} activation files from {activations_dir}")

    for activation_file in activation_files:
        baseline_activations, obfuscated_activations = torch.load(
            activation_file, weights_only=False
        )

        baseline_activations.positions = -1
        obfuscated_activations.positions = -1

        baseline_activations = baseline_activations.extract_positions()
        obfuscated_activations = obfuscated_activations.extract_positions()

        for layer, _ in baseline_activations:
            probe = probes_by_layer[layer]

            baseline_layer_activations = baseline_activations[(layer, "layer_out")]
            baseline_predictions, baseline_logits = quick_pred(probe, baseline_layer_activations)

            obfuscated_layer_activations = obfuscated_activations[(layer, "layer_out")]
            obfuscated_predictions, obfuscated_logits = quick_pred(probe, obfuscated_layer_activations)

            baseline_predictions_by_layer[layer].append(baseline_predictions)
            baseline_logits_by_layer[layer].append(baseline_logits)
            obfuscated_predictions_by_layer[layer].append(obfuscated_predictions)
            obfuscated_logits_by_layer[layer].append(obfuscated_logits)

    return (baseline_predictions_by_layer, baseline_logits_by_layer,
            obfuscated_predictions_by_layer, obfuscated_logits_by_layer)


def compute_average_predictions(baseline_predictions_by_layer, obfuscated_predictions_by_layer):
    """Compute average predictions across all layers."""
    # Concatenate predictions across activation files for each layer
    baseline_preds_concatenated = {}
    obfuscated_preds_concatenated = {}

    for layer in sorted(baseline_predictions_by_layer.keys()):
        baseline_preds_concatenated[layer] = np.concatenate(baseline_predictions_by_layer[layer])
        obfuscated_preds_concatenated[layer] = np.concatenate(obfuscated_predictions_by_layer[layer])

    print(f"\nNumber of layers: {len(baseline_preds_concatenated)}")
    print(f"Number of datapoints per layer: {baseline_preds_concatenated[list(baseline_preds_concatenated.keys())[0]].shape}")

    # Stack predictions from all layers (shape: num_layers x num_datapoints)
    baseline_preds_stacked = np.stack([baseline_preds_concatenated[layer] for layer in sorted(baseline_preds_concatenated.keys())])
    obfuscated_preds_stacked = np.stack([obfuscated_preds_concatenated[layer] for layer in sorted(obfuscated_preds_concatenated.keys())])

    # Average across layers (axis=0)
    baseline_avg_preds = baseline_preds_stacked.mean(axis=0)
    obfuscated_avg_preds = obfuscated_preds_stacked.mean(axis=0)

    print(f"Shape of stacked predictions: {baseline_preds_stacked.shape}")
    print(f"Shape of averaged predictions: {baseline_avg_preds.shape}")
    print(f"\nBaseline average predictions (first 10): {baseline_avg_preds[:10]}")
    print(f"Obfuscated average predictions (first 10): {obfuscated_avg_preds[:10]}")

    return baseline_avg_preds, obfuscated_avg_preds


def plot_average_predictions(baseline_avg_preds, obfuscated_avg_preds, output_dir: Path):
    """Plot distribution of average predictions."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    bins = np.linspace(0, 1, 21)

    ax.hist(baseline_avg_preds, bins=bins, alpha=0.6, label='Baseline', edgecolor='black')
    ax.hist(obfuscated_avg_preds, bins=bins, alpha=0.6, label='Obfuscated', color='orange', edgecolor='black')
    ax.set_xlabel('Average Prediction')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Average Predictions Across Layers')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "average_predictions_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved average predictions plot to {output_file}")
    plt.show()

    # Print statistics
    print(f"\nBaseline Statistics:")
    print(f"  Mean: {baseline_avg_preds.mean():.4f}")
    print(f"  Std: {baseline_avg_preds.std():.4f}")
    print(f"  Min: {baseline_avg_preds.min():.4f}")
    print(f"  Max: {baseline_avg_preds.max():.4f}")

    print(f"\nObfuscated Statistics:")
    print(f"  Mean: {obfuscated_avg_preds.mean():.4f}")
    print(f"  Std: {obfuscated_avg_preds.std():.4f}")
    print(f"  Min: {obfuscated_avg_preds.min():.4f}")
    print(f"  Max: {obfuscated_avg_preds.max():.4f}")


def plot_logit_distributions(baseline_logits_by_layer, obfuscated_logits_by_layer, output_dir: Path):
    """Plot logit distributions for each layer."""
    # Concatenate logits across activation files for each layer
    baseline_logits_concatenated = {}
    obfuscated_logits_concatenated = {}

    for layer in sorted(baseline_logits_by_layer.keys()):
        baseline_logits_concatenated[layer] = np.concatenate(baseline_logits_by_layer[layer])
        obfuscated_logits_concatenated[layer] = np.concatenate(obfuscated_logits_by_layer[layer])

    # Create 4x7 grid of histograms showing baseline and obfuscated logits at each layer
    num_layers = len(baseline_logits_concatenated)
    fig, axes = plt.subplots(4, 7, figsize=(20, 12))
    axes = axes.flatten()

    # Determine global min/max for consistent binning across all layers
    all_baseline_logits = np.concatenate([baseline_logits_concatenated[layer] for layer in sorted(baseline_logits_concatenated.keys())])
    all_obfuscated_logits = np.concatenate([obfuscated_logits_concatenated[layer] for layer in sorted(obfuscated_logits_concatenated.keys())])
    global_min = min(all_baseline_logits.min(), all_obfuscated_logits.min())
    global_max = max(all_baseline_logits.max(), all_obfuscated_logits.max())
    bins = np.linspace(global_min, global_max, 31)

    for idx, layer in enumerate(sorted(baseline_logits_concatenated.keys())):
        if idx >= 28:  # Only plot first 28 layers (4x7 grid)
            break

        ax = axes[idx]

        # Plot both baseline and obfuscated logits with same bins
        ax.hist(baseline_logits_concatenated[layer], bins=bins, alpha=0.6, label='Baseline', edgecolor='black', color='blue')
        ax.hist(obfuscated_logits_concatenated[layer], bins=bins, alpha=0.6, label='Obfuscated', edgecolor='black', color='orange')

        ax.set_title(f'Layer {layer}', fontsize=10)
        ax.set_xlabel('Logits', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    # Hide any unused subplots
    for idx in range(num_layers, 28):
        axes[idx].axis('off')

    plt.suptitle('Logit Distributions (Baseline vs Obfuscated) by Layer', fontsize=14, y=0.995)
    plt.tight_layout()

    output_file = output_dir / "logit_distributions_by_layer.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved logit distributions plot to {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze probe output distributions for baseline and obfuscated activations.'
    )
    parser.add_argument(
        '--probe-dir',
        type=Path,
        required=True,
        help='Directory containing probe .joblib files'
    )
    parser.add_argument(
        '--activations-dir',
        type=Path,
        required=True,
        help='Directory containing cached activation .pt files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Directory to save output plots (default: current directory)'
    )

    args = parser.parse_args()

    # Validate input directories
    if not args.probe_dir.exists():
        raise FileNotFoundError(f"Probe directory not found: {args.probe_dir}")
    if not args.activations_dir.exists():
        raise FileNotFoundError(f"Activations directory not found: {args.activations_dir}")

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Probe directory: {args.probe_dir}")
    print(f"Activations directory: {args.activations_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load probes
    probes_by_layer = load_probes(args.probe_dir)

    # Process activations
    (baseline_predictions_by_layer, baseline_logits_by_layer,
     obfuscated_predictions_by_layer, obfuscated_logits_by_layer) = process_activations(
        args.activations_dir, probes_by_layer
    )

    # Compute and plot average predictions
    baseline_avg_preds, obfuscated_avg_preds = compute_average_predictions(
        baseline_predictions_by_layer, obfuscated_predictions_by_layer
    )
    plot_average_predictions(baseline_avg_preds, obfuscated_avg_preds, args.output_dir)

    # Plot logit distributions
    plot_logit_distributions(baseline_logits_by_layer, obfuscated_logits_by_layer, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
