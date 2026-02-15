import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import json


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def load_activations(save_dir, activation_label, layers):
    """Load activations for specified layers."""
    activations = {}
    for layer in layers:
        path = save_dir / f"{activation_label}_activations_layer_{layer}.pt"
        if path.exists():
            activations[layer] = torch.load(path)
        else:
            print(f"Warning: Missing activations for layer {layer} in {activation_label}")
    return activations


def collect_generation_activations(acts, tokens, prompt_lengths, layer):
    """Collect activations from generation tokens only (excluding prompt)."""
    acts_layer = acts[layer]
    attention_mask = tokens['attention_mask']
    n_examples = acts_layer.shape[0]
    gen_acts_list = []

    for i in range(n_examples):
        pl = prompt_lengths[i].item()
        mask = attention_mask[i] == 1
        total_len = mask.sum().item()
        gen_start = pl

        if gen_start < total_len:
            gen_acts = acts_layer[i, gen_start:total_len, :]
            gen_acts_list.append(gen_acts)

    if gen_acts_list:
        return torch.cat(gen_acts_list, dim=0)
    else:
        return torch.empty(0, acts_layer.shape[-1])


def fit_pca_on_safe(safe_activations, n_components=None, variance_threshold=0.95):
    """
    Fit PCA on safe activations.

    Args:
        safe_activations: torch.Tensor of shape [n_samples, n_features]
        n_components: Fixed number of components (if None, use variance_threshold)
        variance_threshold: Keep components explaining this much variance

    Returns:
        fitted PCA model, number of components used
    """
    safe_np = safe_activations.cpu().numpy().astype(np.float32)

    if n_components is None:
        # Fit with all components first to determine how many we need
        pca_full = PCA()
        pca_full.fit(safe_np)
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1
        print(f"Using {n_components} components to explain {variance_threshold * 100}% variance")

    # Fit final PCA with determined number of components
    pca = PCA(n_components=n_components)
    pca.fit(safe_np)

    return pca, n_components


def compute_reconstruction_error(pca, activations):
    """
    Compute reconstruction error after projecting to PCA space and back.

    Returns:
        Array of reconstruction errors (one per sample)
    """
    acts_np = activations.cpu().numpy().astype(np.float32)

    # Project to PCA space
    projected = pca.transform(acts_np)

    # Reconstruct
    reconstructed = pca.inverse_transform(projected)

    # Compute per-sample reconstruction error (L2 norm)
    errors = np.linalg.norm(acts_np - reconstructed, axis=1)

    return errors


def compute_variance_captured(pca, activations):
    """
    Compute how much variance of each sample is captured by the PCA subspace.

    Returns:
        Array of variance ratios (one per sample)
    """
    acts_np = activations.cpu().numpy().astype(np.float32)

    # Center the data using PCA's mean
    centered = acts_np - pca.mean_

    # Project to PCA space
    projected = pca.transform(acts_np)

    # Reconstruct in original space
    reconstructed = pca.inverse_transform(projected) - pca.mean_

    # Compute variance captured ratio for each sample
    original_var = np.sum(centered ** 2, axis=1)
    reconstructed_var = np.sum(reconstructed ** 2, axis=1)

    # Avoid division by zero
    variance_ratios = np.divide(
        reconstructed_var,
        original_var,
        out=np.ones_like(original_var),
        where=original_var != 0
    )

    return variance_ratios


def compute_pca_metrics(pca, activations, label):
    """
    Compute multiple PCA-based metrics for a set of activations.

    Returns:
        Dictionary of metrics
    """
    recon_errors = compute_reconstruction_error(pca, activations)
    var_captured = compute_variance_captured(pca, activations)

    metrics = {
        'label': label,
        'reconstruction_error': {
            'mean': float(np.mean(recon_errors)),
            'std': float(np.std(recon_errors)),
            'median': float(np.median(recon_errors)),
            'values': recon_errors
        },
        'variance_captured': {
            'mean': float(np.mean(var_captured)),
            'std': float(np.std(var_captured)),
            'median': float(np.median(var_captured)),
            'values': var_captured
        }
    }

    return metrics


def plot_metric_distributions(metrics_list, metric_name, output_path, layer):
    """Plot distributions of a specific metric across different activation types."""
    plt.figure(figsize=(10, 6))

    for metrics in metrics_list:
        label = metrics['label']
        values = metrics[metric_name]['values']
        plt.hist(values, bins=50, alpha=0.5, label=label, density=True)

    plt.xlabel(metric_name.replace('_', ' ').title())
    plt.ylabel('Density')
    plt.title(f'{metric_name.replace("_", " ").title()} Distribution at Layer {layer}')
    plt.legend()
    plt.savefig(output_path / f'{metric_name}_distribution_layer_{layer}.png', dpi=150)
    plt.close()


def plot_roc_curves(metrics_list, output_path, layer):
    """
    Plot ROC curves for different metrics.
    Assumes first metrics in list is "safe" (negative class) and others are variants of harmful.
    """
    safe_metrics = metrics_list[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metric_names = ['reconstruction_error', 'variance_captured']

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]

        for harmful_metrics in metrics_list[1:]:
            # Create binary labels: 0 for safe, 1 for harmful
            y_true = np.concatenate([
                np.zeros(len(safe_metrics[metric_name]['values'])),
                np.ones(len(harmful_metrics[metric_name]['values']))
            ])

            # Scores: for reconstruction_error and projection_magnitude, higher = more anomalous
            # For variance_captured, lower = more anomalous, so we negate it
            safe_scores = safe_metrics[metric_name]['values']
            harmful_scores = harmful_metrics[metric_name]['values']

            if metric_name == 'variance_captured':
                # Flip so that lower variance captured = higher anomaly score
                scores = np.concatenate([-safe_scores, -harmful_scores])
            else:
                scores = np.concatenate([safe_scores, harmful_scores])

            # Compute ROC curve
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f'{harmful_metrics["label"]} (AUC={roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{metric_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f'roc_curves_layer_{layer}.png', dpi=150)
    plt.close()


def evaluate_detection_performance(safe_metrics, harmful_metrics, metric_name):
    """
    Evaluate detection performance using a specific metric.

    Returns:
        Dictionary with AUC, best threshold, and performance at that threshold
    """
    # Create labels
    y_true = np.concatenate([
        np.zeros(len(safe_metrics[metric_name]['values'])),
        np.ones(len(harmful_metrics[metric_name]['values']))
    ])

    # Create scores
    safe_scores = safe_metrics[metric_name]['values']
    harmful_scores = harmful_metrics[metric_name]['values']

    if metric_name == 'variance_captured':
        scores = np.concatenate([-safe_scores, -harmful_scores])
    else:
        scores = np.concatenate([safe_scores, harmful_scores])

    # Compute metrics
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # Find best threshold (maximize TPR - FPR)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]

    # Compute precision-recall AUC
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    return {
        'metric': metric_name,
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'best_threshold': float(best_threshold),
        'tpr_at_best': float(best_tpr),
        'fpr_at_best': float(best_fpr),
        'safe_mean': float(safe_metrics[metric_name]['mean']),
        'harmful_mean': float(harmful_metrics[metric_name]['mean'])
    }


def main(args):
    dir_path = Path(args.activations_dir)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    layers = args.layers if args.layers else list(range(32))

    # Load safe activations (reference for PCA fitting)
    print(f"Loading safe activations: {args.safe_label}")
    safe_acts = load_activations(dir_path, args.safe_label, layers)
    safe_tokens = torch.load(dir_path / f"{args.safe_label}_tokens.pt")
    safe_prompt_lens = torch.load(dir_path / f"{args.safe_label}_prompt_lengths.pt")

    # Load harmful normal activations (no backdoor)
    print(f"Loading harmful normal activations: {args.harmful_normal_label}")
    harmful_normal_acts = load_activations(dir_path, args.harmful_normal_label, layers)
    harmful_normal_tokens = torch.load(dir_path / f"{args.harmful_normal_label}_tokens.pt")
    harmful_normal_prompt_lens = torch.load(dir_path / f"{args.harmful_normal_label}_prompt_lengths.pt")

    # Load backdoored harmful activations (obfuscated)
    print(f"Loading backdoored activations: {args.backdoored_label}")
    backdoored_acts = load_activations(dir_path, args.backdoored_label, layers)
    backdoored_tokens = torch.load(dir_path / f"{args.backdoored_label}_tokens.pt")
    backdoored_prompt_lens = torch.load(dir_path / f"{args.backdoored_label}_prompt_lengths.pt")

    results_per_layer = {}

    for layer in layers:
        if layer not in safe_acts:
            print(f"Skipping layer {layer}: Missing activations")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing Layer {layer}")
        print(f"{'=' * 60}")

        # Collect generation activations
        safe_gen = collect_generation_activations(safe_acts, safe_tokens, safe_prompt_lens, layer)
        harmful_normal_gen = collect_generation_activations(
            harmful_normal_acts, harmful_normal_tokens, harmful_normal_prompt_lens, layer
        )
        backdoored_gen = collect_generation_activations(
            backdoored_acts, backdoored_tokens, backdoored_prompt_lens, layer
        )

        if safe_gen.numel() == 0:
            print(f"Skipping layer {layer}: No safe generation activations")
            continue

        print(f"Safe activations: {safe_gen.shape}")
        print(f"Harmful normal activations: {harmful_normal_gen.shape}")
        print(f"Backdoored activations: {backdoored_gen.shape}")

        # Fit PCA on safe activations
        print(f"\nFitting PCA on safe activations...")
        pca, n_components = fit_pca_on_safe(
            safe_gen,
            n_components=args.n_components,
            variance_threshold=args.variance_threshold
        )

        # Compute metrics for all activation types
        print(f"\nComputing PCA metrics...")
        safe_metrics = compute_pca_metrics(pca, safe_gen, "Safe")
        harmful_normal_metrics = compute_pca_metrics(pca, harmful_normal_gen, "Harmful (No Backdoor)")
        backdoored_metrics = compute_pca_metrics(pca, backdoored_gen, "Backdoored (Obfuscated)")

        metrics_list = [safe_metrics, harmful_normal_metrics, backdoored_metrics]

        # Print summary statistics
        print(f"\n{'Metric':<25} {'Safe':<20} {'Harmful Normal':<20} {'Backdoored':<20}")
        print(f"{'-' * 85}")
        for metric_name in ['reconstruction_error', 'variance_captured']:
            print(f"{metric_name:<25} "
                  f"{safe_metrics[metric_name]['mean']:>8.4f}±{safe_metrics[metric_name]['std']:>6.4f}  "
                  f"{harmful_normal_metrics[metric_name]['mean']:>8.4f}±{harmful_normal_metrics[metric_name]['std']:>6.4f}  "
                  f"{backdoored_metrics[metric_name]['mean']:>8.4f}±{backdoored_metrics[metric_name]['std']:>6.4f}")

        # Evaluate detection performance
        print(f"\n{'=' * 60}")
        print(f"Detection Performance (Safe vs Backdoored)")
        print(f"{'=' * 60}")

        detection_results = {}
        for metric_name in ['reconstruction_error', 'variance_captured']:
            perf = evaluate_detection_performance(safe_metrics, backdoored_metrics, metric_name)
            detection_results[metric_name] = perf
            print(f"\n{metric_name}:")
            print(f"  ROC-AUC: {perf['roc_auc']:.4f}")
            print(f"  PR-AUC: {perf['pr_auc']:.4f}")
            print(f"  Best Threshold: {perf['best_threshold']:.4f}")
            print(f"  TPR at best: {perf['tpr_at_best']:.3f}, FPR at best: {perf['fpr_at_best']:.3f}")

        # Plot distributions
        print(f"\nGenerating plots...")
        for metric_name in ['reconstruction_error', 'variance_captured']:
            plot_metric_distributions(metrics_list, metric_name, output_path, layer)

        # Plot ROC curves
        plot_roc_curves(metrics_list, output_path, layer)

        # Store results
        results_per_layer[layer] = {
            'n_components': n_components,
            'explained_variance_ratio': float(np.sum(pca.explained_variance_ratio_)),
            'detection_results': detection_results,
            'summary_stats': {
                'safe': {k: v for k, v in safe_metrics.items() if k != 'label'},
                'harmful_normal': {k: v for k, v in harmful_normal_metrics.items() if k != 'label'},
                'backdoored': {k: v for k, v in backdoored_metrics.items() if k != 'label'}
            }
        }

        # Remove numpy arrays before saving JSON
        for key in results_per_layer[layer]['summary_stats']:
            for metric in results_per_layer[layer]['summary_stats'][key]:
                if 'values' in results_per_layer[layer]['summary_stats'][key][metric]:
                    del results_per_layer[layer]['summary_stats'][key][metric]['values']

    # Save results
    print(f"\nSaving results to {output_path / 'pca_detection_results.json'}")
    # Convert all numpy types to Python native types for JSON serialization
    results_serializable = convert_to_json_serializable(results_per_layer)
    with open(output_path / 'pca_detection_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # Create summary plot across layers
    if len(results_per_layer) > 1:
        plot_layer_summary(results_per_layer, output_path)

    print(f"\n{'=' * 60}")
    print(f"Analysis complete! Results saved to {output_path}")
    print(f"{'=' * 60}")


def plot_layer_summary(results_per_layer, output_path):
    """Plot summary of detection performance across layers."""
    layers = sorted(results_per_layer.keys())

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: ROC-AUC across layers for each metric
    ax = axes[0, 0]
    for metric_name in ['reconstruction_error', 'variance_captured']:
        aucs = [results_per_layer[l]['detection_results'][metric_name]['roc_auc'] for l in layers]
        ax.plot(layers, aucs, marker='o', label=metric_name.replace('_', ' ').title())
    ax.set_xlabel('Layer')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Detection Performance Across Layers')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.3)

    # Plot 2: Number of components needed
    ax = axes[0, 1]
    n_components = [results_per_layer[l]['n_components'] for l in layers]
    ax.plot(layers, n_components, marker='o', color='purple')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of PCA Components')
    ax.set_title('PCA Dimensionality Across Layers')
    ax.grid(alpha=0.3)

    # Plot 3: Mean reconstruction error by type
    ax = axes[1, 0]
    safe_errors = [results_per_layer[l]['summary_stats']['safe']['reconstruction_error']['mean'] for l in layers]
    backdoored_errors = [results_per_layer[l]['summary_stats']['backdoored']['reconstruction_error']['mean'] for l in
                         layers]
    ax.plot(layers, safe_errors, marker='o', label='Safe')
    ax.plot(layers, backdoored_errors, marker='s', label='Backdoored')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Reconstruction Error')
    ax.set_title('Reconstruction Error Across Layers')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Mean variance captured by type
    ax = axes[1, 1]
    safe_var = [results_per_layer[l]['summary_stats']['safe']['variance_captured']['mean'] for l in layers]
    backdoored_var = [results_per_layer[l]['summary_stats']['backdoored']['variance_captured']['mean'] for l in layers]
    ax.plot(layers, safe_var, marker='o', label='Safe')
    ax.plot(layers, backdoored_var, marker='s', label='Backdoored')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Variance Captured')
    ax.set_title('Variance Captured Across Layers')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'layer_summary.png', dpi=150)
    plt.close()
    print(f"Saved layer summary plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect obfuscated activations using PCA-based methods"
    )
    parser.add_argument("--activations_dir", type=str, required=True,
                        help="Directory where activations are saved")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory where to save results")
    parser.add_argument("--safe_label", type=str, default="safe",
                        help="Label for safe activations (reference for PCA)")
    parser.add_argument("--harmful_normal_label", type=str, default="unsafe_normal",
                        help="Label for harmful activations without backdoor")
    parser.add_argument("--backdoored_label", type=str, default="backdoored_unsafe",
                        help="Label for backdoored (obfuscated) activations")
    parser.add_argument("--layers", type=int, nargs='+', default=None,
                        help="Specific layers to analyze (default: all 0-31)")
    parser.add_argument("--n_components", type=int, default=None,
                        help="Fixed number of PCA components (if None, use variance_threshold)")
    parser.add_argument("--variance_threshold", type=float, default=0.95,
                        help="Variance threshold for determining number of components (default: 0.95)")

    args = parser.parse_args()
    main(args)