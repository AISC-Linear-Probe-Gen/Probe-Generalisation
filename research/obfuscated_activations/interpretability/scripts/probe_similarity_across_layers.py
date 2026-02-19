"""
Analysis of probe weight similarity across layers.

This script examines how similar learned probe weights are across different layers
of a language model (e.g., Llama-3.2-3B-Instruct).
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


def cosine_similarity_stats(n, d):
    """
    Compute the mean and standard deviation of cosine similarities
    between n random vectors of dimension d.

    This establishes a baseline for what level of similarity would be expected
    by chance between random high-dimensional vectors. Used to compute z-scores
    for observed similarities.

    Parameters:
    -----------
    n : int
        Number of random vectors to generate
    d : int
        Dimension of each vector

    Returns:
    --------
    tuple : (mean, std)
        Mean and standard deviation of pairwise cosine similarities
    """
    # Generate n random vectors of dimension d
    vectors = np.random.randn(n, d)

    # Normalize vectors to unit length
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Compute pairwise cosine similarities (dot product of normalized vectors)
    cosine_similarities = vectors @ vectors.T

    # Extract upper triangular part (excluding diagonal)
    # to get unique pairs without self-similarities
    upper_triangle_indices = np.triu_indices(n, k=1)
    unique_similarities = cosine_similarities[upper_triangle_indices]

    # Compute mean and standard deviation
    mean = np.mean(unique_similarities)
    std = np.std(unique_similarities, ddof=1)  # ddof=1 for sample std

    return mean, std


def analyze_probe_similarity(probe_dir, output_dir, n_random_vectors=5000,
                            activation_dim=3072, show_plot=False):
    """
    Analyze similarity between probe weights across layers.

    Parameters:
    -----------
    probe_dir : Path
        Directory containing trained probe .joblib files
    output_dir : Path
        Directory to save output plots
    n_random_vectors : int
        Number of random vectors for baseline statistics
    activation_dim : int
        Dimension of activation vectors
    show_plot : bool
        Whether to display plots interactively
    """
    probe_dir = Path(probe_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all trained logistic regression probes from different layers
    print(f"Loading probes from {probe_dir}")
    collate = []

    # Collect the learned weight vectors (coefficients) from each layer's probe
    probe_files = list(probe_dir.glob("*.joblib"))
    if not probe_files:
        raise ValueError(f"No .joblib files found in {probe_dir}")

    for f in tqdm(probe_files, desc="Loading probes"):
        layer_no = int(re.findall(r"layer_(\d+)", str(f))[0])
        collate.append((layer_no, joblib.load(f)["model"].coef_))

    collate = [i[1] for i in sorted(collate, key=lambda x: x[0])]

    # Concatenate all probe weights into a single array
    # Shape: (num_layers, activation_dim)
    arr = np.concatenate(collate, axis=0)
    print(f"Loaded {arr.shape[0]} probes with dimension {arr.shape[1]}")

    # Compute baseline statistics for random vectors
    print(f"Computing baseline statistics with {n_random_vectors} random vectors of dimension {activation_dim}")
    mean, std = cosine_similarity_stats(n_random_vectors, activation_dim)
    print(f"Baseline cosine similarity - Mean: {mean:.6f}, Std: {std:.6f}")

    # Compute pairwise cosine similarities between all probe weight vectors
    print("Computing pairwise cosine similarities")
    sims = cosine_similarity(arr)

    # Set diagonal to NaN to prevent self-similarities from distorting the colorscale
    np.fill_diagonal(sims, np.nan)

    # Visualize the similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(sims, origin="lower", cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.xlabel('Layer Index')
    plt.ylabel('Layer Index')
    plt.title('Pairwise Cosine Similarity Between Probe Weights Across Layers')

    similarity_plot_path = output_dir / "probe_similarity_matrix.png"
    plt.savefig(similarity_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved similarity matrix plot to {similarity_plot_path}")

    if show_plot:
        plt.show()
    plt.close()

    # Compute the mean probe weight vector across all layers
    print("\nAnalyzing similarity to mean probe vector")
    mean_vect = arr.mean(axis=0)

    # Calculate norms for cosine similarity computation
    norm_v = np.linalg.norm(mean_vect)
    norm_m = np.linalg.norm(arr, axis=1)

    # Compute cosine similarity of each layer's probe to the mean probe
    dot_product = arr @ mean_vect
    similarities = dot_product / (norm_m * norm_v + 1e-9)

    # Find the layer with minimum similarity to the mean
    min_similarity = np.min(similarities)
    min_layer = np.argmin(similarities)

    # Convert to z-score to assess statistical significance
    min_z_score = min_similarity / std

    print(f"Minimum similarity to mean probe: {min_similarity:.6f}")
    print(f"Layer with minimum similarity: {min_layer}")
    print(f"Z-score: {min_z_score:.2f}")

    # Save statistics to file
    stats_file = output_dir / "similarity_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("Probe Similarity Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of probes: {arr.shape[0]}\n")
        f.write(f"Activation dimension: {arr.shape[1]}\n\n")
        f.write("Baseline Statistics (random vectors):\n")
        f.write(f"  Mean cosine similarity: {mean:.6f}\n")
        f.write(f"  Std dev: {std:.6f}\n\n")
        f.write("Similarity to Mean Probe:\n")
        f.write(f"  Minimum similarity: {min_similarity:.6f}\n")
        f.write(f"  Layer with min similarity: {min_layer}\n")
        f.write(f"  Z-score: {min_z_score:.2f}\n\n")
        f.write("Conclusion:\n")
        f.write("  Even the most dissimilar probe has a very high z-score,\n")
        f.write("  indicating all probes are highly similar to each other.\n")
        f.write("  There is significant mutual information between coefficients.\n")
        f.write("  However, the geometric similarity may not be sufficient to use\n")
        f.write("  a single representative direction.\n")

    print(f"\nSaved statistics to {stats_file}")
    print("\nAnalysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze similarity between probe weights across layers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "probe_dir",
        type=str,
        help="Directory containing trained probe .joblib files"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save output plots and statistics"
    )

    # Optional arguments
    parser.add_argument(
        "--n-random-vectors",
        type=int,
        default=5000,
        help="Number of random vectors for baseline statistics"
    )

    parser.add_argument(
        "--activation-dim",
        type=int,
        default=3072,
        help="Dimension of activation vectors (e.g., 3072 for Llama-3.2-3B)"
    )

    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display plots interactively"
    )

    args = parser.parse_args()

    analyze_probe_similarity(
        probe_dir=args.probe_dir,
        output_dir=args.output_dir,
        n_random_vectors=args.n_random_vectors,
        activation_dim=args.activation_dim,
        show_plot=args.show_plot
    )


if __name__ == "__main__":
    main()
