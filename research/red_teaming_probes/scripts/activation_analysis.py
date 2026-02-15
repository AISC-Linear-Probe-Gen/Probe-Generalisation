import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, in_features):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


def load_activations(save_dir, activation_label, layers):
    activations = {}
    for layer in layers:
        path = save_dir / f"{activation_label}_activations_layer_{layer}.pt"
        if path.exists():
            activations[layer] = torch.load(path)
        else:
            print(f"Warning: Missing activations for layer {layer} in {activation_label}")
    return activations


def collect_generation_activations(acts, tokens, prompt_lengths, layer):
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
        return torch.empty(0, acts_layer.shape[-1], device=acts_layer.device if hasattr(acts_layer, 'device') else None)


def compute_pca_cum_var(acts):
    if acts.numel() == 0:
        return np.array([])
    device = acts.device if hasattr(acts, 'device') else torch.device('cpu')
    acts = acts.to(device)
    acts_np = acts.cpu().numpy().astype(np.float32)  # Convert to float32 for SVD compatibility
    mean = np.mean(acts_np, axis=0)
    centered = acts_np - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = (S ** 2) / np.sum(S ** 2)
    cum_var = np.cumsum(explained_var)
    return cum_var

def plot_delta_norm_over_layers(delta_norms, layers, output_path):
    plt.figure()
    plt.plot(layers, delta_norms)
    plt.xlabel('Model Layer')
    plt.ylabel('Normalized ||Δh||')
    plt.title('Trigger-Induced Activation Shift Magnitude')
    plt.savefig(output_path / 'delta_norm_plot.png')
    plt.close()


def plot_pca_cum_var(normal_cum_var, triggered_cum_var, probe_layer, output_path):
    plt.figure()
    # Fixed: Compute separate components for each to avoid length mismatch
    normal_components = np.arange(1, len(normal_cum_var) + 1)
    triggered_components = np.arange(1, len(triggered_cum_var) + 1)
    plt.plot(triggered_components, triggered_cum_var, label='Triggered')
    plt.plot(normal_components, normal_cum_var, label='Normal')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title(f'PCA Cumulative Variance at Layer {probe_layer}')
    plt.axhline(0.9, color='r', linestyle='--', label='90% variance')
    plt.axhline(0.95, color='g', linestyle='--', label='95% variance')
    # plt.xlim(1, 300)  # Limit to 100 components as per user request
    plt.ylim(0, 1)

    # Find and annotate k for 90% and 95%
    def find_k(cum_var, threshold):
        k = np.argmax(cum_var >= threshold) + 1
        if k > len(cum_var):
            return None
        return k

    triggered_k90 = find_k(triggered_cum_var, 0.9)
    triggered_k95 = find_k(triggered_cum_var, 0.95)
    normal_k90 = find_k(normal_cum_var, 0.9)
    normal_k95 = find_k(normal_cum_var, 0.95)

    if triggered_k90:
        plt.axvline(triggered_k90, color='r', linestyle=':', alpha=0.5)
        plt.text(triggered_k90, 0.92, f'k={triggered_k90}', rotation=90, va='bottom', ha='right', color='blue')
    if triggered_k95:
        plt.axvline(triggered_k95, color='g', linestyle=':', alpha=0.5)
        plt.text(triggered_k95, 0.97, f'k={triggered_k95}', rotation=90, va='bottom', ha='right', color='blue')

    if normal_k90:
        plt.axvline(normal_k90, color='r', linestyle=':', alpha=0.5)
        plt.text(normal_k90, 0.88, f'k={normal_k90}', rotation=90, va='top', ha='left', color='orange')
    if normal_k95:
        plt.axvline(normal_k95, color='g', linestyle=':', alpha=0.5)
        plt.text(normal_k95, 0.93, f'k={normal_k95}', rotation=90, va='top', ha='left', color='orange')

    plt.legend()
    plt.savefig(output_path / 'pca_variance_plot.png')
    plt.close()

def plot_dot_products(dot_products, layers, probe_layer, output_path):
    plt.figure()
    plt.plot(layers, dot_products, label='Triggered (Harmful)')
    # Add random trigger if available (placeholder)
    # plt.plot(layers, random_dots, label='Random (jokey)', color='orange')
    plt.xlabel('Model Layer')
    plt.ylabel('Dot product with probe direction')
    plt.title('Activation Shift Alignment with Probe')
    plt.axvline(probe_layer, color='r', linestyle='--', label=f'Probe Layer {probe_layer}')
    plt.legend()
    plt.savefig(output_path / 'dot_product_plot.png')
    plt.close()


def main(activations_dir, normal_label='unsafe_normal', triggered_label='backdoored_unsafe',
         reference_label='unsafe_normal', probe_layer=10):
    dir_path = Path(activations_dir)

    layers = list(range(32))  # Adjust if needed

    # Load data for normal (h_normal)
    normal_acts = load_activations(dir_path, normal_label, layers)
    normal_tokens = torch.load(dir_path / f"{normal_label}_tokens.pt")
    normal_prompt_lens = torch.load(dir_path / f"{normal_label}_prompt_lengths.pt")

    # Load data for triggered (h_triggered)
    triggered_acts = load_activations(dir_path, triggered_label, layers)
    triggered_tokens = torch.load(dir_path / f"{triggered_label}_tokens.pt")
    triggered_prompt_lens = torch.load(dir_path / f"{triggered_label}_prompt_lengths.pt")

    # Load data for reference (for normalization)
    ref_acts = load_activations(dir_path, reference_label, layers)
    ref_tokens = torch.load(dir_path / f"{reference_label}_tokens.pt")
    ref_prompt_lens = torch.load(dir_path / f"{reference_label}_prompt_lengths.pt")

    # Load probe weights (assuming single probe layer)
    probe_weights = torch.load(dir_path / "probe_weights.pt")
    probe = LinearProbe(next(iter(normal_acts.values())).shape[-1])
    probe.load_state_dict(probe_weights[probe_layer])
    v_probe = probe.linear.weight.data[0]

    loaded_layers = sorted(normal_acts.keys())

    delta_norms = []
    dot_products = []

    for layer in loaded_layers:
    # Collect generation activations
        normal_gen = collect_generation_activations(normal_acts, normal_tokens, normal_prompt_lens, layer)
        triggered_gen = collect_generation_activations(triggered_acts, triggered_tokens, triggered_prompt_lens, layer)
        ref_gen = collect_generation_activations(ref_acts, ref_tokens, ref_prompt_lens, layer)

        if normal_gen.numel() == 0 or triggered_gen.numel() == 0 or ref_gen.numel() == 0:
            print(f"Skipping layer {layer}: No generation activations")
            continue

        mean_normal = normal_gen.mean(dim=0)
        mean_triggered = triggered_gen.mean(dim=0)
        delta_h = mean_triggered - mean_normal

        avg_norm_ref = ref_gen.norm(dim=1).mean()

        normalized_delta = delta_h.norm() / avg_norm_ref
        delta_norms.append(normalized_delta.item())

        dot_prod = torch.dot(delta_h.float(), v_probe.float()).item()
        dot_products.append(dot_prod)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot delta norm over layers (Figure 5 equivalent)
    plot_delta_norm_over_layers(delta_norms, loaded_layers, output_path=output_path)

    # PCA at probe layer (Figure 6)
    normal_gen_probe = collect_generation_activations(normal_acts, normal_tokens, normal_prompt_lens, probe_layer)
    triggered_gen_probe = collect_generation_activations(triggered_acts, triggered_tokens, triggered_prompt_lens,
                                                         probe_layer)
    normal_cum_var = compute_pca_cum_var(normal_gen_probe)
    triggered_cum_var = compute_pca_cum_var(triggered_gen_probe)
    plot_pca_cum_var(normal_cum_var, triggered_cum_var, probe_layer, output_path=output_path)

    # Dot products over layers (Figure 7)
    plot_dot_products(dot_products, loaded_layers, probe_layer, output_path=output_path)

    print("Plots generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", type=str, required=True, help="Directory where activations are saved")
    parser.add_argument("--output_path", type=str, required=True, help="Directory where to save results")
    parser.add_argument("--normal_label", type=str, default="safe", help="Label for normal activations")
    parser.add_argument("--triggered_label", type=str, default="backdoored_unsafe",
                        help="Label for triggered activations")
    parser.add_argument("--reference_label", type=str, default="unsafe_normal", help="Label for reference dataset")
    parser.add_argument("--probe_layer", type=int, default=10, help="Layer where the probe is attached")
    args = parser.parse_args()
    main(args.activations_dir, args.normal_label, args.triggered_label, args.reference_label, args.probe_layer)