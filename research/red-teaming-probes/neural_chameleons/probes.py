import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

class LinearProbe(nn.Module):
    """Simple logistic regression probe: p(y|h) = σ(w·h + b)"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, h: Tensor) -> Tensor:
        """Returns logits (pre-sigmoid)."""
        return self.linear(h.float()).squeeze(-1)

    def predict_proba(self, h: Tensor) -> Tensor:
        """Returns probability of positive class."""
        return torch.sigmoid(self.forward(h.float()))

class MLPProbe(nn.Module):
    """MLP probe with one hidden layer: σ(w₂·ReLU(W₁·h + b₁) + b₂)"""

    def __init__(self, hidden_dim: int, mlp_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h.float()).squeeze(-1)

    def predict_proba(self, h: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(h.float()))

def train_probe(
    probe: nn.Module,
    train_activations: Tensor,
    train_labels: Tensor,
    val_activations: Tensor | None = None,
    val_labels: Tensor | None = None,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 64,
    patience: int = 50,
    device: str = "cuda",
    use_early_stopping: bool = False,  # Disabled by default - AUROC doesn't ensure calibration
) -> dict:
    """Train a probe until convergence.

    By default, trains for full epochs without early stopping. AUROC-based early
    stopping can be enabled but doesn't guarantee good calibration (probe may rank
    correctly but output high probabilities for everything).

    Args:
        probe: LinearProbe or MLPProbe
        train_activations: (n_train, hidden_dim)
        train_labels: (n_train,) binary labels
        val_activations: Optional validation set
        val_labels: Optional validation labels
        lr: Learning rate
        epochs: Max epochs
        batch_size: Batch size
        patience: Early stopping patience (only used if use_early_stopping=True)
        device: Device
        use_early_stopping: If True, stop when val AUROC stops improving

    Returns:
        Dict with training history and best metrics
    """
    # Convert to float32 for stable training (model may output bfloat16)
    train_activations = train_activations.float()
    if val_activations is not None:
        val_activations = val_activations.float()

    # Handle NaN/Inf from quantized models
    train_activations = torch.nan_to_num(train_activations, nan=0.0, posinf=1e6, neginf=-1e6)
    if val_activations is not None:
        val_activations = torch.nan_to_num(val_activations, nan=0.0, posinf=1e6, neginf=-1e6)

    probe = probe.to(device)
    # Use Adam without weight decay for better convergence on small linear probes
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Full batch training for linear probes (more stable), mini-batch for MLPs
    effective_batch = min(batch_size, len(train_activations))
    train_dataset = TensorDataset(train_activations, train_labels.float())
    train_loader = DataLoader(train_dataset, batch_size=effective_batch, shuffle=True)

    best_auroc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_auroc": []}

    for epoch in range(epochs):
        # Training
        probe.train()
        total_loss = 0.0
        for batch_acts, batch_labels in train_loader:
            batch_acts = batch_acts.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = probe(batch_acts)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # Validation
        if val_activations is not None:
            probe.eval()
            with torch.no_grad():
                val_probs = probe.predict_proba(val_activations.to(device)).cpu().numpy()
            # Handle NaN in predictions (can happen with extreme activations)
            val_probs = np.nan_to_num(val_probs, nan=0.5, posinf=1.0, neginf=0.0)
            val_probs = np.clip(val_probs, 0.0, 1.0)
            auroc = roc_auc_score(val_labels.numpy(), val_probs)
            history["val_auroc"].append(auroc)

            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                patience_counter = 0
            elif use_early_stopping:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    # Only restore best state if using early stopping
    if use_early_stopping and best_state is not None:
        probe.load_state_dict(best_state)

    return {
        "history": history,
        "best_auroc": best_auroc,
        "epochs_trained": len(history["train_loss"]),
    }


def evaluate_probe(
    probe: nn.Module,
    activations: Tensor,
    labels: Tensor,
    fpr_threshold: float = 0.01,
    device: str = "cuda",
) -> dict:
    """Evaluate probe at a fixed FPR threshold.

    Args:
        probe: Trained probe
        activations: Test activations
        labels: Test labels
        fpr_threshold: Target false positive rate (default 1%)
        device: Device

    Returns:
        Dict with AUROC, TPR at threshold, threshold value
    """
    activations = activations.float()  # Ensure float32
    probe.eval()
    with torch.no_grad():
        probs = probe.predict_proba(activations.to(device)).cpu().numpy()

    labels_np = labels.numpy()
    auroc = roc_auc_score(labels_np, probs)

    # Find threshold for target FPR
    fpr, tpr, thresholds = roc_curve(labels_np, probs)

    # Find threshold closest to target FPR
    idx = np.searchsorted(fpr, fpr_threshold)
    if idx >= len(thresholds):
        idx = len(thresholds) - 1

    threshold = thresholds[idx]
    tpr_at_threshold = tpr[idx]
    actual_fpr = fpr[idx]

    return {
        "auroc": auroc,
        "tpr_at_1pct_fpr": tpr_at_threshold,
        "threshold": threshold,
        "actual_fpr": actual_fpr,
    }


def predict_sequence_proba(
    probe: nn.Module,
    token_activations: Tensor,
    attention_mask: Tensor | None = None,
    device: str = "cuda",
) -> Tensor:
    """Apply probe to each token and average predictions (paper method).

    Args:
        probe: Trained probe
        token_activations: (batch, seq_len, hidden_dim) per-token activations
        attention_mask: (batch, seq_len) mask for valid tokens (1=valid, 0=pad)
        device: Device

    Returns:
        (batch,) averaged probabilities per sequence
    """
    probe.eval()
    batch_size, seq_len, hidden_dim = token_activations.shape

    # Flatten to (batch * seq_len, hidden_dim) for probe
    flat_acts = token_activations.view(-1, hidden_dim).to(device)

    with torch.no_grad():
        flat_probs = probe.predict_proba(flat_acts)  # (batch * seq_len,)

    # Reshape back to (batch, seq_len)
    token_probs = flat_probs.view(batch_size, seq_len)

    if attention_mask is not None:
        # Mask out padding tokens
        mask = attention_mask.float().to(device)
        masked_probs = token_probs * mask
        seq_probs = masked_probs.sum(dim=1) / mask.sum(dim=1)
    else:
        seq_probs = token_probs.mean(dim=1)

    return seq_probs.cpu()


def calibrate_threshold(
    probe: nn.Module,
    clean_activations: Tensor,
    clean_labels: Tensor,
    fpr_target: float = 0.01,
    device: str = "cuda",
) -> float:
    """Calibrate probe threshold on clean (non-triggered) data.

    Returns threshold that achieves target FPR on negative examples.
    """
    clean_activations = clean_activations.float()  # Ensure float32
    probe.eval()
    with torch.no_grad():
        probs = probe.predict_proba(clean_activations.to(device)).cpu().numpy()

    labels_np = clean_labels.numpy()

    # Get probabilities for negative class only
    neg_probs = probs[labels_np == 0]

    # Find threshold such that fpr_target fraction of negatives are above it
    threshold = np.percentile(neg_probs, 100 * (1 - fpr_target))

    return float(threshold)
