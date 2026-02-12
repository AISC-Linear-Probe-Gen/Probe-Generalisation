import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from typing import Union, List


def train_probe(
    probe: nn.Module,
    train_activations: Tensor,
    train_labels: Tensor,
    val_activations: Tensor | None = None,
    val_labels: Tensor | None = None,
    lr: float = 1e-3,
    epochs: int = 200,
    l2_lambda :int = 0,
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
        l2_lambda: regularization strength
        batch_size: Batch size
        patience: Early stopping patience (only used if use_early_stopping=True)
        device: Device
        use_early_stopping: If True, stop when val AUROC stops improving

    Returns:
        Dict with training history and the best metrics
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

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=l2_lambda)
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
        activations: Union[Tensor, List[Tensor]],
        labels: Tensor,
        threshold: float | None = None,
        fpr_threshold: float = 0.01,
        device: str = "cuda",
) -> dict:
    """
    Evaluate probe.

    Handles either flattened/mean-pooled Tensor or List[Tensor] (per sample for "all" pooling).
    For List, averages probs per sample before metrics.

    If threshold is provided → use it (paper protocol).
    If threshold is None → calibrate from this dataset.
    """
    probe.eval()

    if isinstance(activations, list):
        # Per-sample: Average probs across tokens
        sample_probs = []
        with torch.no_grad():
            for acts in activations:
                if len(acts) == 0:
                    sample_probs.append(0.5)  # Neutral for empty
                else:
                    acts = acts.float().to(device)
                    token_probs = probe.predict_proba(acts)  # Includes norm + sigmoid
                    sample_probs.append(token_probs.mean().item())
        probs = np.array(sample_probs)
    else:
        # Original: Flattened or mean-pooled
        activations = activations.float()
        with torch.no_grad():
            probs = probe.predict_proba(activations.to(device)).cpu().numpy()

    labels_np = labels.numpy()
    auroc = roc_auc_score(labels_np, probs)

    # CASE 1: fixed threshold
    if threshold is not None:
        preds = (probs >= threshold).astype(int)
        tpr = ((preds == 1) & (labels_np == 1)).sum() / max((labels_np == 1).sum(), 1)
        fpr = ((preds == 1) & (labels_np == 0)).sum() / max((labels_np == 0).sum(), 1)
        return {
            "auroc": auroc,
            "tpr_at_1pct_fpr": tpr,
            "threshold": threshold,
            "actual_fpr": fpr,
        }

    # CASE 2: calibrate from this dataset (fallback)
    fpr, tpr, thresholds = roc_curve(labels_np, probs)
    idx = np.searchsorted(fpr, fpr_threshold)
    idx = min(idx, len(thresholds) - 1)
    return {
        "auroc": auroc,
        "tpr_at_1pct_fpr": tpr[idx],
        "threshold": thresholds[idx],
        "actual_fpr": fpr[idx],
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

    # Find the threshold such that fpr_target fraction of negatives is above it
    threshold = np.percentile(neg_probs, 100 * (1 - fpr_target))

    return float(threshold)
