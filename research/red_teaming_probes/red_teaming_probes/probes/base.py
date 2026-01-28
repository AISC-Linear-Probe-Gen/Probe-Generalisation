import torch
from torch import nn, Tensor

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


class AttentionProbe(nn.Module):
    """Attention-based probe: pAttn(H) = σ(Σ_k (softmax(Hq_k)^T H)^T w_k + b)"""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(hidden_dim, num_heads, bias=False)
        self.output_proj = nn.Linear(hidden_dim * num_heads, 1)

    def forward(self, h: Tensor, attention_mask: Tensor = None) -> Tensor:
        h = h.float()
        if h.dim() == 2:
            h = h.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Attention scores: (batch, seq, num_heads)
        attn_scores = self.query_proj(h)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)

        # Attention weights and contexts: (batch, num_heads, hidden)
        attn_weights = torch.softmax(attn_scores, dim=1)
        contexts = attn_weights.transpose(1, 2) @ h  # (batch, num_heads, hidden)

        # Flatten and project to output
        output = self.output_proj(contexts.flatten(1)).squeeze(-1)

        return output.squeeze(0) if squeeze else output

    def predict_proba(self, h: Tensor, attention_mask: Tensor = None) -> Tensor:
        return torch.sigmoid(self.forward(h, attention_mask))
