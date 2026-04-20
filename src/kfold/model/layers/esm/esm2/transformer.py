import math

import torch
import torch.nn as nn

from .attention import MultiHeadAttention

sqrt_2 = math.sqrt(2.0)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / sqrt_2))


class TransformerLayer(nn.Module):
    """A ESM2 transformer layer

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_ratio: int = 4,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * expansion_ratio)
        self.fc2 = nn.Linear(d_model * expansion_ratio, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor,
        pos_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(x, seq_id, pos_id)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x, attn_weights
