import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention


class SwiGLU(nn.Module):
    """SwiGLU activation function as an nn.Module"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class TransformerBlock(nn.Module):
    """A transformer block

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    n_layers : int
        The number of layers in the transformer block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        return_attn: bool = False,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, return_attn=return_attn)
        d_ffn = swiglu_correction_fn(expansion_ratio, d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ffn * 2, bias=False),
            SwiGLU(),
            nn.Linear(d_ffn, d_model, bias=False),
        )
        self.scaling_factor: float = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor,
        pos_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r1, attn = self.attn(x, seq_id, pos_id)
        x = x + r1 / self.scaling_factor
        r2 = self.ffn(x)
        x = x + r2 / self.scaling_factor
        return x, attn
