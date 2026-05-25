import torch
import torch.nn.functional as F
from torch import nn

from .rotary import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    """A multi-head attention module with rotary positional embeddings."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.d_head: int = self.d_model // self.n_heads

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3, bias=False),
        )
        self.q_ln = nn.LayerNorm(d_model, bias=False)
        self.k_ln = nn.LayerNorm(d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Assume max sequence length of 20k, which is sufficient for most sequences.
        self.rotary = RotaryEmbedding(self.d_head, max_seqlen=20000)

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor,
        pos_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of multi-head attention.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (*, L, D).
        seq_id: torch.Tensor
            Sequence ids of shape (*, L) for attention masking.
        pos_id: torch.Tensor
            Position ids of shape (*, L) for rotary positional embeddings.

        Returns
        -------
        out: torch.Tensor
            Output tensor of shape (*, L, D).
        attn_weights: torch.Tensor | None
            Attention weights of shape (*, H, L, L), where H is number of heads.
        """
        H, Dh = self.n_heads, self.d_head

        # [*, L, D] -> 3 * [*, L, D]
        q, k, v = self.layernorm_qkv(x).chunk(3, dim=-1)
        q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(k.dtype)

        # [*, L, D] -> [*, L, H, Dh]
        q, k, v = map(lambda t: t.unflatten(-1, (H, Dh)), (q, k, v))
        q, k = self.rotary(q, k, pos_id)

        # [B, L, H, Dh] -> [B, H, L, Dh]
        q, k, v = map(lambda t: t.transpose(-2, -3), (q, k, v))

        attn_mask = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
        attn_mask = attn_mask.unsqueeze(-3)  # [B, 1, L, L]

        # [B, H, L, Dh] @ [B, H, Dh, L] -> [B, H, L, L]
        q *= Dh**-0.5  # Scale query by sqrt(d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [*, H, L, L]
        attn_weights.masked_fill_(~attn_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1).to(v.dtype)
        out = torch.matmul(attn_weights, v)  # [*, H, L, Dh]

        # [*, H, L, Dh] -> [*, L, H, Dh] -> [*, L, D]
        out = out.transpose(-2, -3).flatten(-2)
        out = self.out_proj(out)
        return out, attn_weights
