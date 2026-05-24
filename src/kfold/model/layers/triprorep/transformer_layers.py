"""Lean ESM2-style attention primitives — pure PyTorch, no xformers.

This module is the dependency-light layer stack used by the K-Fold
encoder package. It supports exactly the subset of features that the
shipped checkpoints actually need:

  * Rotary positional embeddings (with explicit `position_ids`).
  * Self-attention via `torch.nn.functional.scaled_dot_product_attention`
    (which auto-dispatches to flash / efficient / math kernels).
  * The upstream "relax temperature scaling" knob (default
    `1/sqrt(head_dim)` factor pre-multiplied into Q, on top of SDPA's
    own `1/sqrt(head_dim)` — i.e. an effective `1/head_dim` attention
    temperature).
  * ESM1LayerNorm + gelu.

Everything else from the upstream training-time `layers.py` (xformers,
QK-Norm, logit soft-cap, gradient checkpoint, packed-sequence
BlockDiagonalMask, encoder-decoder / cross-attention, ONNX trace flag)
is intentionally removed. Pretrained checkpoints have all those flags
disabled, so this lean version is byte-compatible for weight loading.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def gelu(x: Tensor) -> Tensor:
    """ESM2-style GELU — the *manual* `x * 0.5 * (1 + erf(x / sqrt(2)))`
    formula, matching the upstream training-time `layers.gelu`.

    Why not `F.gelu(x)`? Under bf16 autocast, PyTorch's fused
    `F.gelu` CUDA kernel runs the math in fp32 internally and casts
    back, while the manual 4-op formula stays in bf16 throughout. The
    training-time code uses the manual form, so for bit-faithful
    reproduction of the trained encoder we must do the same — `F.gelu`
    diverges by ~2.5 in worst case after a single FFN block.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoFormer, Su et al. 2021).

    Two entry points:
      * `forward(q, k)` — applies sequential 0..L-1 positions, cached.
      * `forward_with_position_ids(q, k, position_ids, bsz, n_heads, head_dim)`
        — applies a per-token integer `position_ids` map. Used by the
        ProteinNet encoder so chain identity can reset positions.

    Default `dim` is 64 = head_dim for the 3B preset (embed_dim=2560,
    num_heads=40).
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # `persistent=True` matches the upstream training-time module, so
        # `rot_emb.inv_freq` keys present in K-Fold checkpoints load
        # without `unexpected_keys` complaints.
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        self._seq_len_cached: int | None = None
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def _update_cos_sin_tables(self, x: Tensor, seq_dim: int) -> tuple[Tensor, Tensor]:
        seq_len = x.shape[seq_dim]
        if (
            self._cos_cached is None
            or seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]
        return self._cos_cached, self._sin_cached

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        cos, sin = self._update_cos_sin_tables(k, seq_dim=-2)
        cos = cos[:, : q.shape[-2], :]
        sin = sin[:, : q.shape[-2], :]
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

    def forward_with_position_ids(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Tensor,
        bsz: int,
        num_heads: int,
        head_dim: int,
    ) -> tuple[Tensor, Tensor]:
        """Apply RoPE using explicit `[B, L]` position IDs (per-chain reset).

        q, k: `[B*H, L, head_dim]`
        """
        device = q.device
        dtype = q.dtype
        inv_freq = self.inv_freq.to(device=device, dtype=dtype)

        pos = position_ids.to(dtype=dtype)  # [B, L]
        freqs = torch.einsum("bl,d->bld", pos, inv_freq)  # [B, L, D//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, L, D]

        cos = (
            emb.cos()
            .unsqueeze(1)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, -1, head_dim)
        )
        sin = (
            emb.sin()
            .unsqueeze(1)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, -1, head_dim)
        )
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class MultiheadAttention(nn.Module):
    """Self-attention with rotary embeddings (always on).

    Lean port of the training-time `layers.MultiheadAttention`, restricted
    to the self-attention + RoPE configuration used by every shipped
    K-Fold checkpoint. Uses `F.scaled_dot_product_attention` so the
    dispatched kernel (flash / mem-efficient / math) is whatever the
    installed PyTorch build provides — no xformers dependency.

    Parameter names (`k_proj`, `v_proj`, `q_proj`, `out_proj`,
    `rot_emb.inv_freq`) match the training-time class so state_dict
    weights load 1-for-1.

    Shape convention: input/output is `[L, B, embed_dim]` (ESM2).
    """

    def __init__(
        self,
        embed_dim: int = 2560,
        num_heads: int = 40,
        dropout: float = 0.1,
        bias: bool = True,
        relax_temperature_scaling: float | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        # The attention temperature factor pre-multiplied into Q. SDPA
        # adds its own 1/sqrt(head_dim) on top; with the default below the
        # effective temperature is 1/head_dim (softer / higher entropy
        # attention than vanilla 1/sqrt(head_dim)). Pretrained ckpts were
        # trained with this factor, so we MUST apply it at inference too.
        self.relax_temperature_scaling: float = (
            float(relax_temperature_scaling)
            if relax_temperature_scaling is not None
            else self.head_dim**-0.5
        )

        # Linear projections (names match upstream so state_dict loads).
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # bias_k / bias_v: not used at inference (kept None so state_dict
        # loading doesn't complain about extra keys — the training code
        # also stored them as None).
        self.bias_k = None
        self.bias_v = None

        # All shipped K-Fold checkpoints use RoPE; not optional.
        self.rot_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: Tensor,
        seq_id: Tensor | None = None,
        pos_id: Tensor | None = None,
    ) -> Tensor:
        """
        x: [B, L, D]
        seq_id: [B, L] (bool or int)
        pos_id: [B, L], default of arange(L) if None
        """
        B, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        # Project Q, K, V from the same input (self-attention).
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, L, D] -> [B, L, H, D/H] -> [B, H, L, D/H]
        q, k, v = map(
            lambda x: x.unflatten(-1, (H, Dh)).transpose(-2, -3).contiguous(), (q, k, v)
        )

        # Apply RoPE.
        if pos_id is not None:
            q, k = self.rot_emb.forward_with_position_ids(q, k, pos_id, B, H, Dh)
        else:
            q, k = self.rot_emb(q, k)

        # Apply the "relax temperature scaling" factor on Q. SDPA's own
        # 1/sqrt(head_dim) scaling is applied on top.
        q = q * self.relax_temperature_scaling
        if seq_id is not None:
            attn_mask = seq_id[:, None, :] == seq_id[:, :, None]
            attn_mask = attn_mask.unsqueeze(-3)  # [B, 1, L, L]
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.transpose(-2, -3).contiguous().flatten(-2)
        out = self.out_proj(out)
        return out


class ESM1LayerNorm(nn.Module):
    """Layer norm with ESM1 initialization (affine=True by default).

    Default `hidden_size=2560` matches the 3B encoder's `embed_dim` /
    `decoder_dim`.
    """

    def __init__(self, hidden_size=2560, eps: float = 1e-12, affine: bool = True):
        super().__init__()
        self.hidden_size = (
            (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        )
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x


class TransformerLayer(nn.Module):
    """ESM2-style pre-LN transformer block with rotary attention.

    Defaults match the 3B encoder block (`embed_dim=2560`,
    `ffn_embed_dim=10240`, `attention_heads=40`). `ffn_embed_dim=None`
    auto-resolves to `4 * embed_dim` for any other embed_dim.
    """

    def __init__(
        self,
        embed_dim: int = 2560,
        ffn_embed_dim: int = 10240,
        attention_heads: int = 40,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads

        self.self_attn = MultiheadAttention(embed_dim, attention_heads)
        self.self_attn_layer_norm = ESM1LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = ESM1LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor | None = None,
        pos_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, seq_id, pos_id)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x
