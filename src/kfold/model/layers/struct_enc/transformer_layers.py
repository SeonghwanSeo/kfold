"""Fused transformer layers for the frozen protein structure encoder."""

import math

import torch
import torch.nn.functional as F

from .nn import Linear


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Apply the training-time GELU formula, preserving intermediate rounding."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    """Cache 20,000 rotary positions shared across inputs and encoder layers."""

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.init_cache()

    def init_cache(self) -> None:
        """Cache BF16 factors with the previous inference-time angle rounding."""
        positions = torch.arange(20_000, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.bfloat16())
        self.register_buffer("_cos_cached", freqs.cos().tile(1, 2), persistent=False)
        self.register_buffer("_sin_cached", freqs.sin().tile(1, 2), persistent=False)

    def forward(self, pos_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up factors for position IDs in [0, 20000), returning (B, 1, L, D)."""
        cos = self._cos_cached[pos_id]
        sin = self._sin_cached[pos_id]
        return cos.unsqueeze(1), sin.unsqueeze(1)


class MultiheadAttention(torch.nn.Module):
    """Self-attention with a combined QKV projection and shared rotary factors."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = Linear(embed_dim, 3 * embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply masked attention to features of shape (B, L, D)."""
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = (
            t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
            for t in (q, k, v)
        )
        cos, sin = rotary
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # Preserve the pretrained encoder's extra 1/sqrt(head_dim) scaling.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=1.0 / self.head_dim
        )
        out = out.transpose(1, 2).reshape(x.shape)
        return self.out_proj(out)


class TransformerLayer(torch.nn.Module):
    """Pre-LN transformer with native LayerNorm and training-time GELU."""

    def __init__(self, embed_dim: int, ffn_embed_dim: int, attention_heads: int):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, attention_heads)
        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim, eps=1e-12)
        self.fc1 = Linear(embed_dim, ffn_embed_dim)
        self.fc2 = Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim, eps=1e-12)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Update features using the input's shared attention mask and rotary factors."""
        x = x + self.self_attn(self.self_attn_layer_norm(x), attn_mask, rotary)
        residual = x
        x = self.final_layer_norm(x)
        x = self.fc2(gelu(self.fc1(x)))
        return residual + x
