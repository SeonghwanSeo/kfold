"""Code adapted from ESM2 (https://github.com/facebookresearch/esm)."""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


## Rotary Embedding
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    """The rotary position embeddings from RoFormer_ (Su et. al)."""

    def __init__(self, dim: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)


# MHA attention
class MultiheadAttention(nn.Module):
    """Multi-headed attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = True,
        use_rotary_embeddings: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scaling = self.d_head**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.d_head)

    def forward(
        self,
        x: Tensor,
        seq_id: Tensor | None,
        pos_id: Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: [B, L, D]
        seq_id: [B, L] (bool or int)
        pos_id: [B, L], default of arange(L) if None
        """
        B, L, D = x.shape
        H = self.num_heads
        Dh = self.d_head

        # Compute attention mask based on seq_id if provided
        if seq_id is not None:
            # NOTE: Following ESM3, we handle two cases for seq_id:
            # - bool: directly use as mask (True for attend, False for no attend)
            # - int: compute mask based on equality of seq_id
            attn_mask = seq_id[:, None, :] == seq_id[:, :, None]
            attn_mask = attn_mask.unsqueeze(-3)  # [B, 1, L, L]
            if attn_mask.all():
                # if all tokens attend to all tokens, we can skip the mask for efficiency
                attn_mask = None
        else:
            attn_mask = None

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, L, D] -> [B, L, H, D/H] -> [B, H, L, D/H]
        q, k, v = map(
            lambda x: x.unflatten(-1, (H, Dh)).transpose(-2, -3).contiguous(), (q, k, v)
        )

        # Apply rotary embeddings if enabled
        if self.rot_emb is not None:
            if pos_id is None:
                pos_id = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            q, k = self.apply_feature_rotary(q, k, pos_id)

        # shape
        # q: [B, H, L, D_h]
        # k: [B, H, L, D_h]
        # v: [B, H, L, D_h]
        # attn_mask: [B, 1, L, L] or None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.transpose(-2, -3).contiguous().flatten(-2)
        out = self.out_proj(out)
        return out

    def apply_feature_rotary(self, q: Tensor, k: Tensor, pos_id: Tensor):
        B, H, L, Dh = q.shape
        device, dtype = q.device, q.dtype

        inv_freq = self.rot_emb.inv_freq.to(device=device, dtype=dtype)
        # Split inv_freq into 4 parts for (t, x, y, z) dimensions
        inv_t = inv_freq[0::4]
        inv_x = inv_freq[1::4]
        inv_y = inv_freq[2::4]
        inv_z = inv_freq[3::4]
        inv_reorder = torch.cat((inv_t, inv_x, inv_y, inv_z), dim=0)  # [inv_dim]

        # Apply each dimension of position IDs to corresponding inv_freq slice
        pos_id = pos_id.to(dtype)[..., None]  # [B, L, 1]
        freq = pos_id * inv_reorder  # [B, L, inv_dim]
        emb = torch.cat((freq, freq), dim=-1)  # [B, L, 2*inv_dim]
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


class ESM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
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
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerLayer(nn.Module):
    """Transformer encoder block with rotary attention support."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_ratio: float,
        use_rotary_embeddings: bool = True,
    ):
        super().__init__()
        self.self_attn_layer_norm = ESM1LayerNorm(d_model)
        self.self_attn = MultiheadAttention(
            d_model=d_model,
            num_heads=n_heads,
            use_rotary_embeddings=use_rotary_embeddings,
        )

        d_expanded = int(d_model * expansion_ratio)
        self.final_layer_norm = ESM1LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_expanded)
        self.fc2 = nn.Linear(d_expanded, d_model)

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor | None = None,
        pos_id: torch.Tensor | None = None,
    ):
        r1 = self.self_attn_layer_norm(x)
        r1 = self.self_attn(r1, seq_id, pos_id)
        x += r1

        r2 = self.final_layer_norm(x)
        r2 = self.fc2(gelu(self.fc1(r2)))
        x += r2
        return x
