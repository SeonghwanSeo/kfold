import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .rotary import RotaryEmbedding


class VanillaRelativePositionEmbedding(nn.Module):
    def __init__(self, bins, embedding_dim, init_std=0.02):
        super().__init__()
        self.bins = bins
        self.embedding = nn.Embedding(2 * bins + 2, embedding_dim)
        self.embedding.weight.data.normal_(0, init_std)

    def forward(self, query_residue_index, key_residue_index):
        diff = key_residue_index - query_residue_index.unsqueeze(1)
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1
        return self.embedding(diff)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )


class VanillaMultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()
        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, attention_mask):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = self.q_ln(query_BLD), self.k_ln(key_BLD)
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)
        n_heads = self.n_heads
        reshaper = functools.partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        query_BHLD, key_BHLD, value_BHLD = map(reshaper, (query_BLD, key_BLD, value_BLD))
        attn_mask_BLL = torch.logical_and(
            attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)
        )
        attn_mask_BHLL = attn_mask_BLL.unsqueeze(1)
        context_BHLD = F.scaled_dot_product_attention(
            query_BHLD, key_BHLD, value_BHLD, attn_mask_BHLL
        )
        context_BLD = rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD)


class VanillaGeometricReasoningOriginalImpl(nn.Module):
    def __init__(
        self,
        c_s: int,
        v_heads: int,
        num_vector_messages: int = 1,
        mask_and_zero_frameless: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.c_s = c_s
        self.v_heads = v_heads
        self.num_vector_messages = num_vector_messages
        self.mask_and_zero_frameless = mask_and_zero_frameless
        self.s_norm = nn.LayerNorm(c_s, bias=bias)
        dim_proj = 4 * self.v_heads * 3 + self.v_heads * 3 * self.num_vector_messages
        self.proj = nn.Linear(c_s, dim_proj, bias=bias)
        channels_out = self.v_heads * 3 * self.num_vector_messages
        self.out_proj = nn.Linear(channels_out, c_s, bias=bias)
        self.distance_scale_per_head = nn.Parameter(torch.zeros(self.v_heads))
        self.rotation_scale_per_head = nn.Parameter(torch.zeros(self.v_heads))

    def forward(self, s, attention_mask, affine, affine_mask):
        B, L = attention_mask.shape
        attn_bias = torch.zeros((B, 1, L, L), device=s.device)
        attn_bias = attn_bias.masked_fill_(
            ~torch.logical_and(affine_mask, attention_mask)[:, None, None, :],
            torch.finfo(attn_bias.dtype).min,
        )
        ns = self.s_norm(s)
        vec_rot, vec_dist = self.proj(ns).split(
            [
                self.v_heads * 2 * 3 + self.v_heads * 3 * self.num_vector_messages,
                self.v_heads * 2 * 3,
            ],
            dim=-1,
        )
        query_rot, key_rot, value = (
            affine.rot[..., None]
            .apply(rearrange(vec_rot, "... (h c) -> ... h c", c=3))
            .split(
                [self.v_heads, self.v_heads, self.v_heads * self.num_vector_messages],
                dim=-2,
            )
        )
        query_dist, key_dist = (
            affine[..., None]
            .apply(rearrange(vec_dist, "... (h c) -> ... h c", c=3))
            .chunk(2, dim=-2)
        )
        query_dist = rearrange(query_dist, "b s h d -> b h s 1 d")
        key_dist = rearrange(key_dist, "b s h d -> b h 1 s d")
        query_rot = rearrange(query_rot, "b s h d -> b h s d")
        key_rot = rearrange(key_rot, "b s h d -> b h d s")
        value = rearrange(value, "b s (h m) d -> b h s (m d)", m=self.num_vector_messages)
        distance_term = (query_dist - key_dist).norm(dim=-1) / math.sqrt(3)
        rotation_term = query_rot.matmul(key_rot) / math.sqrt(3)
        distance_term_weight = rearrange(
            F.softplus(self.distance_scale_per_head), "h -> h 1 1"
        )
        rotation_term_weight = rearrange(
            F.softplus(self.rotation_scale_per_head), "h -> h 1 1"
        )
        attn_weight = (
            rotation_term * rotation_term_weight - distance_term * distance_term_weight
        )
        if attn_bias is not None:
            s_q = attn_weight.size(2)
            s_k = attn_weight.size(3)
            _s_q = max(0, attn_bias.size(2) - s_q)
            _s_k = max(0, attn_bias.size(3) - s_k)
            attn_bias = attn_bias[:, :, _s_q:, _s_k:]
            attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_out = attn_weight.matmul(value)
        attn_out = (
            affine.rot[..., None]
            .invert()
            .apply(
                rearrange(
                    attn_out, "b h s (m d) -> b s (h m) d", m=self.num_vector_messages
                )
            )
        )
        attn_out = rearrange(
            attn_out, "b s (h m) d -> b s (h m d)", m=self.num_vector_messages
        )
        if self.mask_and_zero_frameless:
            attn_out.masked_fill_(~affine_mask[..., None], 0.0)
        s = self.out_proj(attn_out)
        return s


class VanillaUnifiedTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_geom_attn: bool = False,
        use_plain_attn: bool = True,
        v_heads: int | None = None,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        mask_and_zero_frameless: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.use_plain_attn = use_plain_attn
        if self.use_plain_attn:
            self.attn = VanillaMultiHeadAttention(
                d_model, n_heads, bias, qk_layernorm=qk_layernorm
            )
        self.use_geom_attn = use_geom_attn
        if self.use_geom_attn:
            if v_heads is None:
                raise ValueError("v_heads must be specified when use_geom_attn is True")
            self.geom_attn = VanillaGeometricReasoningOriginalImpl(
                c_s=d_model,
                v_heads=v_heads,
                bias=bias,
                mask_and_zero_frameless=mask_and_zero_frameless,
            )
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.scaling_factor = residue_scaling_factor

    def forward(self, x, attention_mask, frames, frames_mask):
        inv_scaling = 1 / self.scaling_factor
        if self.use_plain_attn:
            r1 = self.attn(x, attention_mask)
            x.add_(r1, alpha=inv_scaling)
        if self.use_geom_attn:
            r2 = self.geom_attn(x, attention_mask, frames, frames_mask)
            x.add_(r2, alpha=inv_scaling)
        r3 = self.ffn(x)
        x.add_(r3, alpha=inv_scaling)
        return x


class VanillaTransformerStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int | None,
        n_layers: int,
        n_layers_geom: int = 1,
        scale_residue: bool = True,
        mask_and_zero_frameless: bool = False,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",
        expansion_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                VanillaUnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    v_heads=v_heads,
                    use_geom_attn=i < n_layers_geom,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    expansion_ratio=expansion_ratio,
                    mask_and_zero_frameless=mask_and_zero_frameless,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x,
        attention_mask=None,
        affine=None,
        affine_mask=None,
    ):
        for block in self.blocks:
            x = block(x, attention_mask, affine, affine_mask)
        return self.norm(x), x


class VanillaGeometricEncoderStack(VanillaTransformerStack):
    def __init__(self, d_model, n_heads, v_heads, n_layers):
        super().__init__(d_model, n_heads, v_heads, 0)
        self.blocks = nn.ModuleList(
            [
                VanillaUnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    v_heads=v_heads,
                    use_geom_attn=True,
                    use_plain_attn=False,
                    expansion_ratio=4,
                    bias=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.Identity()
