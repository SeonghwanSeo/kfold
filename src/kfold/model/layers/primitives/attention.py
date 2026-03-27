import math

import torch
import torch.nn.functional as F

try:
    from cuequivariance_torch import attention_pair_bias as cueq_attention_pair_bias
except ImportError:
    cueq_attention_pair_bias = None

from .utils import add, mul, permute_final_dims


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale: float | None = None,
    inplace: bool = False,
) -> torch.Tensor:
    """Compute the attention operation.
    Parameters
    ----------
    query : torch.Tensor
        The query tensor of shape (..., Q, C)
    key : torch.Tensor
        The key tensor of shape (..., K, C)
    value : torch.Tensor
        The value tensor of shape (..., K, C)
    bias : Optional[torch.Tensor]
        The attention bias of shape (..., Q, K), default None
    scale : Optional[float]
        The scaling factor for the query-key dot product, default None
    inplace : bool
        Whether to perform operations in-place, default False

    Returns
    -------
    out : torch.Tensor
        The output tensor of shape (..., Q, C)
    """

    with torch.autocast("cuda", dtype=torch.float32):
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        bias = bias.to(torch.float32) if bias is not None else None

        if scale is not None:
            query = mul(query, scale, inplace=inplace)

        # Compute attention weights
        attn = torch.einsum("...qc,...kc->...qk", query, key)

        # Add attention bias
        if bias is not None:
            attn = add(attn, bias, inplace=inplace)

        # Softmax
        attn = attn.softmax(dim=-1)

    # Compute output
    out = torch.einsum("...qk,...kc->...qc", attn.to(value.dtype), value)

    return out


@torch.compiler.disable
def kernel_attention_pair_bias(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    w_proj_z: torch.Tensor,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    w_ln_z: torch.Tensor,
    b_ln_z: torch.Tensor | None,
    b_proj_z: torch.Tensor | None,
    b_proj_g: torch.Tensor | None,
    b_proj_o: torch.Tensor | None,
    num_heads: int = 32,
    inf: float = 1e6,
    eps: float = 1e-5,
    attn_scale: float | None = None,
) -> torch.Tensor:
    """A wrapper for attention_pair_bias with kernel support disabled."""
    if cueq_attention_pair_bias is None:
        raise ImportError(
            "cuequivariance_torch is not installed. Please install it to use the kernel."
        )
    out, _ = cueq_attention_pair_bias(
        s,
        q,
        k,
        v,
        z,
        mask,
        num_heads,
        w_proj_z,
        w_proj_g,
        w_proj_o,
        w_ln_z,
        b_ln_z,
        b_proj_z,
        b_proj_g,
        b_proj_o,
        inf,
        eps,
        attn_scale,
        return_z_proj=False,
    )
    return out


def attention_pair_bias(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    w_proj_z: torch.Tensor,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    w_ln_z: torch.Tensor,
    b_ln_z: torch.Tensor | None,
    b_proj_z: torch.Tensor | None,
    b_proj_g: torch.Tensor | None,
    b_proj_o: torch.Tensor | None,
    num_heads: int = 32,
    inf: float = 1e6,
    eps: float = 1e-5,
    attn_scale: float | None = None,
    use_kernels: bool = False,
) -> torch.Tensor:
    """Compute the attention operation with pair bias using a custom kernel if available.
    Parameters
    ----------
    s : torch.Tensor
        The input tensor of shape (*, L, C)
        L is the sequence length, and C is the feature dimension.
    q : torch.Tensor
        The query tensor of shape (*, H, Lq, C_h)
    k : torch.Tensor
        The key tensor of shape (*, H, Lk, C_h)
    v : torch.Tensor
        The value tensor of shape (*, H, Lk, C_h)
    z : torch.Tensor
        The pair bias tensor of shape (*, Lq, Lk, C_z)
    mask : torch.Tensor
        The attention mask tensor of shape (*, Lk)
    w_proj_z : torch.Tensor
        The weight tensor for projecting z
    w_proj_g : torch.Tensor
        The weight tensor for projecting g
    w_proj_o : torch.Tensor
        The weight tensor for projecting o
    w_ln_z : torch.Tensor
        The weight tensor for layer norm on z
    b_ln_z : Optional[torch.Tensor]
        The bias tensor for layer norm on z
    b_proj_z : torch.Tensor
        The bias tensor for projecting z
    b_proj_g : Optional[torch.Tensor]
        The bias tensor for projecting g
    b_proj_o : Optional[torch.Tensor]
        The bias tensor for projecting o
    num_heads : int
        The number of attention heads, default 32
    inf : float
        The value to use for masking, default 1e6
    eps : float
        The epsilon value for numerical stability, default 1e-5
    attn_scale : Optional[float]
        The scaling factor for the query-key dot product, default None
    use_kernels : bool
        Whether to use the custom kernel if available, default False
    """
    if use_kernels:
        L = s.shape[-2]
        Lq = q.shape[-2]
        Lk = k.shape[-2]
        if not (L == Lq == Lk):
            raise NotImplementedError(
                "cueq_attention_pair_bias only supports Lq == Lk == L."
            )
        # Merge batch dims for compatibility
        batch_dims = s.shape[:-2]
        s = s.flatten(0, -3)  # [*, L, C]
        q = q.flatten(0, -4)  # [*, H, Q, C_h]
        k = k.flatten(0, -4)  # [*, H, K, C_h]
        v = v.flatten(0, -4)  # [*, H, K, C_h]
        z = z.flatten(0, -4)  # [*, Q, K, C_z]
        mask = mask.flatten(0, -2)  # [*, K]

        out = kernel_attention_pair_bias(
            s,
            q,
            k,
            v,
            z,
            mask,
            w_proj_z,
            w_proj_g,
            w_proj_o,
            w_ln_z,
            b_ln_z,
            b_proj_z,
            b_proj_g,
            b_proj_o,
            num_heads,
            inf,
            eps,
            attn_scale,
        )  # [B*N, L, C]
        out = out.unflatten(0, batch_dims)  # restore batch dims

    else:
        # layernorm on z
        z_ln = F.layer_norm(z, z.shape[-1:], w_ln_z, b_ln_z)  # [*, Q, K, C_z]
        # project z to get attention bias
        attn_bias = F.linear(z_ln, w_proj_z, b_proj_z)  # [*, Q, K, H]
        attn_bias = permute_final_dims(attn_bias, (2, 0, 1))  # [*, H, Q, K]
        del z_ln
        attn_bias = attn_bias - inf * (
            1 - mask.to(attn_bias.dtype)[..., None, None, :]
        )  # [*, H, Q, K]

        # === Attention === #
        if attn_scale is None:
            attn_scale = 1 / math.sqrt(q.shape[-1])
        Av = _attention(
            q,  # [B, N, H, Q, C_h]
            k,  # [B, N, H, K, C_h]
            v,  # [B, N, H, K, C_h]
            bias=attn_bias,  # [B, N, H, Q, K]
            scale=attn_scale,
        )  # [B, N, H, Q, C_h]
        Av = permute_final_dims(Av, (1, 0, 2))  # [B, N, Q, H, C_h]
        Av = Av.reshape(s.shape)  # [B, N, Q, C]

        g = F.sigmoid(F.linear(s, w_proj_g, b_proj_g))  # [B, N, L, C]
        out = F.linear(g * Av, w_proj_o, b_proj_o)  # [B, N, L, C]

    return out
