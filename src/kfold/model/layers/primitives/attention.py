import math

import torch
import torch.nn.functional as F

try:
    from cuequivariance_torch import attention_pair_bias as cueq_attention_pair_bias
except ImportError:
    cueq_attention_pair_bias = None

from .utils import permute_final_dims


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute the attention operation.
    Parameters
    ----------
    query : torch.Tensor
        The query tensor of shape (..., H, Lq, Dh)
    key : torch.Tensor
        The key tensor of shape (..., H, Lk, Dh)
    value : torch.Tensor
        The value tensor of shape (..., H, Lk, Dh)
    bias : Optional[torch.Tensor]
        The attention bias of shape (..., H, Lq, Lk), default None
    scale : Optional[float]
        The scaling factor for the query-key dot product, default None

    Returns
    -------
    out : torch.Tensor
        The output tensor of shape (..., H, Lq, Dh)
    """
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    bias = bias.to(torch.float32) if bias is not None else None
    if scale is None:
        scale = 1 / math.sqrt(query.shape[-1])

    with torch.autocast(query.device.type, enabled=False):
        # Compute attention weights
        attn = torch.einsum("...qc,...kc->...qk", query * scale, key)  # [*, H, Lq, Lk]

        # Add attention bias
        if bias is not None:
            attn = attn + bias

        # Softmax
        attn = attn.softmax(dim=-1).to(value.dtype)

    # Compute output
    out = torch.einsum("...qk,...kc->...qc", attn, value)

    return out


@torch.compiler.disable
def _kernel_attention_pair_bias(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
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
    # Check batch dimensions match
    if not (s.shape[:-3] == q.shape[:-4] == k.shape[:-4] == v.shape[:-4]):
        raise ValueError(
            f"cueq_attention_pair_bias supports inputs with the same batch dims: "
            f"s={s.shape}, q={q.shape}, k={k.shape}, v={v.shape}"
        )
    # Check sequence length dimensions match
    if not (s.shape[-2] == q.shape[-2] == k.shape[-2] == v.shape[-2]):
        raise ValueError(
            f"cueq_attention_pair_bias supports inputs with the same sequence "
            f"length dims: s={s.shape}, q={q.shape}, k={k.shape}, v={v.shape}"
        )
    # Merge batch dims for compatibility
    batch_dims = s.shape[:-2]
    s = s.flatten(0, -3)  # [*, L, D]
    q = q.flatten(0, -4)  # [*, H, Lq, Dh]
    k = k.flatten(0, -4)  # [*, H, Lk, Dh]
    v = v.flatten(0, -4)  # [*, H, Lk, Dh]
    pair_bias = pair_bias.flatten(0, -4)  # [*, H, Lq, Lk]
    mask = mask.flatten(0, -2)  # [*, Lk]

    out, _ = cueq_attention_pair_bias(
        s=s,  # [B*M, L, D]
        q=q,  # [B*M, H, Lq, Dh]
        k=k,  # [B*M, H, Lk, Dh]
        v=v,  # [B*M, H, Lv, Dh]
        z=pair_bias,  # [B, H, Lq, Lk]
        mask=mask,  # [B, Lk] or [B*M, Lk]
        num_heads=num_heads,  # int
        w_proj_z=None,  # is_cached_z_proj=True
        w_proj_g=w_proj_g,  # [D, D]
        w_proj_o=w_proj_o,  # [D, D]
        w_ln_z=None,  # is_cached_z_proj=True
        b_ln_z=None,  # is_cached_z_proj=True
        b_proj_z=None,  # is_cached_z_proj=True
        b_proj_g=b_proj_g,  # [D]
        b_proj_o=b_proj_o,  # [D]
        inf=inf,  # float
        eps=eps,  # float
        attn_scale=attn_scale,  # float or None
        return_z_proj=False,
        is_cached_z_proj=True,
    )
    out = out.unflatten(0, batch_dims)  # restore batch dims
    return out


def _torch_attention_pair_bias(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    b_proj_g: torch.Tensor | None,
    b_proj_o: torch.Tensor | None,
    num_heads: int = 32,
    inf: float = 1e6,
    attn_scale: float | None = None,
) -> torch.Tensor:
    """A wrapper for attention_pair_bias with kernel support disabled."""
    # Mask out invalid positions in pair bias
    pair_bias = (
        pair_bias - inf * ((~mask.bool()).float()[..., None, None, :])
    )  # [*, H, Lq, Lk]

    # === Attention === #
    Av = _attention(
        q,  # [*, H, Lq, Dh]
        k,  # [*, H, Lk, Dh]
        v,  # [*, H, Lk, Dh]
        bias=pair_bias,  # [*, H, Lq, Lk]
        scale=attn_scale,
    )  # [*, H, Lq, Dh]
    Av = permute_final_dims(Av, (1, 0, 2)).reshape(s.shape)  # [*, Lq, D]

    g = torch.sigmoid(F.linear(s, w_proj_g, b_proj_g))  # [*, L, D]
    out = F.linear(g * Av, w_proj_o, b_proj_o)  # [*, L, D]
    return out


def attention_pair_bias(
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_bias: torch.Tensor,
    mask: torch.Tensor,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    b_proj_g: torch.Tensor | None,
    b_proj_o: torch.Tensor | None,
    num_heads: int = 32,
    inf: float = 1e9,
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
    pair_bias : torch.Tensor
        The pair bias tensor of shape (*, H, Lq, Lk)
    mask : torch.Tensor
        The attention mask tensor of shape (*, Lk)
    w_proj_g : torch.Tensor
        The weight tensor for projecting g
    w_proj_o : torch.Tensor
        The weight tensor for projecting o
    b_proj_g : Optional[torch.Tensor]
        The bias tensor for projecting g
    b_proj_o : Optional[torch.Tensor]
        The bias tensor for projecting o
    num_heads : int
        The number of attention heads, default 32
    inf : float
        The value to use for masking, default 1e6
    attn_scale : Optional[float]
        The scaling factor for the query-key dot product, default None
    use_kernels : bool
        Whether to use the custom kernel if available, default False
    """
    if use_kernels:
        return _kernel_attention_pair_bias(
            s,
            q,
            k,
            v,
            pair_bias,
            mask,
            w_proj_g,
            w_proj_o,
            b_proj_g,
            b_proj_o,
            num_heads,
            inf,
            attn_scale,
        )
    else:
        return _torch_attention_pair_bias(
            s,
            q,
            k,
            v,
            pair_bias,
            mask,
            w_proj_g,
            w_proj_o,
            b_proj_g,
            b_proj_o,
            num_heads,
            inf,
            attn_scale,
        )
