"""Public API for fused starting- and ending-node triangle attention."""

import torch

from .._common.layouts import interleave_kv
from .kernels import fused_gate_forward, triangle_attn_forward


def precompute(W_ln, B_ln, WQ, WK, WV, W_proj_z, B_proj_z, H, D):
    """Pack projection weights into layouts consumed by forward."""
    WQ_c = WQ.contiguous()  # (C_in, H*D) matmul convention, no fold
    WKV_c = interleave_kv(WK, WV, H, D).contiguous()  # (C_in, 2*H*D), no fold
    # Triton's reshape/split lowering used by the interleaved K/V fast path can
    # issue an illegal access for the Apo-module shape D=16 on Hopper.  Keep the
    # original K and V layouts as well so the kernel can select a genuine
    # two-matmul Triton specialization for D=16.  D>=32 continues to use WKV_c.
    WK_c = WK.contiguous() if D == 16 else None
    WV_c = WV.contiguous() if D == 16 else None
    WZ_c = W_proj_z.t().contiguous()  # (C_in, H) -> (H, C_in) for the bias kernel

    # bias-proj bias as an (H,) fp32 tensor (K-Fold's bias-proj is LinearNoBias).
    if B_proj_z is None:
        BZ = torch.zeros(H, device=WQ.device, dtype=torch.float32)
    else:
        BZ = B_proj_z.float().contiguous()

    return {
        "WQ_c": WQ_c,
        "WKV_c": WKV_c,
        "WK_c": WK_c,
        "WV_c": WV_c,
        "WZ_c": WZ_c,
        "BZ": BZ,
        # x̃ = LN(x) is computed once in `forward`; the kernels and the gate
        # epilogue all consume it, so the LN affine is needed here.
        "W_ln": W_ln,
        "B_ln": B_ln,
    }


def forward(
    X,
    pre,
    *,
    mask=None,
    scale=1.0,
    eps=1e-5,
    W_proj_g,
    B_proj_g,
    W_proj_o,
    B_proj_o,
    pad_to=32,
):
    """Run triangle attention on batched or unbatched pair features.

    The output rank matches X; pad_to controls sequence padding.
    """
    assert B_proj_g is None and B_proj_o is None, (
        "fused gate path assumes bias-free linear_g / linear_o (K-Fold LinearNoBias)"
    )
    unbatched = X.ndim == 3
    if unbatched:
        X = X.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    elif X.ndim != 4:
        raise ValueError(f"X must have shape (N,N,C) or (B,N,N,C), got {X.shape}")
    B, N, N2, C_in = X.shape
    if N != N2:
        raise ValueError(f"triangle attention requires square pair axes, got {X.shape}")
    H = pre["WZ_c"].shape[0]
    D = pre["WQ_c"].shape[1] // H
    if mask is not None and mask.shape != (B, N, N):
        raise ValueError(f"mask must have shape {(B, N, N)}, got {mask.shape}")
    # x̃ = LN(x) computed once and shared by the kernels and the gate.
    X_ln = torch.nn.functional.layer_norm(
        X,
        (C_in,),
        pre["W_ln"].to(X.dtype),
        pre["B_ln"].to(X.dtype) if pre["B_ln"] is not None else None,
        eps,
    )

    Np = ((N + pad_to - 1) // pad_to) * pad_to if pad_to and pad_to > 1 else N
    if Np != N:
        # Pad both N dims with zeros; mask the padding keys (and rows, which are
        # discarded).  This runs triangle_attn on the fast (multiple-of-pad_to) shape.
        X_ln_k = torch.nn.functional.pad(X_ln, (0, 0, 0, Np - N, 0, Np - N))
        if mask is None:
            mask_k = torch.zeros(B, Np, Np, dtype=torch.bool, device=X.device)
            mask_k[..., :N] = True
        else:
            mask_k = torch.nn.functional.pad(
                mask.bool(), (0, Np - N, 0, Np - N), value=False
            )
        O_pad = torch.empty(B, Np, Np, H * D, device=X.device, dtype=X.dtype)
        triangle_attn_forward(
            X_ln_k,
            pre["WQ_c"],
            pre["WKV_c"],
            pre["WZ_c"],
            pre["BZ"],
            O_pad,
            scale=scale,
            mask=mask_k,
            WK_c=pre["WK_c"],
            WV_c=pre["WV_c"],
        )
        O_attn = O_pad[:, :N, :N]  # strided view; gate reads native batch/pair strides
    else:
        O_attn = torch.empty(B, N, N, H * D, device=X.device, dtype=X.dtype)
        triangle_attn_forward(
            X_ln,
            pre["WQ_c"],
            pre["WKV_c"],
            pre["WZ_c"],
            pre["BZ"],
            O_attn,
            scale=scale,
            mask=mask,
            WK_c=pre["WK_c"],
            WV_c=pre["WV_c"],
        )
    # Gate epilogue (fused): K-Fold gates on q_x = LN(x), so reuse normalized x directly.
    # O_attn may be a strided padded slice; the gate gathers at native strides.
    output = fused_gate_forward(O_attn.view(B, N, N, H, D), X_ln, W_proj_g, W_proj_o)
    return output.squeeze(0) if unbatched else output


def triangle_attention(
    X,
    W_ln,
    B_ln,
    WQ,
    WK,
    WV,
    W_proj_z,
    B_proj_z,
    W_proj_g,
    B_proj_g,
    W_proj_o,
    B_proj_o,
    H,
    D,
    scale=1.0,
    eps=1e-5,
    mask=None,
):
    """Run triangle attention with projection precomputation included."""
    pre = precompute(W_ln, B_ln, WQ, WK, WV, W_proj_z, B_proj_z, H, D)
    return forward(
        X,
        pre,
        mask=mask,
        scale=scale,
        eps=eps,
        W_proj_g=W_proj_g,
        B_proj_g=B_proj_g,
        W_proj_o=W_proj_o,
        B_proj_o=B_proj_o,
    )
