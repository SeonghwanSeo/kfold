"""Triton kernels for triangle multiplicative update projection phases.

The shared kernel applies LayerNorm with folded projection parameters, sigmoid
gating, optional masking, and layout-aware stores.
"""

import torch
import triton
import triton.language as tl

from .._common.dtypes import tl_io_dtype


def _early_config_prune(configs, named_args, **_kwargs):
    """Drop register-infeasible wide tiles for the K-Fold C=256 path."""
    arguments = {**(named_args or {}), **_kwargs}
    wide_channel = max(int(arguments["D"]), int(arguments["D_OUT"])) >= 256
    if not wide_channel:
        return configs
    return [
        config
        for config in configs
        if not (config.kwargs["BLOCK_M"] >= 128 and config.kwargs["BLOCK_K"] >= 128)
    ]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        # output phase (D_OUT=128) likes one-shot BLOCK_K=128
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    ],
    key=["M", "D", "D_OUT", "TWO_INPUTS", "X1_DMAJOR", "TRANS", "HAS_MASK", "USE_INT64"],
    prune_configs_by={"early_config_prune": _early_config_prune},
)
@triton.jit
def fused_layer_norm_sigmoid_gated_transpose(
    X1_ptr,
    X1_stride0,
    X1_stride1,  # value-branch input (stride0=token-m, stride1=feature-d)
    X2_ptr,
    X2_stride0,
    X2_stride1,  # gate-branch input  (unused if not TWO_INPUTS)
    Wp_ptr,
    Wp_stride0,
    Wp_stride1,  # (D_OUT, D) value proj (LN-folded)
    Wg_ptr,
    Wg_stride0,
    Wg_stride1,  # (D_OUT, D) gate proj  (LN-folded)
    sWp_ptr,
    sWg_ptr,
    BPc_ptr,
    BGc_ptr,
    Mask_ptr,
    Mask_stride0,
    Out_ptr,
    Out_stride0,
    Out_stride1,  # output strides: stride0=token-m, stride1=feature-k
    M: tl.constexpr,
    D: tl.constexpr,
    D_OUT: tl.constexpr,
    EPS: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    USE_INT64: tl.constexpr,
    TWO_INPUTS: tl.constexpr,
    X1_DMAJOR: tl.constexpr,
    TRANS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Flattened pair tensors cross the signed-int32 element-offset limit when
    # D_OUT=512 and L>2048. Promote pointer offsets only for those large shapes.
    d_offs = tl.arange(0, D)
    if USE_INT64:
        m_ptr_offs = m_offs.to(tl.int64)
        d_ptr_offs = d_offs.to(tl.int64)
    else:
        m_ptr_offs = m_offs
        d_ptr_offs = d_offs
    m_mask = m_offs < M
    inv_D = 1.0 / D

    # ---- load X1 (value branch) as (BLOCK_M, D), coalesced for either layout ----
    if X1_DMAJOR:  # X1 is (D, M): load (D, BM) coalesced, then trans
        X1t = tl.load(
            X1_ptr + d_ptr_offs[:, None] * X1_stride1 + m_ptr_offs[None, :] * X1_stride0,
            mask=m_mask[None, :],
            other=0.0,
        )
        X1 = tl.trans(X1t)
    else:  # X1 is (M, D): load (BM, D) directly
        X1 = tl.load(
            X1_ptr + m_ptr_offs[:, None] * X1_stride0 + d_offs[None, :] * X1_stride1,
            mask=m_mask[:, None],
            other=0.0,
        )

    X1_fp32 = X1.to(tl.float32)
    mean1 = tl.sum(X1_fp32, axis=-1) * inv_D
    diff1 = X1_fp32 - mean1[:, None]
    rstd1 = 1.0 / tl.sqrt(tl.sum(diff1 * diff1, axis=-1) * inv_D + EPS)

    # ---- load X2 (gate branch) — reuse X1 if single-input ----
    if TWO_INPUTS:
        X2 = tl.load(
            X2_ptr + m_ptr_offs[:, None] * X2_stride0 + d_offs[None, :] * X2_stride1,
            mask=m_mask[:, None],
            other=0.0,
        )
        X2_fp32 = X2.to(tl.float32)
        mean2 = tl.sum(X2_fp32, axis=-1) * inv_D
        diff2 = X2_fp32 - mean2[:, None]
        rstd2 = 1.0 / tl.sqrt(tl.sum(diff2 * diff2, axis=-1) * inv_D + EPS)
    else:
        X2 = X1
        mean2 = mean1
        rstd2 = rstd1

    if HAS_MASK:
        m_val = tl.load(Mask_ptr + m_ptr_offs * Mask_stride0, mask=m_mask, other=0.0).to(
            tl.float32
        )

    for k in tl.range(0, D_OUT, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < D_OUT
        k_ptr_offs = k_offs.to(tl.int64) if USE_INT64 else k_offs
        Wp = tl.load(
            Wp_ptr + k_offs[None, :] * Wp_stride0 + d_offs[:, None] * Wp_stride1,
            mask=k_mask[None, :],
            other=0.0,
        )
        Wg = tl.load(
            Wg_ptr + k_offs[None, :] * Wg_stride0 + d_offs[:, None] * Wg_stride1,
            mask=k_mask[None, :],
            other=0.0,
        )
        sWp = tl.load(sWp_ptr + k_offs, mask=k_mask, other=0.0)
        sWg = tl.load(sWg_ptr + k_offs, mask=k_mask, other=0.0)
        BPc = tl.load(BPc_ptr + k_offs, mask=k_mask, other=0.0)
        BGc = tl.load(BGc_ptr + k_offs, mask=k_mask, other=0.0)

        P = tl.dot(X1, Wp)  # value ← X1
        G = tl.dot(X2, Wg)  # gate  ← X2
        P = rstd1[:, None] * (P - mean1[:, None] * sWp[None, :]) + BPc[None, :]
        G = rstd2[:, None] * (G - mean2[:, None] * sWg[None, :]) + BGc[None, :]
        Out = tl.sigmoid(G) * P
        if HAS_MASK:
            Out = Out * m_val[:, None]

        if TRANS:  # write (D_OUT, M): Out[k, m], m contiguous
            tl.store(
                Out_ptr
                + k_ptr_offs[:, None] * Out_stride1
                + m_ptr_offs[None, :] * Out_stride0,
                tl.trans(Out).to(IO_DTYPE),
                mask=k_mask[:, None] & m_mask[None, :],
            )
        else:  # write (M, D_OUT): Out[m, k]
            tl.store(
                Out_ptr
                + m_ptr_offs[:, None] * Out_stride0
                + k_ptr_offs[None, :] * Out_stride1,
                Out.to(IO_DTYPE),
                mask=m_mask[:, None] & k_mask[None, :],
            )


# ---------------------------------------------------------------------------
# Phase launch wrappers (signatures unchanged — module.py depends on them)
# ---------------------------------------------------------------------------
def _launch(
    X1,
    X1_s0,
    X1_s1,
    X2,
    X2_s0,
    X2_s1,
    Wp,
    Wg,
    sWp,
    sWg,
    BPc,
    BGc,
    mask,
    out,
    out_s0,
    out_s1,
    M,
    D,
    D_OUT,
    eps,
    *,
    two_inputs,
    x1_dmajor,
    trans,
):
    if mask is not None:
        mask_ptr, mask_s0, has_mask = mask, mask.stride(0), True
    else:
        mask_ptr, mask_s0, has_mask = X1, 0, False
    if X2 is None:
        X2, X2_s0, X2_s1 = X1, X1_s0, X1_s1

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    use_int64 = max(M * D, M * D_OUT) > (1 << 31)
    fused_layer_norm_sigmoid_gated_transpose[grid](
        X1,
        X1_s0,
        X1_s1,
        X2,
        X2_s0,
        X2_s1,
        Wp,
        Wp.stride(0),
        Wp.stride(1),
        Wg,
        Wg.stride(0),
        Wg.stride(1),
        sWp,
        sWg,
        BPc,
        BGc,
        mask_ptr,
        mask_s0,
        out,
        out_s0,
        out_s1,
        M=M,
        D=D,
        D_OUT=D_OUT,
        EPS=eps,
        IO_DTYPE=tl_io_dtype(X1.dtype),
        HAS_MASK=has_mask,
        USE_INT64=use_int64,
        TWO_INPUTS=two_inputs,
        X1_DMAJOR=x1_dmajor,
        TRANS=trans,
    )


def input_phase(
    x,
    mask,
    p_in_combined,
    g_in_combined,
    sum_W_p_in,
    sum_W_g_in,
    B_p_in_const,
    B_g_in_const,
    eps,
):
    """Compute the masked input projection in D-major layout."""
    B, L, _, D = x.shape
    D_OUT = p_in_combined.shape[0]
    M = B * L * L
    x_flat = x.reshape(M, D)
    out_t = torch.empty(D_OUT, M, device=x.device, dtype=x.dtype)  # (D_OUT, M)
    mask_flat = mask.reshape(M).to(torch.int8).contiguous() if mask is not None else None
    # X1=x (M,D): s0=token=D, s1=feature=1.  Out (D_OUT,M): s0=token=1, s1=feature=M.
    _launch(
        x_flat,
        x_flat.stride(0),
        x_flat.stride(1),
        None,
        0,
        0,
        p_in_combined,
        g_in_combined,
        sum_W_p_in,
        sum_W_g_in,
        B_p_in_const,
        B_g_in_const,
        mask_flat,
        out_t,
        1,
        M,
        M,
        D,
        D_OUT,
        eps,
        two_inputs=False,
        x1_dmajor=False,
        trans=True,
    )
    return out_t.view(D_OUT, B, L, L)


def output_phase(
    y_t,
    x,
    p_out_combined,
    g_out_combined,
    sum_W_p_out,
    sum_W_g_out,
    B_p_out_const,
    B_g_out_const,
    eps,
):
    """Gate the triangle product and return row-major output."""
    B, L, _, D = x.shape
    D_OUT = p_out_combined.shape[0]
    M = B * L * L
    y_flat = y_t.reshape(D, M)  # (D, M)
    x_flat = x.reshape(M, D)
    out = torch.empty(M, D_OUT, device=x.device, dtype=x.dtype)  # (M, D_OUT)
    # X1=y (D,M): s0=token=1, s1=feature=M. X2=x (M,D): s0=D, s1=1.
    # Out (M,D_OUT): s0=D_OUT, s1=1.
    _launch(
        y_flat,
        1,
        M,
        x_flat,
        x_flat.stride(0),
        x_flat.stride(1),
        p_out_combined,
        g_out_combined,
        sum_W_p_out,
        sum_W_g_out,
        B_p_out_const,
        B_g_out_const,
        None,
        out,
        out.stride(0),
        out.stride(1),
        M,
        D,
        D_OUT,
        eps,
        two_inputs=True,
        x1_dmajor=True,
        trans=False,
    )
    return out.view(B, L, L, D_OUT)
