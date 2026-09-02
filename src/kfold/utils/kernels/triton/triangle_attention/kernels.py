"""Triton kernels for triangle attention.

The implementation materializes shared pair bias once, fuses QKV projection
with attention, and uses a separate gated epilogue.
"""

import torch
import triton
import triton.language as tl

from .._common.dtypes import tl_io_dtype


# ===========================================================================
# Kernel 1: bias projection  (LN(x) + W_proj_z → Bp (H, N, N), materialized once)
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_QK": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_QK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_QK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_QK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_QK": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_QK": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_QK": 256}, num_warps=4, num_stages=2),
    ],
    key=["N", "C_in", "H"],
)
@triton.jit
def bias_proj_kernel(
    X_ptr,
    X_strideB,
    X_stride0,
    X_stride1,
    X_stride2,
    WZ_ptr,
    WZ_stride0,
    WZ_stride1,
    BZ_ptr,
    BZ_stride0,
    Bias_ptr,
    Bias_strideB,
    Bias_stride0,
    Bias_stride1,
    Bias_stride2,
    B: tl.constexpr,
    N: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    USE_INT64: tl.constexpr,
    BLOCK_QK: tl.constexpr,
):
    # X is the pre-normalized x̃ = LN(x); bias[h,q,k] = Σ_c x̃[q,k,c]·WZ[h,c] + BZ[h].
    # Grid is (batched_qk_blocks, H) so qk_blocks (large for big N) uses x dim,
    # avoiding the 65535 CUDA y-dim limit at N ≥ 1536 with small BLOCK_QK.
    pid_qk = tl.program_id(0)
    pid_h = tl.program_id(1)

    global_qk_offs = pid_qk * BLOCK_QK + tl.arange(0, BLOCK_QK)
    c_offs = tl.arange(0, C_in)
    qk_mask = global_qk_offs < (B * N * N)

    b_idx = global_qk_offs // (N * N)
    qk_offs = global_qk_offs % (N * N)
    q_idx = qk_offs // N
    k_idx = qk_offs % N
    b_ptr_idx = b_idx.to(tl.int64) if USE_INT64 else b_idx
    q_ptr_idx = q_idx.to(tl.int64) if USE_INT64 else q_idx
    k_ptr_idx = k_idx.to(tl.int64) if USE_INT64 else k_idx

    X_tile = tl.load(
        X_ptr
        + b_ptr_idx[:, None] * X_strideB
        + q_ptr_idx[:, None] * X_stride0
        + k_ptr_idx[:, None] * X_stride1
        + c_offs[None, :] * X_stride2,
        mask=qk_mask[:, None],
        other=0.0,
    )
    X_fp32 = X_tile.to(tl.float32)

    w_z = tl.load(WZ_ptr + pid_h * WZ_stride0 + c_offs * WZ_stride1)
    BZ_h = tl.load(BZ_ptr + pid_h * BZ_stride0)

    bias = tl.sum(X_fp32 * w_z[None, :], axis=-1) + BZ_h

    tl.store(
        Bias_ptr
        + b_ptr_idx * Bias_strideB
        + pid_h * Bias_stride0
        + q_ptr_idx * Bias_stride1
        + k_ptr_idx * Bias_stride2,
        bias.to(IO_DTYPE),
        mask=qk_mask,
    )


# ===========================================================================
# Kernel 2: triangle attention with K/V concat (+ tl.split)
#
# D=16 uses a separate-WK/WV specialization.  Triton's reshape/split lowering
# for the interleaved (BLOCK_K, D, 2) accumulator is not memory-safe for that
# shape on Hopper (it can fail during autotuning with an illegal memory access).
# Keeping this as a constexpr branch preserves the existing D>=32 generated
# kernel while giving the K-Fold Apo module a fully Triton implementation.
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["N", "C_in", "H", "D"],
)
@triton.jit
def triangle_attn_kernel(
    X_ptr,
    X_strideB,
    X_stride0,
    X_stride1,
    X_stride2,
    WQ_ptr,
    WQ_stride0,
    WQ_stride1,  # (C_in, H*D)
    WKV_ptr,
    WKV_stride0,
    WKV_stride1,  # (C_in, 2*H*D) feature-interleaved per head
    WK_ptr,
    WK_stride0,
    WK_stride1,  # (C_in, H*D), used iff SEPARATE_KV
    WV_ptr,
    WV_stride0,
    WV_stride1,  # (C_in, H*D), used iff SEPARATE_KV
    Bias_ptr,
    Bias_strideB,
    Bias_stride0,
    Bias_stride1,
    Bias_stride2,  # (H, N, N)
    Mask_ptr,
    Mask_strideB,
    Mask_stride0,
    Mask_stride1,
    O_ptr,
    O_strideB,
    O_stride0,
    O_stride1,
    O_stride2,  # (N, N, H*D)
    N: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    NEG_INF: tl.constexpr,
    HAS_MASK: tl.constexpr,
    SEPARATE_KV: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    USE_INT64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # X is the pre-normalized x̃ = LN(x); Q/K/V are plain projections of x̃.
    pid_bn = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_b = pid_bn // N
    pid_n = pid_bn % N
    pid_b_ptr = pid_b.to(tl.int64) if USE_INT64 else pid_b

    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, D)
    kv_cols = pid_h * 2 * D + tl.arange(0, 2 * D)
    c_offs = tl.arange(0, C_in)
    head_col = pid_h * D + d_offs
    q_mask = q_offs < N

    X_q = tl.load(
        X_ptr
        + pid_b_ptr * X_strideB
        + pid_n * X_stride0
        + q_offs[:, None] * X_stride1
        + c_offs[None, :] * X_stride2,
        mask=q_mask[:, None],
        other=0.0,
    )

    WQ_h = tl.load(WQ_ptr + c_offs[:, None] * WQ_stride0 + head_col[None, :] * WQ_stride1)
    if SEPARATE_KV:
        WK_h = tl.load(
            WK_ptr + c_offs[:, None] * WK_stride0 + head_col[None, :] * WK_stride1
        )
        WV_h = tl.load(
            WV_ptr + c_offs[:, None] * WV_stride0 + head_col[None, :] * WV_stride1
        )
    else:
        WKV_h = tl.load(
            WKV_ptr + c_offs[:, None] * WKV_stride0 + kv_cols[None, :] * WKV_stride1
        )

    Q_acc = tl.dot(X_q, WQ_h)
    Q_scaled = (Q_acc * SCALE).to(IO_DTYPE)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < N

        X_k = tl.load(
            X_ptr
            + pid_b_ptr * X_strideB
            + pid_n * X_stride0
            + k_offs[:, None] * X_stride1
            + c_offs[None, :] * X_stride2,
            mask=k_mask[:, None],
            other=0.0,
        )

        if SEPARATE_KV:
            K_acc = tl.dot(X_k, WK_h)
            V_acc = tl.dot(X_k, WV_h)
        else:
            KV_acc = tl.dot(X_k, WKV_h)  # (BLOCK_K, 2*D)
            KV_3d = tl.reshape(KV_acc, (BLOCK_K, D, 2))
            K_acc, V_acc = tl.split(KV_3d)  # each (BLOCK_K, D)
        K_block = K_acc.to(IO_DTYPE)
        V_block = V_acc.to(IO_DTYPE)

        S = tl.dot(Q_scaled, tl.trans(K_block)).to(tl.float32)

        bias_tile = tl.load(
            Bias_ptr
            + pid_b_ptr * Bias_strideB
            + pid_h * Bias_stride0
            + q_offs[:, None] * Bias_stride1
            + k_offs[None, :] * Bias_stride2,
            mask=(q_mask[:, None] & k_mask[None, :]),
            other=0.0,
        ).to(tl.float32)
        S = S + bias_tile

        if HAS_MASK:
            mask_row = tl.load(
                Mask_ptr
                + pid_b_ptr * Mask_strideB
                + pid_n * Mask_stride0
                + k_offs * Mask_stride1,
                mask=k_mask,
                other=0,
            )
            mask_bias = tl.where(mask_row != 0, 0.0, NEG_INF)
            S = S + mask_bias[None, :]
        S = tl.where(k_mask[None, :], S, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(S, axis=1))
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])
        l_i = l_i * alpha + tl.sum(P, axis=1)
        O_acc = O_acc * alpha[:, None] + tl.dot(P.to(IO_DTYPE), V_block).to(tl.float32)
        m_i = m_new

    Out = O_acc / l_i[:, None]
    o_col = pid_h * D + d_offs
    tl.store(
        O_ptr
        + pid_b_ptr * O_strideB
        + pid_n * O_stride0
        + q_offs[:, None] * O_stride1
        + o_col[None, :] * O_stride2,
        Out.to(IO_DTYPE),
        mask=q_mask[:, None],
    )


def triangle_attn_forward(
    X_ln,
    WQ_c,
    WKV_c,
    WZ_c,
    BZ,
    out,
    scale=1.0,
    mask=None,
    Bias_buf=None,
    *,
    WK_c=None,
    WV_c=None,
):
    """Run bias projection and attention for batched or unbatched pair tensors."""
    original_out = out
    if X_ln.ndim == 3:
        if out.ndim != 3:
            raise ValueError(f"unbatched X_ln requires rank-3 out, got {out.shape}")
        X_ln = X_ln.unsqueeze(0)
        out = out.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
        if Bias_buf is not None:
            Bias_buf = Bias_buf.unsqueeze(0)
    elif X_ln.ndim != 4 or out.ndim != 4:
        raise ValueError(
            f"X_ln/out must both be rank 3 or 4, got {X_ln.shape} and {out.shape}"
        )

    B, N, N2, C_in = X_ln.shape
    assert N == N2
    H = WZ_c.shape[0]
    D = WQ_c.shape[1] // H
    assert WKV_c.shape == (C_in, 2 * H * D)
    assert out.shape == (B, N, N, H * D)

    # The interleaved tl.reshape/tl.split path is retained unchanged for the
    # tuned D>=32 cases.  D=16 is the K-Fold Apo-module configuration and uses
    # separate projection matrices to avoid Hopper's illegal-access lowering.
    separate_kv = D == 16
    if separate_kv:
        if WK_c is None or WV_c is None:
            # Preserve the low-level public API: callers that only provide the
            # historical interleaved WKV tensor are deinterleaved once here.
            wkv_view = WKV_c.view(C_in, H, D, 2)
            WK_c = wkv_view[..., 0].reshape(C_in, H * D).contiguous()
            WV_c = wkv_view[..., 1].reshape(C_in, H * D).contiguous()
        assert WK_c.shape == WV_c.shape == (C_in, H * D)
        wk_ptr, wv_ptr = WK_c, WV_c
        wk_stride0, wk_stride1 = WK_c.stride()
        wv_stride0, wv_stride1 = WV_c.stride()
    else:
        # These pointers are compiled out when SEPARATE_KV=False.
        wk_ptr = wv_ptr = WKV_c
        wk_stride0 = wk_stride1 = wv_stride0 = wv_stride1 = 0

    io_dtype = tl_io_dtype(X_ln.dtype)
    use_int64 = max(X_ln.numel(), out.numel(), B * H * N * N) > (1 << 31)

    if Bias_buf is None:
        Bias_buf = torch.empty(B, H, N, N, device=X_ln.device, dtype=X_ln.dtype)
    elif Bias_buf.shape != (B, H, N, N):
        raise ValueError(f"Bias_buf must have shape {(B, H, N, N)}, got {Bias_buf.shape}")

    def grid_b(meta):
        return (triton.cdiv(B * N * N, meta["BLOCK_QK"]), H)

    bias_proj_kernel[grid_b](
        X_ln,
        X_ln.stride(0),
        X_ln.stride(1),
        X_ln.stride(2),
        X_ln.stride(3),
        WZ_c,
        WZ_c.stride(0),
        WZ_c.stride(1),
        BZ,
        BZ.stride(0),
        Bias_buf,
        Bias_buf.stride(0),
        Bias_buf.stride(1),
        Bias_buf.stride(2),
        Bias_buf.stride(3),
        B=B,
        N=N,
        C_in=C_in,
        H=H,
        IO_DTYPE=io_dtype,
        USE_INT64=use_int64,
    )

    has_mask = mask is not None
    if has_mask:
        if mask.shape != (B, N, N):
            raise ValueError(f"mask must have shape {(B, N, N)}, got {mask.shape}")
        mask_i8 = mask.to(torch.int8).contiguous()
        mask_ptr, mask_strideB, mask_stride0, mask_stride1 = (
            mask_i8,
            mask_i8.stride(0),
            mask_i8.stride(1),
            mask_i8.stride(2),
        )
    else:
        mask_ptr, mask_strideB, mask_stride0, mask_stride1 = X_ln, 0, 0, 0

    def grid_a(meta):
        return (B * N, H, triton.cdiv(N, meta["BLOCK_M"]))

    triangle_attn_kernel[grid_a](
        X_ln,
        X_ln.stride(0),
        X_ln.stride(1),
        X_ln.stride(2),
        X_ln.stride(3),
        WQ_c,
        WQ_c.stride(0),
        WQ_c.stride(1),
        WKV_c,
        WKV_c.stride(0),
        WKV_c.stride(1),
        wk_ptr,
        wk_stride0,
        wk_stride1,
        wv_ptr,
        wv_stride0,
        wv_stride1,
        Bias_buf,
        Bias_buf.stride(0),
        Bias_buf.stride(1),
        Bias_buf.stride(2),
        Bias_buf.stride(3),
        mask_ptr,
        mask_strideB,
        mask_stride0,
        mask_stride1,
        out,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        N=N,
        C_in=C_in,
        H=H,
        D=D,
        SCALE=scale,
        NEG_INF=-1e9,
        HAS_MASK=has_mask,
        SEPARATE_KV=separate_kv,
        IO_DTYPE=io_dtype,
        USE_INT64=use_int64,
    )
    return original_out


# ===========================================================================
# Kernel 3: fused gate epilogue  —  Out = (O * sigmoid(g_in @ Wg)) @ Wo
# Shared by triangle-attn wrap-up AND attention-pair-bias (trunk). The ONLY
# difference is how O is read (STRIDED_O constexpr):
#   STRIDED_O=True  : triangle wrap-up — O is (N, N, H, D), non-contiguous.
#                     decode row m -> (n1, n2), col c -> (h, d); gather at strides.
#   STRIDED_O=False : apb — O is (M, C) contiguous; read m*C + c directly.
# Everything else (g_in read, both matmuls, sigmoid, multiply, store) is shared.
# Compile-time branch -> two specialized kernels, zero runtime cost.
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm}, num_warps=w, num_stages=s)
        for bm in (64, 128, 256, 512)
        for w in (4, 8)
        for s in (2, 3)
    ],
    key=["M", "C"],
)
@triton.jit
def fused_gate_kernel(
    O_ptr,
    O_sB,
    O_sN1,
    O_sN2,
    O_sH,
    O_sD,  # strided-O args (used iff STRIDED_O)
    GIN_ptr,  # gate input (M, C) contiguous: LN(x) [tri] or ã [apb]
    WG_ptr,
    WO_ptr,  # (C, C) matmul-convention (already transposed)
    Out_ptr,  # (M, C) contiguous output
    M,
    N,  # N used iff STRIDED_O
    C: tl.constexpr,
    D: tl.constexpr,
    USE_INT64: tl.constexpr,
    STRIDED_O: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ptr = rows.to(tl.int64) if USE_INT64 else rows
    rmask = rows < M
    cols = tl.arange(0, C)

    if STRIDED_O:  # triangle wrap-up: O = (B, N, N, H, D), possibly strided
        b = row_ptr // (N * N)
        pair = row_ptr % (N * N)
        n1 = pair // N
        n2 = pair % N
        h = cols // D
        d = cols % D
        o_off = (
            b[:, None] * O_sB
            + n1[:, None] * O_sN1
            + n2[:, None] * O_sN2
            + h[None, :] * O_sH
            + d[None, :] * O_sD
        )
        o = tl.load(O_ptr + o_off, mask=rmask[:, None], other=0.0)
    else:  # apb: O = (M, C) contiguous
        o = tl.load(
            O_ptr + row_ptr[:, None] * C + cols[None, :], mask=rmask[:, None], other=0.0
        )

    gin = tl.load(
        GIN_ptr + row_ptr[:, None] * C + cols[None, :], mask=rmask[:, None], other=0.0
    )
    wg = tl.load(WG_ptr + cols[:, None] * C + cols[None, :])
    wo = tl.load(WO_ptr + cols[:, None] * C + cols[None, :])

    g = tl.sigmoid(tl.dot(gin, wg))  # gate = sigmoid(g_in @ Wg)
    acc = tl.dot((o.to(tl.float32) * g).to(gin.dtype), wo)  # (O * g) @ Wo
    tl.store(
        Out_ptr + row_ptr[:, None] * C + cols[None, :],
        acc.to(gin.dtype),
        mask=rmask[:, None],
    )


def fused_gate_forward(o_attn, qx_ln, WG, WO):
    """Apply sigmoid gating and output projection to triangle-attention output."""
    unbatched = o_attn.ndim == 4
    if unbatched:
        o_attn = o_attn.unsqueeze(0)
        qx_ln = qx_ln.unsqueeze(0)
    if o_attn.ndim != 5 or qx_ln.ndim != 4:
        raise ValueError(
            f"invalid triangle gate shapes: {o_attn.shape} and {qx_ln.shape}"
        )
    B, N1, N2, H, D = o_attn.shape
    if N1 != N2 or qx_ln.shape != (B, N1, N2, H * D):
        raise ValueError(
            f"incompatible triangle gate shapes: {o_attn.shape} and {qx_ln.shape}"
        )
    N, C, M = N1, H * D, B * N1 * N2
    out = torch.empty(B, N1, N2, C, device=o_attn.device, dtype=o_attn.dtype)
    Qf = qx_ln.reshape(M, C)  # free view (contiguous)
    Of = out.reshape(M, C)  # free view (contiguous)
    wg = WG.t().contiguous()  # (out,in) -> (in,out) so qx @ wg == linear_g(qx)
    wo = WO.t().contiguous()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    fused_gate_kernel[grid](
        o_attn,
        o_attn.stride(0),
        o_attn.stride(1),
        o_attn.stride(2),
        o_attn.stride(3),
        o_attn.stride(4),
        Qf,
        wg,
        wo,
        Of,
        M,
        N,
        USE_INT64=M * C > (1 << 31),
        C=C,
        D=D,
        STRIDED_O=True,
    )
    return out.squeeze(0) if unbatched else out
