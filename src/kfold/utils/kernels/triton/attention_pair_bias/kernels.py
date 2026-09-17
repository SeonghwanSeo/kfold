"""Triton kernels for attention with a cached pair bias.

Short queries fuse QKV projection with attention; longer queries use a single
QKV projection followed by attention.
"""

import torch
import triton
import triton.language as tl

from .._common.dtypes import tl_io_dtype


@triton.autotune(
    configs=[
        # ---- Small BLOCK_M (16) — short query rows, min registers ----
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_K": 32, "BLOCK_C": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_K": 32, "BLOCK_C": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_K": 64, "BLOCK_C": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_K": 64, "BLOCK_C": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_K": 128, "BLOCK_C": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_K": 128, "BLOCK_C": 64}, num_warps=8, num_stages=2
        ),
        # ---- Medium BLOCK_M (32) — one program per atom window (Lq=32) ----
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_C": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_C": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_C": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_C": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_C": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_C": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_C": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 128, "BLOCK_C": 32}, num_warps=8, num_stages=2
        ),
        # ---- Large BLOCK_M (64) — token lengths: fewer K/V re-projections ----
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 32, "BLOCK_C": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 32, "BLOCK_C": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_C": 32}, num_warps=8, num_stages=2
        ),
        # Do not add BLOCK_M/K/C=64/64/64 with 8 warps here. On H100 with
        # Triton 3.7.0 both stage-2 and stage-3 variants pass a single launch,
        # but reproducibly illegal-address during triton.autotune's repeated
        # local-APB benchmark (MQ/MK/N/H/D=32/128/128/4/32, BF16).
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_C": 128}, num_warps=8, num_stages=2
        ),
        # ---- XL BLOCK_M (128) — long token rows; recompute amortized hardest ----
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 32, "BLOCK_C": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_C": 32}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_C": 64}, num_warps=8, num_stages=2
        ),
    ],
    key=["MQ", "MK", "N", "H", "D"],
)
@triton.jit
def _apb_fused_qkv_attention_kernel(
    AQ_ptr,
    AQ_stride0,
    AQ_stride1,
    AQ_stride2,  # (B, MQ, N) normalized query-side input
    AK_ptr,
    AK_stride0,
    AK_stride1,
    AK_stride2,  # (B, MK, N) normalized key/value-side input
    Wqkv_ptr,
    Wqkv_stride0,
    Wqkv_stride1,  # (N, 3*H*D) concat: per-head [Q | K | V]
    Bq_ptr,
    Bq_stride0,  # (H*D,) linear_q bias (K/V projections have none)
    Z_ptr,
    Z_stride0,
    Z_stride1,
    Z_stride2,
    Z_stride3,  # (B/Z_MULT, H, MQ, MK) cached projected pair bias
    Msk_ptr,
    Msk_stride0,
    Msk_stride1,  # (B/MSK_MULT, MK) key mask (nonzero = attend); unused if not HAS_MASK
    Og_ptr,
    Og_stride0,
    Og_stride1,
    Og_stride2,  # (B, MQ, H*D) attention output (pre-gate, pre-Wo)
    MQ: tl.constexpr,
    MK: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,  # next_pow2(D); d_offs >= D lanes are masked to 0
    Z_MULT: tl.constexpr,  # batch entries sharing one Z batch slot
    MSK_MULT: tl.constexpr,  # batch entries sharing one mask row
    HAS_MASK: tl.constexpr,
    SCALE: tl.constexpr,
    INF: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    DOT_INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)
    q_mask = q_offs < MQ
    d_mask = d_offs < D

    qkv_base = 3 * pid_h * D
    head_col = pid_h * D + d_offs

    AQ_b = AQ_ptr + pid_b * AQ_stride0
    AK_b = AK_ptr + pid_b * AK_stride0
    Z_bh = Z_ptr + (pid_b // Z_MULT) * Z_stride0 + pid_h * Z_stride1
    Msk_b = Msk_ptr + (pid_b // MSK_MULT) * Msk_stride0

    # --- Q projection: Q = A_q @ WQ_h + bq_h, streamed over BLOCK_C chunks ---
    Q_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for c_start in range(0, N, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < N
        X_q = tl.load(
            AQ_b + q_offs[:, None] * AQ_stride1 + c_offs[None, :] * AQ_stride2,
            mask=q_mask[:, None] & c_mask[None, :],
            other=0.0,
        )
        WQ_c = tl.load(
            Wqkv_ptr
            + c_offs[:, None] * Wqkv_stride0
            + (qkv_base + d_offs)[None, :] * Wqkv_stride1,
            mask=c_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        Q_acc += tl.dot(X_q, WQ_c, input_precision=DOT_INPUT_PRECISION)
    Bq_h = tl.load(Bq_ptr + head_col * Bq_stride0, mask=d_mask, other=0.0).to(tl.float32)
    Q_io = (Q_acc + Bq_h[None, :]).to(IO_DTYPE)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, MK, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < MK

        # --- K/V projection for this key block (recomputed per q_block) ---
        K_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
        V_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
        for c_start in range(0, N, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < N
            X_k = tl.load(
                AK_b + k_offs[:, None] * AK_stride1 + c_offs[None, :] * AK_stride2,
                mask=k_mask[:, None] & c_mask[None, :],
                other=0.0,
            )
            WK_c = tl.load(
                Wqkv_ptr
                + c_offs[:, None] * Wqkv_stride0
                + (qkv_base + D + d_offs)[None, :] * Wqkv_stride1,
                mask=c_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            WV_c = tl.load(
                Wqkv_ptr
                + c_offs[:, None] * Wqkv_stride0
                + (qkv_base + 2 * D + d_offs)[None, :] * Wqkv_stride1,
                mask=c_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            K_acc += tl.dot(X_k, WK_c, input_precision=DOT_INPUT_PRECISION)
            V_acc += tl.dot(X_k, WV_c, input_precision=DOT_INPUT_PRECISION)
        K_io = K_acc.to(IO_DTYPE)
        V_io = V_acc.to(IO_DTYPE)

        S = (
            tl.dot(Q_io, tl.trans(K_io), input_precision=DOT_INPUT_PRECISION).to(
                tl.float32
            )
            * SCALE
        )

        bias_tile = tl.load(
            Z_bh + q_offs[:, None] * Z_stride2 + k_offs[None, :] * Z_stride3,
            mask=q_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # attn_bias = pair_bias + (~mask) * (-INF): FINITE (matches the additive
        # -1e9 upstream), so an all-masked row stays NaN-free; only the k >= MK
        # padding uses a true -inf (never a whole block, so m_i stays finite).
        if HAS_MASK:
            key_valid = tl.load(Msk_b + k_offs * Msk_stride1, mask=k_mask, other=0) != 0
            bias_tile += tl.where(key_valid, 0.0, -INF)[None, :]
        S += bias_tile
        S = tl.where(k_mask[None, :], S, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(S, axis=1))
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])
        l_i = l_i * alpha + tl.sum(P, axis=1)
        O_acc = O_acc * alpha[:, None] + tl.dot(
            P.to(IO_DTYPE), V_io, input_precision=DOT_INPUT_PRECISION
        ).to(tl.float32)
        m_i = m_new

    Out = O_acc / l_i[:, None]
    tl.store(
        Og_ptr
        + pid_b * Og_stride0
        + q_offs[:, None] * Og_stride1
        + head_col[None, :] * Og_stride2,
        Out.to(IO_DTYPE),
        mask=q_mask[:, None] & d_mask[None, :],
    )


@triton.autotune(
    configs=[
        # No projection inside: register pressure is just Q/K/V/S tiles, so
        # larger BLOCK_M/BLOCK_K are viable than in the fused kernel.
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["MQ", "MK", "H", "D"],
)
@triton.jit
def _apb_split_attention_kernel(
    Q_ptr,
    Q_stride0,
    Q_stride1,
    Q_stride2,
    Q_stride3,  # (B, MQ, H, D) pre-projected, possibly strided view of a QKV GEMM
    K_ptr,
    K_stride0,
    K_stride1,
    K_stride2,
    K_stride3,  # (B, MK, H, D)
    V_ptr,
    V_stride0,
    V_stride1,
    V_stride2,
    V_stride3,  # (B, MK, H, D)
    Bq_ptr,
    Bq_stride0,  # (H*D,) linear_q bias, added to the loaded Q tile
    Z_ptr,
    Z_stride0,
    Z_stride1,
    Z_stride2,
    Z_stride3,  # (B/Z_MULT, H, MQ, MK) cached projected pair bias
    Msk_ptr,
    Msk_stride0,
    Msk_stride1,  # (B/MSK_MULT, MK) key mask; unused if not HAS_MASK
    Og_ptr,
    Og_stride0,
    Og_stride1,
    Og_stride2,  # (B, MQ, H*D)
    MQ: tl.constexpr,
    MK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    Z_MULT: tl.constexpr,
    MSK_MULT: tl.constexpr,
    HAS_MASK: tl.constexpr,
    SCALE: tl.constexpr,
    INF: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    DOT_INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)
    q_mask = q_offs < MQ
    d_mask = d_offs < D

    head_col = pid_h * D + d_offs

    Q_bh = Q_ptr + pid_b * Q_stride0 + pid_h * Q_stride2
    K_bh = K_ptr + pid_b * K_stride0 + pid_h * K_stride2
    V_bh = V_ptr + pid_b * V_stride0 + pid_h * V_stride2
    Z_bh = Z_ptr + (pid_b // Z_MULT) * Z_stride0 + pid_h * Z_stride1
    Msk_b = Msk_ptr + (pid_b // MSK_MULT) * Msk_stride0

    Q_tile = tl.load(
        Q_bh + q_offs[:, None] * Q_stride1 + d_offs[None, :] * Q_stride3,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    Bq_h = tl.load(Bq_ptr + head_col * Bq_stride0, mask=d_mask, other=0.0).to(tl.float32)
    Q_io = (Q_tile + Bq_h[None, :]).to(IO_DTYPE)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, MK, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < MK

        kv_mask = k_mask[:, None] & d_mask[None, :]
        K_io = tl.load(
            K_bh + k_offs[:, None] * K_stride1 + d_offs[None, :] * K_stride3,
            mask=kv_mask,
            other=0.0,
        ).to(IO_DTYPE)
        V_io = tl.load(
            V_bh + k_offs[:, None] * V_stride1 + d_offs[None, :] * V_stride3,
            mask=kv_mask,
            other=0.0,
        ).to(IO_DTYPE)

        S = (
            tl.dot(Q_io, tl.trans(K_io), input_precision=DOT_INPUT_PRECISION).to(
                tl.float32
            )
            * SCALE
        )

        bias_tile = tl.load(
            Z_bh + q_offs[:, None] * Z_stride2 + k_offs[None, :] * Z_stride3,
            mask=q_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if HAS_MASK:
            key_valid = tl.load(Msk_b + k_offs * Msk_stride1, mask=k_mask, other=0) != 0
            bias_tile += tl.where(key_valid, 0.0, -INF)[None, :]
        S += bias_tile
        S = tl.where(k_mask[None, :], S, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(S, axis=1))
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])
        l_i = l_i * alpha + tl.sum(P, axis=1)
        O_acc = O_acc * alpha[:, None] + tl.dot(
            P.to(IO_DTYPE), V_io, input_precision=DOT_INPUT_PRECISION
        ).to(tl.float32)
        m_i = m_new

    Out = O_acc / l_i[:, None]
    tl.store(
        Og_ptr
        + pid_b * Og_stride0
        + q_offs[:, None] * Og_stride1
        + head_col[None, :] * Og_stride2,
        Out.to(IO_DTYPE),
        mask=q_mask[:, None] & d_mask[None, :],
    )


# Above this many query rows the fused kernel's per-q_block K/V re-projection
# (ceil(MQ/BLOCK_M) duplicates) outweighs one QKV GEMM + HBM round trip.
_SPLIT_MQ_THRESHOLD = 128


def _apb_split_qkv_projection(A_q, A_k, W_QKV, H, D, W_Q=None, W_KV=None):
    """Project QKV once and return (B, L, H, D) views."""
    N = A_q.shape[-1]
    if (
        A_q.data_ptr() == A_k.data_ptr()
        and A_q.shape == A_k.shape
        and A_q.stride() == A_k.stride()
    ):
        # Self attention: one dense GEMM, then strided per-head [Q|K|V] views.
        qkv = A_q @ W_QKV  # (B, L, 3*H*D)
        qkv = qkv.view(*qkv.shape[:-1], H, 3, D)
        return qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
    if W_Q is not None or W_KV is not None:
        if W_Q is None or W_KV is None:
            raise ValueError("cross-attention split requires both W_Q and W_KV")
        if W_Q.shape != (N, H * D) or W_KV.shape != (N, 2 * H * D):
            raise ValueError(
                "invalid packed cross-attention weights: expected "
                f"{(N, H * D)} and {(N, 2 * H * D)}, got "
                f"{W_Q.shape} and {W_KV.shape}"
            )
        q = (A_q @ W_Q).view(*A_q.shape[:-1], H, D)
        kv = (A_k @ W_KV).view(*A_k.shape[:-1], H, 2, D)
        return q, kv[..., 0, :], kv[..., 1, :]

    # Compatibility fallback. Production callers should pre-pack W_Q/W_KV
    # once; these slices otherwise copy weights and launch three GEMMs.
    W = W_QKV.view(N, H, 3, D)
    q = (A_q @ W[:, :, 0, :].reshape(N, H * D)).view(*A_q.shape[:-1], H, D)
    k = (A_k @ W[:, :, 1, :].reshape(N, H * D)).view(*A_k.shape[:-1], H, D)
    v = (A_k @ W[:, :, 2, :].reshape(N, H * D)).view(*A_k.shape[:-1], H, D)
    return q, k, v


def apb_diffusion_forward(
    A_q,
    A_k,
    W_QKV,
    B_q,
    Z,
    key_mask,
    Og,
    scale=1.0,
    inf=1e9,
    mode="auto",
    fp32_dot_precision="ieee",
    split_w_q=None,
    split_w_kv=None,
):
    """Write cached-pair-bias attention output to Og.

    mode selects fused or split QKV projection. Z and key_mask may broadcast over
    consecutive batch groups.
    """
    if A_q.ndim == 2:
        A_q, A_k = A_q.unsqueeze(0), A_k.unsqueeze(0)
    if Z.ndim == 3:
        Z = Z.unsqueeze(0)

    B, MQ, N = A_q.shape
    MK = A_k.shape[1]
    H = Z.shape[1]
    D = W_QKV.shape[1] // (3 * H)
    assert A_k.shape == (B, MK, N)
    assert W_QKV.shape == (N, 3 * H * D)
    assert B_q.shape == (H * D,)
    assert Z.shape == (Z.shape[0], H, MQ, MK) and B % Z.shape[0] == 0
    assert Og.shape == (B, MQ, H * D)
    assert D >= 16, "tl.dot needs head_dim >= 16"
    if fp32_dot_precision not in ("ieee", "tf32"):
        raise ValueError(
            f"fp32_dot_precision must be 'ieee' or 'tf32', got {fp32_dot_precision!r}"
        )

    if key_mask is None:
        # No mask: point the mask args at Z with zero strides; never loaded.
        has_mask, msk_mult = False, 1
        key_mask, msk_stride0, msk_stride1 = Z, 0, 0
    else:
        if key_mask.ndim == 1:
            key_mask = key_mask.unsqueeze(0)
        if key_mask.dtype == torch.bool:
            key_mask = key_mask.to(torch.uint8)
        assert key_mask.shape == (key_mask.shape[0], MK) and B % key_mask.shape[0] == 0
        has_mask, msk_mult = True, B // key_mask.shape[0]
        msk_stride0, msk_stride1 = key_mask.stride(0), key_mask.stride(1)

    if mode == "auto":
        mode = "fused" if MQ <= _SPLIT_MQ_THRESHOLD else "split"

    def grid(meta):
        return (triton.cdiv(MQ, meta["BLOCK_M"]), H, B)

    common = dict(
        MQ=MQ,
        MK=MK,
        H=H,
        D=D,
        BLOCK_D=triton.next_power_of_2(D),
        Z_MULT=B // Z.shape[0],
        MSK_MULT=msk_mult,
        HAS_MASK=has_mask,
        SCALE=scale,
        INF=inf,
        IO_DTYPE=tl_io_dtype(A_q.dtype),
        # KFold and AtlasFold inference use "highest" FP32 matmul precision.
        # Keep BF16/FP16 on tensor cores, but do not silently lower FP32 APB
        # to TF32 inside Triton.
        DOT_INPUT_PRECISION=(
            fp32_dot_precision if A_q.dtype == torch.float32 else "tf32"
        ),
    )
    if mode == "fused":
        _apb_fused_qkv_attention_kernel[grid](
            A_q,
            A_q.stride(0),
            A_q.stride(1),
            A_q.stride(2),
            A_k,
            A_k.stride(0),
            A_k.stride(1),
            A_k.stride(2),
            W_QKV,
            W_QKV.stride(0),
            W_QKV.stride(1),
            B_q,
            B_q.stride(0),
            Z,
            Z.stride(0),
            Z.stride(1),
            Z.stride(2),
            Z.stride(3),
            key_mask,
            msk_stride0,
            msk_stride1,
            Og,
            Og.stride(0),
            Og.stride(1),
            Og.stride(2),
            N=N,
            **common,
        )
    elif mode == "split":
        q, k, v = _apb_split_qkv_projection(
            A_q,
            A_k,
            W_QKV,
            H,
            D,
            W_Q=split_w_q,
            W_KV=split_w_kv,
        )  # (B, L, H, D) views
        _apb_split_attention_kernel[grid](
            q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            B_q,
            B_q.stride(0),
            Z,
            Z.stride(0),
            Z.stride(1),
            Z.stride(2),
            Z.stride(3),
            key_mask,
            msk_stride0,
            msk_stride1,
            Og,
            Og.stride(0),
            Og.stride(1),
            Og.stride(2),
            **common,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'auto', 'fused', or 'split'")
    return Og
