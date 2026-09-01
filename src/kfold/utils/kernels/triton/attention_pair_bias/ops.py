"""Triton epilogues for attention-pair-bias output projection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._common.dtypes import tl_io_dtype


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K", "ATTENTION_BHLD", "IO_DTYPE", "DOT_INPUT_PRECISION"],
)
@triton.jit
def _gated_output_projection_kernel(
    gate_logits_ptr,
    attention_ptr,
    weight_ptr,
    output_ptr,
    attention_stride_b,
    attention_stride_h,
    attention_stride_l,
    attention_stride_d,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    LENGTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ATTENTION_BHLD: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    DOT_INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        input_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        gate_logits = tl.load(
            gate_logits_ptr + m_offsets[:, None] * K + k_offsets[None, :],
            mask=input_mask,
            other=0.0,
        ).to(tl.float32)

        if ATTENTION_BHLD:
            batch_offsets = m_offsets // LENGTH
            length_offsets = m_offsets % LENGTH
            head_offsets = k_offsets // HEAD_DIM
            dim_offsets = k_offsets % HEAD_DIM
            attention = tl.load(
                attention_ptr
                + batch_offsets[:, None] * attention_stride_b
                + head_offsets[None, :] * attention_stride_h
                + length_offsets[:, None] * attention_stride_l
                + dim_offsets[None, :] * attention_stride_d,
                mask=input_mask,
                other=0.0,
            )
        else:
            attention = tl.load(
                attention_ptr + m_offsets[:, None] * K + k_offsets[None, :],
                mask=input_mask,
                other=0.0,
            )

        gate = 1.0 / (1.0 + tl.exp(-gate_logits))
        gated = (gate * attention.to(tl.float32)).to(IO_DTYPE)
        weight = tl.load(
            weight_ptr + k_offsets[:, None] * N + n_offsets[None, :],
            mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(
            gated,
            weight,
            input_precision=DOT_INPUT_PRECISION,
        ).to(tl.float32)

    tl.store(
        output_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        accumulator.to(IO_DTYPE),
        mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N),
    )


def _validate_projection_inputs(
    x: torch.Tensor,
    attention: torch.Tensor,
    w_gate: torch.Tensor,
    w_out: torch.Tensor,
) -> None:
    if not x.is_cuda:
        raise RuntimeError("APB fused epilogue requires CUDA tensors")
    if x.ndim != 3 or not x.is_contiguous():
        raise ValueError(f"x must be contiguous BLC, got {x.shape} {x.stride()}")
    channel = x.shape[-1]
    if w_gate.shape != (channel, channel) or w_out.shape != (channel, channel):
        raise ValueError(
            "gate/output weights must use matmul convention (C, C), got "
            f"{w_gate.shape} and {w_out.shape} for C={channel}"
        )
    tensors = (attention, w_gate, w_out)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("attention and projection weights must share x.device")
    if any(tensor.dtype != x.dtype for tensor in tensors):
        raise ValueError("attention and projection weights must share x.dtype")
    if not w_gate.is_contiguous() or not w_out.is_contiguous():
        raise ValueError("projection weights must be contiguous")


def _gated_output_projection(
    x: torch.Tensor,
    attention: torch.Tensor,
    w_gate: torch.Tensor,
    w_out: torch.Tensor,
    *,
    attention_bhld: bool,
    fp32_dot_precision: str,
) -> torch.Tensor:
    _validate_projection_inputs(x, attention, w_gate, w_out)
    if fp32_dot_precision not in {"ieee", "tf32"}:
        raise ValueError(f"unsupported FP32 dot precision {fp32_dot_precision!r}")

    batch, length, channel = x.shape
    if attention_bhld:
        if (
            attention.ndim != 4
            or attention.shape[0] != batch
            or attention.shape[2] != length
        ):
            raise ValueError(f"attention must be BHLD for gate shape {x.shape}")
        heads, head_dim = attention.shape[1], attention.shape[3]
        if heads * head_dim != channel:
            raise ValueError(
                f"attention H*D must equal C, got {heads}*{head_dim} != {channel}"
            )
        attention_strides = attention.stride()
    else:
        if attention.shape != x.shape or not attention.is_contiguous():
            raise ValueError(
                f"attention must be contiguous BLC with x shape, got {attention.shape} "
                f"{attention.stride()}"
            )
        head_dim = channel
        attention_strides = (0, 0, 0, 0)

    gate_logits = x @ w_gate
    output = torch.empty_like(x)
    rows = batch * length

    def grid(meta):
        return (
            triton.cdiv(rows, meta["BLOCK_M"]),
            triton.cdiv(channel, meta["BLOCK_N"]),
        )

    _gated_output_projection_kernel[grid](
        gate_logits,
        attention,
        w_out,
        output,
        *attention_strides,
        M=rows,
        N=channel,
        K=channel,
        LENGTH=length,
        HEAD_DIM=head_dim,
        ATTENTION_BHLD=attention_bhld,
        IO_DTYPE=tl_io_dtype(x.dtype),
        DOT_INPUT_PRECISION=(fp32_dot_precision if x.dtype == torch.float32 else "tf32"),
    )
    return output


def gated_output_projection_blc(
    x: torch.Tensor,
    attention: torch.Tensor,
    w_gate: torch.Tensor,
    w_out: torch.Tensor,
    *,
    fp32_dot_precision: str = "ieee",
) -> torch.Tensor:
    """Apply gating and output projection to BLC attention output."""
    return _gated_output_projection(
        x,
        attention,
        w_gate,
        w_out,
        attention_bhld=False,
        fp32_dot_precision=fp32_dot_precision,
    )


def gated_output_projection_bhld(
    x: torch.Tensor,
    attention: torch.Tensor,
    w_gate: torch.Tensor,
    w_out: torch.Tensor,
    *,
    fp32_dot_precision: str = "ieee",
) -> torch.Tensor:
    """Apply gating and output projection directly to BHLD attention output."""
    return _gated_output_projection(
        x,
        attention,
        w_gate,
        w_out,
        attention_bhld=True,
        fp32_dot_precision=fp32_dot_precision,
    )


__all__ = ["gated_output_projection_bhld", "gated_output_projection_blc"]
