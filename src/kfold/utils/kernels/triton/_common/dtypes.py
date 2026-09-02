"""Shared dtype helpers."""

import torch
import triton.language as tl

# Map a torch IO dtype to the Triton constexpr the kernels store/cast with.
TORCH_TO_TL = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def tl_io_dtype(dtype: torch.dtype):
    """Return the `tl.*` dtype for a torch IO dtype, raising on unsupported."""
    try:
        return TORCH_TO_TL[dtype]
    except KeyError as e:
        raise ValueError(
            f"unsupported IO dtype {dtype}; expected one of {list(TORCH_TO_TL)}"
        ) from e
