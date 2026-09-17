# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
