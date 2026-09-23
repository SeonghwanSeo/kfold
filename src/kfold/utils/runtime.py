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

import importlib.util
from functools import lru_cache


@lru_cache(maxsize=1)
def is_cuda_available() -> bool:
    """Return whether CUDA is available."""
    if importlib.util.find_spec("torch") is None:
        return False

    import torch

    return torch.cuda.is_available()


@lru_cache(maxsize=1)
def is_cuequivariance_installed() -> bool:
    """Return whether the cuequivariance PyTorch package is installed."""
    return importlib.util.find_spec("cuequivariance_torch") is not None


@lru_cache(maxsize=1)
def is_triton_available() -> bool:
    """Return whether Triton is installed and CUDA is available."""
    if importlib.util.find_spec("triton") is None:
        return False
    return is_cuda_available()


def select_kernel_backend(backend: str = "auto") -> str:
    """Resolve auto to Triton, cuEquivariance, or PyTorch, in that order."""
    if backend not in ("auto", "torch", "triton", "cuequiv"):
        raise ValueError(
            f"Unknown kernel_backend {backend!r}. "
            "Expected 'auto', 'torch', 'triton', or 'cuequiv'."
        )
    if backend == "auto":
        if is_triton_available():
            return "triton"
        if is_cuequivariance_installed():
            return "cuequiv"
        return "torch"
    if backend == "triton":
        if not is_triton_available():
            raise ValueError(
                "Triton is not available. "
                "Please ensure that Triton is installed and CUDA is available."
            )
    elif backend == "cuequiv":
        if not is_cuequivariance_installed():
            raise ValueError(
                "cuequivariance_torch is not installed. "
                "Please install it to use the 'cuequiv' backend."
            )
    return backend
