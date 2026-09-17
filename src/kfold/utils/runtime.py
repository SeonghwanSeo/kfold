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
