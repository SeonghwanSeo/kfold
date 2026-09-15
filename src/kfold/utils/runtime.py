import importlib.util
from functools import lru_cache


@lru_cache(maxsize=1)
def is_cuequivariance_installed() -> bool:
    """Return whether the cuequivariance PyTorch package is installed."""
    return importlib.util.find_spec("cuequivariance_torch") is not None
