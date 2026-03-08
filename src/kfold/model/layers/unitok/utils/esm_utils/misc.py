from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch


def fp32_autocast_context(device_type: str) -> AbstractContextManager[Any]:
    """
    Returns an autocast context manager that disables downcasting by AMP.

    Args:
        device_type: The device type ('cpu' or 'cuda')

    Returns:
        An autocast context manager with the specified behavior.
    """
    if device_type == "cpu":
        return torch.amp.autocast(device_type, enabled=False)
    elif device_type == "mps":
        # For MPS, just return a no-op context manager (nullcontext) since MPS does not
        # support autocast.
        return nullcontext()
    elif device_type == "cuda":
        return torch.amp.autocast(device_type, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported device type: {device_type}")
