"""Kernel backend selection for optimized K-Fold operations."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any


class KernelBackend(enum.StrEnum):
    """Implementations available to an optimized model operation."""

    TORCH = "torch"
    CUEQUIVARIANCE = "cuequivariance"
    TRITON = "triton"


@dataclasses.dataclass(frozen=True)
class KernelPolicy:
    """Backend selected for each optimized operation."""

    triangle_attention: KernelBackend
    triangle_multiplication: KernelBackend
    attention_pair_bias: KernelBackend

    @classmethod
    def from_config(cls, config: Any) -> KernelPolicy:
        """Resolve the new backend field or the legacy cueq boolean."""
        backend = getattr(config, "kernel_backend", None)
        if backend is None:
            backend = (
                KernelBackend.CUEQUIVARIANCE
                if getattr(config, "kernel_cuequivariance", True)
                else KernelBackend.TORCH
            )
        return kernel_policy_from_name(backend)


TORCH_POLICY = KernelPolicy(
    triangle_attention=KernelBackend.TORCH,
    triangle_multiplication=KernelBackend.TORCH,
    attention_pair_bias=KernelBackend.TORCH,
)
CUEQUIVARIANCE_POLICY = KernelPolicy(
    triangle_attention=KernelBackend.CUEQUIVARIANCE,
    triangle_multiplication=KernelBackend.CUEQUIVARIANCE,
    attention_pair_bias=KernelBackend.CUEQUIVARIANCE,
)
TRITON_POLICY = KernelPolicy(
    triangle_attention=KernelBackend.TRITON,
    triangle_multiplication=KernelBackend.TRITON,
    attention_pair_bias=KernelBackend.TRITON,
)

KERNEL_POLICIES = {
    KernelBackend.TORCH.value: TORCH_POLICY,
    KernelBackend.CUEQUIVARIANCE.value: CUEQUIVARIANCE_POLICY,
    KernelBackend.TRITON.value: TRITON_POLICY,
}

_BACKEND_ALIASES = {
    "naive": KernelBackend.TORCH.value,
    "cueq": KernelBackend.CUEQUIVARIANCE.value,
}


def kernel_policy_from_name(name: str | KernelBackend) -> KernelPolicy:
    """Return a named policy and reject unknown YAML values."""
    normalized = str(name)
    normalized = _BACKEND_ALIASES.get(normalized, normalized)
    try:
        return KERNEL_POLICIES[normalized]
    except (KeyError, TypeError) as exc:
        expected = tuple(KERNEL_POLICIES)
        raise ValueError(
            f"Unknown kernel backend {name!r}; expected one of {expected}."
        ) from exc
