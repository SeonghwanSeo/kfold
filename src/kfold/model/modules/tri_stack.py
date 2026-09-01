from functools import partial

import torch
import torch.nn as nn

from kfold.model.layers.folding.transition import Transition
from kfold.model.primitives import (
    DropoutColumnwise,
    DropoutRowwise,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from kfold.model.primitives.utils import add
from kfold.utils.checkpointing import checkpoint_blocks
from kfold.utils.kernels import TORCH_POLICY, KernelBackend, KernelPolicy


class TrianglularStack(nn.Module):
    """Pairformer stack."""

    def __init__(
        self,
        channel_z: int = 256,
        num_blocks: int = 48,
        dropout: float = 0.25,
        blocks_per_ckpt: int | None = None,
        kernel_policy: KernelPolicy = TORCH_POLICY,
    ):
        """Initialize the Pairformer module."""
        super().__init__()
        self.channel_z: int = channel_z
        self.dropout: float = dropout
        self.num_blocks: int = num_blocks
        self.kernel_policy = kernel_policy

        self.blocks_per_ckpt: int | None = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                TriangularBlock(
                    self.channel_z,
                    self.dropout,
                    kernel_policy=kernel_policy,
                )
            )

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_cuequiv_kernels: bool | None = None,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : torch.Tensor
            The pairwise embeddings
        pair mask : torch.Tensor
            The pair token mask
        use_cuequiv_kernels : bool | None, optional
            Legacy training argument. The backend is selected when the stack is
            constructed.
        Returns
        -------
        torch.Tensor
            The updated sequence embeddings.
        """
        if use_cuequiv_kernels is not None:
            expected = (
                KernelBackend.CUEQUIVARIANCE
                if use_cuequiv_kernels
                else KernelBackend.TORCH
            )
            if self.kernel_policy.triangle_multiplication is not expected:
                raise ValueError(
                    "The legacy training kernel flag does not match the "
                    "stack backend selected at construction."
                )

        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
            )
            for b in self.blocks
        ]
        z = checkpoint_blocks(
            blocks,
            (z,),
            self.blocks_per_ckpt,
            use_reentrant=False,
        )[0]

        return z


class TriangularBlock(nn.Module):
    """Pairformer block."""

    def __init__(
        self,
        channel_z: int = 256,
        dropout: float = 0.25,
        kernel_policy: KernelPolicy = TORCH_POLICY,
    ):
        """Initialize the Pairformer module.

        Parameters
        ----------
        channel_z : int
            The token pairwise embedding size.
        dropout : float, optional
            The dropout rate, by default 0.25
        """
        super().__init__()
        self.channel_z: int = channel_z
        self.dropout: float = dropout

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            channel_z, backend=kernel_policy.triangle_multiplication
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            channel_z, backend=kernel_policy.triangle_multiplication
        )
        self.transition_z = Transition(channel_z, expansion_factor=4)

        self.dropout_rowwise = DropoutRowwise(dropout)
        self.dropout_columnwise = DropoutColumnwise(dropout)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform the forward pass.
        See Section 3.6 Algorithm 20 Pairformer Stack
        """
        _add = partial(add, inplace=not self.training)

        z = _add(
            z,
            self.dropout_rowwise(self.tri_mul_out(z, pair_mask)),
        )

        z = _add(
            z,
            self.dropout_rowwise(self.tri_mul_in(z, pair_mask)),
        )

        z = _add(z, self.transition_z(z))
        return z
