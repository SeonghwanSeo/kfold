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


class TrianglularStack(nn.Module):
    """Pairformer stack."""

    def __init__(
        self,
        channel_z: int = 256,
        num_blocks: int = 48,
        dropout: float = 0.25,
        blocks_per_ckpt: int | None = None,
    ):
        """Initialize the Pairformer module."""
        super().__init__()
        self.channel_z: int = channel_z
        self.dropout: float = dropout
        self.num_blocks: int = num_blocks

        self.blocks_per_ckpt: int | None = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                TriangularBlock(
                    self.channel_z,
                    self.dropout,
                )
            )

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_cuequiv_kernels: bool = False,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : torch.Tensor
            The pairwise embeddings
        pair mask : torch.Tensor
            The pair token mask
        use_cuequiv_kernels : bool, optional
            Whether to use CuEQuiv kernels, by default False

        Returns
        -------
        torch.Tensor
            The updated sequence embeddings.
        """
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                use_cuequiv_kernels=use_cuequiv_kernels,
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

        self.tri_mul_out = TriangleMultiplicationOutgoing(channel_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(channel_z)
        self.transition_z = Transition(channel_z, expansion_factor=4)

        self.dropout_rowwise = DropoutRowwise(dropout)
        self.dropout_columnwise = DropoutColumnwise(dropout)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_cuequiv_kernels: bool = False,
    ) -> torch.Tensor:
        """Perform the forward pass.
        See Section 3.6 Algorithm 20 Pairformer Stack
        """
        _add = partial(add, inplace=not self.training)

        z = _add(
            z,
            self.dropout_rowwise(
                self.tri_mul_out(z, pair_mask, use_kernels=use_cuequiv_kernels)
            ),
        )

        z = _add(
            z,
            self.dropout_rowwise(
                self.tri_mul_in(z, pair_mask, use_kernels=use_cuequiv_kernels)
            ),
        )

        z = _add(z, self.transition_z(z))
        return z
