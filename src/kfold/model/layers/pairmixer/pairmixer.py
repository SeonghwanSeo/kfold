"""Implementation of Pairmixer from
'Triangle Multiplication Is All You Need
For Biomolecular Structure Representations'.

Implementation follows details in https://doi.org/10.48550/arXiv.2510.18870.
"""

"""The following explains the authors' rationale for the adjustments.

Removing sequence updates
 - The MSA representation, obtained via the MSA Module, should sufficiently
   encode evolutionary information in z. The expressivity of z *should*
   mean that the sequences s don't need to be updated by z.

NOTE: That being said, if the rationale is a rich representation existing
**due to** the MSA representation, then might not be able to see comparative
performance by implementing vanilla Pairmixer.
 - Might require an intermediate between Pairmixer and Pairformer-style?

Removing triangular attention
 - Both triangular attention and triangular multiplication are used to update
   the pairwise embeddings z in a geometrically-consistent manner.
   The researchers claim that the ablations in AlphaFold2 shows that both
   triangular attention and triangular multiplication give strong performance
   and thus opt to use traingular multiplication only.

NOTE: 'removing triangular attention' rationale in p.586 of (Jumper, 2021)
"The triangle multiplicative update was developed originally as a more
symmetric and cheaper replacement for the attention, and networks that use
only the attention or multiplicative update are both able to produce high-
accuracy structures. However, the combination of the two updates is more
accurate."
"""

"""
NOTE: recall that the Pairformer stack architecture is used in largely three
contexts in AlphaFold3:

 1. Template Embedder (Algorithm 1 L9, Algorithm 16) - CAN'T
    - The original Pairformer code usees the `PairformerNoSeqModule`.
    - Note that we do not use a Template embedder yet.

 2. Pairformer trunk (Algorithm 1 L12, Algorithm 17) - DONE
    - The pairformer stack is where the main change exists.
      The fully modified pairmixer is used, and recycling of s, z are done.

 3. Confidence Module (Algorithm 1 L16, Algorithm 31) - CAN'T
    - The confidence module uses a modified pairmixer where
      the sequence attention is performed on the single representation
      (so the "removing sequence updates" section isn't applied here).
      TODO: think about why? this might warrant some more thinking/discussion
"""

from functools import partial

import torch
import torch.nn as nn

from kfold.model.layers.primitives.dropout import get_dropout_mask
from kfold.model.layers.primitives.triangle_multiplication import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from kfold.utils.checkpointing import checkpoint_blocks

from .transition import Transition
from .transformers import AttentionPairBias



class PairmixerStack(nn.Module):
    """Pairmixer stack.
    Follows the Pairformer stack implementation.
    """

    def __init__(
        self,
        channel_z: int = 128,
        num_blocks: int = 48,
        dropout: float = 0.25,
        blocks_per_ckpt: int | None = None,
    ):
        """Initialize the Pairmixer module."""
        super().__init__()
        self.channel_z: int = channel_z
        self.num_blocks: int = num_blocks
        self.dropout: float = dropout
        
        self.blocks_per_ckpt: int | None = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                PairmixerBlock(
                    self.channel_z,
                    self.dropout,
                )
            )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        use_cuequiv_mul: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The sequence embeddings; note that this is not updated at all
        z : torch.Tensor
            The pairwise embeddings
        mask : torch.Tensor
            The token mask
        use_cuequiv_mul : bool, optional
            Whether to use Cuequiv multiplication, by default False

        Returns
        -------
        torch.Tensor
            The sequence embeddings (unchanged)
        torch.Tensor
            The updated pairwise embeddings
        """
        # no assertion on chunk size for triangular attention

        pair_mask = mask[..., None] & mask[..., None, :]

        blocks = [
            partial(
                b,
                pair_mask=pair_mask.float(),
                use_cuequiv_mul=use_cuequiv_mul,
            )
            for b in self.blocks
        ]
        blocks_per_ckpt = self.blocks_per_ckpt

        if self.training and torch.is_grad_enabled():
            s, z = checkpoint_blocks(
                blocks,
                (s, z),
                blocks_per_ckpt,
                use_reentrant=False,
            )
        else:
            for block in blocks:
                s, z = block(s, z)

        # Line 10
        return s, z


class PairmixerBlock(nn.Module):
    """Pairmixer block.
    Follows the Pairformer block implementation."""

    def __init__(
        self,
        channel_z: int = 128,
        dropout: float = 0.25,
    ):
        """Initialize the Pairmixer module.
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

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_cuequiv_mul: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.
        See Section 3.6 Algorithm 20 Pairformer Stack
        """

        # Line 2
        dropout = get_dropout_mask(z, self.dropout, self.training)
        z = z + dropout * self.tri_mul_out(
            z,
            mask=pair_mask,
            use_kernels=use_cuequiv_mul,
        )

        # Line 3
        dropout = get_dropout_mask(z, self.dropout, self.training)
        z = z + dropout * self.tri_mul_in(
            z,
            mask=pair_mask,
            use_kernels=use_cuequiv_mul,
        )

        # Line 4, 5 removed (no attention)

        # Line 6
        z = z + self.transition_z(z)

        # Lines 7, 8 removed (no signal from pair rep. to single rep.)

        return s, z


# NOTE: the following two classes implement the Pairmixer Stack
# to be used in the confidence module.

class PairmixerWithSeqAttnModule(nn.Module):
    """Pairmixer with sequence attention module.
    """

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_blocks: int = 48,
        num_heads: int = 16,
        dropout: float = 0.25,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        """Initialize the Pairmixer with sequence attention updates.
        """
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.num_blocks: int = num_blocks
        self.num_heads: int = num_heads
        self.dropout: float = dropout
        self.blocks_per_ckpt: int | None = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range (num_blocks):
            self.blocks.append(
                PairmixerWithSeqAttnBlock(
                    self.channel_s,
                    self.channel_z,
                    self.num_heads,
                    self.dropout,
                )
            )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        use_cuequiv_mul: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.
        """

        pair_mask = mask[..., None] & mask[..., None, :]

        blocks = [
            partial(
                b,
                single_mask=mask.float(),
                pair_mask=pair_mask.float(),
                use_cuequiv_mul=use_cuequiv_mul,
            )
            for b in self.blocks
        ]
        blocks_per_ckpt = self.blocks_per_ckpt

        if self.training and torch.is_grad_enabled():
            s, z = checkpoint_blocks(
                blocks,
                (s, z),
                blocks_per_ckpt,
                use_reentrant=False,
            )
        else:
            for block in blocks:
                s, z = block(s, z)

        return s, z



class PairmixerWithSeqAttnBlock(nn.Module):
    """Pairmixer with sequence attention block.
    """

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        num_heads: int = 16,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.dropout: float = dropout
        self.num_heads: int = num_heads

        self.tri_mul_out = TriangleMultiplicationOutgoing(channel_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(channel_z)

        self.attention = AttentionPairBias(
            channel_a=channel_s,
            channel_z=channel_z,
            num_heads=num_heads,
            channel_s=None,
            use_single_cond=False,
        )

        self.transition_s = Transition(channel_s, expansion_factor=4)
        self.transition_z = Transition(channel_z, expansion_factor=4)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        use_cuequiv_mul: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pairwise rep updates
        dropout = get_dropout_mask(z, self.dropout, self.training)
        z = z + dropout * self.tri_mul_out(
            z,
            mask=pair_mask,
            use_kernels=use_cuequiv_mul,
        )

        dropout = get_dropout_mask(z, self.dropout, self.training)
        z = z + dropout * self.tri_mul_in(
            z,
            mask=pair_mask,
            use_kernels=use_cuequiv_mul,
        )

        z = z + self.transition_z(z)

        # single rep updates
        s = s + self.attention(
            s,
            None,
            z,
            attn_mask=single_mask
        )
        s = s + self.transition_s(s)

        return s, z
