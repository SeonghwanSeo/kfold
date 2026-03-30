from functools import partial

import torch
import torch.nn as nn

from kfold.model.layers.primitives import AdaLN, Linear, LinearNoBias, SwiGLU
from kfold.utils.checkpointing import checkpoint_blocks

from .attention_pair_bias import CrossAttentionPairBias, SelfAttentionPairBias
from .utils import build_atom_to_qk_fn


class ConditionedTransitionBlock(nn.Module):
    """Conditioned Transition Block
    Section 3.7 Algorithm 25 Conditioned Transition Block
    """

    def __init__(self, channel_a: int, channel_s: int, expansion_factor: int = 2):
        """Initialize the conditioned transition block.

        Parameters
        ----------
        channel_a : int
            The atom/token dimension.
        channel_s : int
            The single conditioning dimension.
        expansion_factor : int, optional
            The expansion factor, by default 2

        """
        super().__init__()
        self.adaln = AdaLN(channel_a, channel_s)

        model_dim = int(channel_a * expansion_factor)  # Line 2
        self.swiglu = SwiGLU(channel_a, model_dim)

        self.linear_g = Linear(channel_s, channel_a, init="gating_closed")
        self.linear_out = LinearNoBias(model_dim, channel_a, init="default")
        self.sigmoid = nn.Sigmoid()

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """See Section 3.7 Algorithm 25 Conditioned Transition Block"""
        # Line 1
        a = self.adaln(a, s)
        # Line 2
        b = self.swiglu(a)
        # Line 3
        a = self.sigmoid(self.linear_g(s)) * self.linear_out(b)
        return a


class GlobalTransformerStack(nn.Module):
    """Global Attention Diffusion Transformer Stack.
    Section 3.7 Algorithm 23 Diffusion Transformer [Line 1,4]
    """

    def __init__(
        self,
        channel_a: int,
        channel_s: int,
        channel_z: int,
        num_heads: int,
        num_blocks: int,
        blocks_per_ckpt: int | None = None,
    ):
        """Initialize the diffusion transformer.

        Parameters
        ----------
        channel_a : int
            The single representation dimension.
        channel_s : int
            The single conditioning dimension.
        channel_z : int
            The pair representation dimension.
        num_heads : int
            The number of heads.
        num_blocks : int
            The number of blocks.
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks_per_ckpt: int | None = blocks_per_ckpt
        for _ in range(num_blocks):
            self.blocks.append(
                GlobalTransformerBlock(channel_a, channel_s, channel_z, num_heads)
            )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        use_cuequiv_kernels: bool = False,
    ):
        """See Section 3.7 Algorithm 23 Diffusion Transformer

        Parameters
        ----------
        a : torch.Tensor
            The input single representation tensor (*, L, c_a)
        s : torch.Tensor
            The input single conditioning tensor (*, L, c_s)
        z : torch.Tensor
            The input pair representation tensor (*, Lq, Lk, c_z)
        mask : torch.Tensor
            The attention mask tensor (*, L)
        use_cuequiv_kernels : bool, optional
            Whether to use custom kernel for attention, by default False
        """
        # Line 1, 4
        blocks = [
            partial(
                b,
                s=s,
                z=z,
                mask=mask,
                use_cuequiv_kernels=use_cuequiv_kernels,
            )
            for b in self.blocks
        ]
        a = checkpoint_blocks(
            blocks,
            args=(a,),
            blocks_per_ckpt=self.blocks_per_ckpt,
            use_reentrant=False,
        )[0]
        return a


class GlobalTransformerBlock(nn.Module):
    """Global Attention Diffusion Transformer Block.
    Section 3.7 Algorithm 23 Diffusion Transformer [Line 2-3]
    """

    def __init__(
        self,
        channel_a: int,
        channel_s: int,
        channel_z: int,
        num_heads: int,
    ):
        """Initialize the diffusion transformer block.

        Parameters
        ----------
        channel_a : int
            The single representation dimension.
        channel_s : int
            The single conditioning dimension.
        channel_z : int
            The pair representation dimension.
        num_heads : int
            The number of heads.
        """
        super().__init__()
        self.attention = SelfAttentionPairBias(channel_a, channel_z, num_heads, channel_s)
        self.transition = ConditionedTransitionBlock(channel_a, channel_s)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        use_cuequiv_kernels: bool = False,
    ) -> torch.Tensor:
        """See Section 3.7 Algorithm 23 Diffusion Transformer

        Parameters
        ----------
        a : torch.Tensor
            The input single representation tensor (*, L, c_a)
        s : torch.Tensor
            The input single conditioning tensor (*, L, c_s)
        z : torch.Tensor
            The input pair representation tensor (*, L, L, c_z)
        mask : torch.Tensor
            The attention mask tensor (*, L)

        Returns
        -------
        a : torch.Tensor
            The output single representation tensor (*, L, c_a)

        NOTE
        ----
        Below is the original Algorithm 23:
        b = AttentionPairBias(a, s, z)  # Line 2
        a = b + ConditionedTransitionBlock(a, s)  # Line 3

        However, its official implementation uses residual connections:
        a = a + AttentionPairBias(a, s, z)
        a = a + ConditionedTransitionBlock(a, s)

        See https://github.com/google-deepmind/alphafold3/blob/f3e86f27dfac16559d16f470bb2f9323eb357f1f/src/alphafold3/model/network/diffusion_transformer.py#L209-L226

        """
        # Line 2
        a = a + self.attention(a, s, z, mask, use_kernels=use_cuequiv_kernels)
        # Line 3
        a = a + self.transition(a, s)
        return a


class LocalTransformerStack(nn.Module):
    """Diffusion Transformer Stack with Local Attention.
    Section 3.7 Algorithm 7 Atom Transformer
    Section 3.7 Algorithm 23 Diffusion Transformer [Line 1,4]
    """

    def __init__(
        self,
        channel_a: int,
        channel_s: int,
        channel_z: int,
        num_heads: int,
        num_blocks: int,
    ):
        """Initialize the diffusion transformer.

        Parameters
        ----------
        channel_a : int
            The single representation dimension.
        channel_s : int
            The single conditioning dimension.
        channel_z : int
            The pair representation dimension.
        num_heads : int
            The number of heads.
        num_blocks : int
            The number of blocks.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                LocalTransformerBlock(channel_a, channel_s, channel_z, num_heads)
            )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """See Section 3.7 Algorithm 23 Diffusion Transformer

        Parameters
        ----------
        a : torch.Tensor
            The input single representation tensor (*, L, c_a)
        s : torch.Tensor
            The input single conditioning tensor (*, L, c_s)
        z : torch.Tensor
            The input pair representation tensor (*, Lq, Lk, c_z)
        mask : torch.Tensor
            The attention mask tensor (*, L)

        Returns
        -------
        a : torch.Tensor
            The output single representation tensor (*, L, c_a)
        """
        to_qk = build_atom_to_qk_fn(a.shape[-2], a.device)
        s_q, s_k = to_qk(s, -2)
        _, mask_k = to_qk(mask, -1)

        for block in self.blocks:
            a_q, a_k = to_qk(a, -2)  # [*, W, Lq/Lk, c_a]
            a_q = block(a_q, a_k, s_q, s_k, z, mask_k)
            a = a_q.flatten(-3, -2)  # [*, L, c_a]
        return a


class LocalTransformerBlock(nn.Module):
    """Diffusion Transformer Block with Local Attention.
    Section 3.7 Algorithm 7 Atom Transformer
    Section 3.7 Algorithm 23 Diffusion Transformer [Line 2-3]
    """

    def __init__(
        self,
        channel_a: int,
        channel_s: int,
        channel_z: int,
        num_heads: int,
    ):
        """Initialize the diffusion transformer block.

        Parameters
        ----------
        channel_a : int
            The single representation dimension.
        channel_s : int
            The single conditioning dimension.
        channel_z : int
            The pairwise dimension.
        heads : int
            The number of heads.
        """
        super().__init__()
        self.attention = CrossAttentionPairBias(
            channel_a, channel_z, num_heads, channel_s
        )
        self.transition = ConditionedTransitionBlock(channel_a, channel_s)

    def forward(
        self,
        a_q: torch.Tensor,
        a_k: torch.Tensor,
        s_q: torch.Tensor,
        s_k: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """See Section 3.7 Algorithm 23 Diffusion Transformer

        Parameters
        ----------
        a_q : torch.Tensor
            The query single representation tensor (*, Lq, c_a)
        a_k : torch.Tensor
            The key single representation tensor (*, Lk, c_a)
        s_q : torch.Tensor
            The input single conditioning tensor (*, Lq, c_s)
        s_k : torch.Tensor
            The key single conditioning tensor (*, Lk, c_s)
        z : torch.Tensor
            The input pair representation tensor (*, Lq, Lk, c_z)
        mask : torch.Tensor
            The attention mask tensor (*, Lk)

        Returns
        -------
        a_q : torch.Tensor
            The output single representation tensor (*, Lq, c_a)

        NOTE
        ----
        Below is the original Algorithm 23:
        b = AttentionPairBias(a, s, z)  # Line 2
        a = b + ConditionedTransitionBlock(a, s)  # Line 3

        However, its official implementation uses residual connections:
        a = a + AttentionPairBias(a, s, z)
        a = a + ConditionedTransitionBlock(a, s)

        See https://github.com/google-deepmind/alphafold3/blob/f3e86f27dfac16559d16f470bb2f9323eb357f1f/src/alphafold3/model/network/diffusion_transformer.py#L209-L226

        """
        # Line 2
        a_q = a_q + self.attention(a_q, a_k, s_q, s_k, z, mask)
        # Line 3
        a_q = a_q + self.transition(a_q, s_q)
        return a_q
