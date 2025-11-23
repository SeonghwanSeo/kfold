import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from kfold.data.model_input import FoldingInput

from .attention import AttentionPairBias
from .dropout import get_dropout_mask
from .encoders import AtomAttentionEncoder
from .transition import Transition
from .triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from .triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class InputEmbedder(nn.Module):
    """Input embedder."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        no_atom_encoder: bool = False,
    ) -> None:
        """Initialize the input embedder.

        Parameters
        ----------
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        token_s : int
            The single token representation dimension.
        token_z : int
            The pair token representation dimension.
        atoms_per_window_queries : int
            The number of atoms per window for queries.
        atoms_per_window_keys : int
            The number of atoms per window for keys.
        atom_feature_dim : int
            The atom feature dimension.
        atom_encoder_depth : int
            The atom encoder depth.
        atom_encoder_heads : int
            The atom encoder heads.
        no_atom_encoder : bool, optional
            Whether to use the atom encoder, by default False

        """
        super().__init__()
        self.token_s = token_s
        self.no_atom_encoder = no_atom_encoder

        if not no_atom_encoder:
            self.atom_attention_encoder = AtomAttentionEncoder(
                atom_s=atom_s,
                atom_z=atom_z,
                token_s=token_s,
                token_z=token_z,
                atoms_per_window_queries=atoms_per_window_queries,
                atoms_per_window_keys=atoms_per_window_keys,
                atom_feature_dim=atom_feature_dim,
                atom_encoder_depth=atom_encoder_depth,
                atom_encoder_heads=atom_encoder_heads,
                structure_prediction=False,
            )

    def forward(self, f_input: FoldingInput) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        f_input : FoldingInput
            Input features

        Returns
        -------
        Tensor
            The embedded tokens.

        """
        # Load relevant features
        res_type = torch.cat(
            [torch.zeros_like(f_input.token.res_type[..., :1]), f_input.token.res_type],
            dim=-1,
        )
        # Use res_type as profile
        profile = res_type

        deletion_mean = torch.zeros_like(res_type[..., :1])

        # pocket_feature: one-hot vector (index=0)
        pocket_feature = torch.zeros_like(res_type[..., :4])
        pocket_feature[..., 0] = 1

        # Compute input embedding
        if self.no_atom_encoder:
            a = torch.zeros(
                (res_type.shape[0], res_type.shape[1], self.token_s),
                device=res_type.device,
            )
        else:
            a, _, _, _, _ = self.atom_attention_encoder(f_input)
        s = torch.cat([a, res_type, profile, deletion_mean, pocket_feature], dim=-1)
        return s


class PairformerModule(nn.Module):
    """Pairformer module."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        no_update_s: bool = False,
        no_update_z: bool = False,
        chunk_size_tri_attn: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize the Pairformer module.

        Parameters
        ----------
        token_s : int
            The token single embedding size.
        token_z : int
            The token pairwise embedding size.
        num_blocks : int
            The number of blocks.
        num_heads : int, optional
            The number of heads, by default 16
        dropout : float, optional
            The dropout rate, by default 0.25
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        no_update_s : bool, optional
            Whether to update the single embeddings, by default False
        no_update_z : bool, optional
            Whether to update the pairwise embeddings, by default False
        chunk_size_tri_attn : int, optional
            The chunk size for triangle attention, by default None

        """
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads

        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            self.layers.append(
                PairformerLayer(
                    token_s,
                    token_z,
                    num_heads,
                    dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                    no_update_s,
                    False if i < num_blocks - 1 else no_update_z,
                    chunk_size_tri_attn,
                ),
            )

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s : Tensor
            The sequence embeddings
        z : Tensor
            The pairwise embeddings
        mask : Tensor
            The token mask
        pair_mask : Tensor
            The pairwise mask
        Returns
        -------
        Tensor
            The updated sequence embeddings.
        Tensor
            The updated pairwise embeddings.

        """
        if self.training:
            for layer in self.layers:
                s, z = torch.utils.checkpoint.checkpoint(
                    layer,
                    s,
                    z,
                    mask,
                    pair_mask,
                    use_reentrant=False,
                )
        else:
            for layer in self.layers:
                s, z = layer(s, z, mask, pair_mask)
        return s, z


class PairformerLayer(nn.Module):
    """Pairformer module."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        no_update_s: bool = False,
        no_update_z: bool = False,
        chunk_size_tri_attn: int | None = None,
    ) -> None:
        """Initialize the Pairformer module.

        Parameters
        ----------
        token_s : int
            The token single embedding size.
        token_z : int
            The token pairwise embedding size.
        num_heads : int, optional
            The number of heads, by default 16
        dropout : float, optiona
            The dropout rate, by default 0.25
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        no_update_s : bool, optional
            Whether to update the single embeddings, by default False
        no_update_z : bool, optional
            Whether to update the pairwise embeddings, by default False
        chunk_size_tri_attn : int, optional
            The chunk size for triangle attention, by default None

        """
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.num_heads = num_heads
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        self.chunk_size_tri_attn = chunk_size_tri_attn
        if not self.no_update_s:
            self.attention = AttentionPairBias(token_s, token_z, num_heads)
        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        if not self.no_update_s:
            self.transition_s = Transition(token_s, token_s * 4)
        self.transition_z = Transition(token_z, token_z * 4)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Perform the forward pass."""
        # Compute pairwise stack
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=self.chunk_size_tri_attn if not self.training else None,
        )

        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=self.chunk_size_tri_attn if not self.training else None,
        )

        z = z + self.transition_z(z)

        # Compute sequence stack
        if not self.no_update_s:
            s = s + self.attention(s, z, mask)
            s = s + self.transition_s(s)

        return s, z
