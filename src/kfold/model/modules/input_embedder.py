from dataclasses import dataclass

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.folding.embeddings import (
    ApoEmbedding,
    ConstraintEncoding,
    RelativePositionEncoding,
)
from kfold.model.layers.folding.input_encoder import InputFeatureEmbedder
from kfold.model.primitives import LinearNoBias
from kfold.utils.config import configurable


@configurable
class InputEmbedder(torch.nn.Module):
    """Input embedding module for KFold model."""

    @dataclass(kw_only=True)
    class Config:
        """Configuration for the Input embedding module.

        Parameters
        ----------
        channel_s : int
            The token single embedding size.
        channel_z : int
            The token pairwise embedding size.
        channel_atom : int
            The atom single embedding size.
        channel_atompair : int
            The atom pairwise embedding size.
        atom_encoder_blocks: int
            The atom encoder blocks.
        atom_encoder_heads: int
            The atom encoder heads.
            The number of bins for apo distance map encoding.
        """

        channel_s: int = 384
        channel_z: int = 256
        channel_atom: int = 128
        channel_atompair: int = 16
        atom_encoder_blocks: int = 3
        atom_encoder_heads: int = 4
        # Apo embedding parameters
        num_bins: int = 39
        min_dist: float = 3.25
        max_dist: float = 50.75
        max_r: int = 64
        # Constraint-related parameters
        constraint_min_dist: float = 2.0
        constraint_max_dist: float = 20.0
        constraint_bin_size: float = 1.0

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.channel_s: int = cfg.channel_s
        self.channel_z: int = cfg.channel_z
        self.channel_atom: int = cfg.channel_atom
        self.channel_atompair: int = cfg.channel_atompair

        self.input_embedder = InputFeatureEmbedder(
            channel_s=cfg.channel_s,
            channel_atom=cfg.channel_atom,
            channel_atompair=cfg.channel_atompair,
            atom_encoder_blocks=cfg.atom_encoder_blocks,
            atom_encoder_heads=cfg.atom_encoder_heads,
        )

        # Initial linear layers for single and pair representations
        self.linear_z_init1 = LinearNoBias(cfg.channel_s, cfg.channel_z)
        self.linear_z_init2 = LinearNoBias(cfg.channel_s, cfg.channel_z)
        self.rel_pos_encoding = RelativePositionEncoding(r_max=32, s_max=2)
        self.linear_rel_pos = LinearNoBias(self.rel_pos_encoding.dimension, cfg.channel_z)
        self.linear_bond = LinearNoBias(1, cfg.channel_z)

        # Apo embedding
        self.apo_embedding = ApoEmbedding(
            num_bins=cfg.num_bins,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            max_r=cfg.max_r,
        )
        self.linear_apo = LinearNoBias(
            self.apo_embedding.num_channels, cfg.channel_z, init="default"
        )

        # Constraint-related
        self.constraint_encoding = ConstraintEncoding(
            min_dist=cfg.constraint_min_dist,
            max_dist=cfg.constraint_max_dist,
            bin_size=cfg.constraint_bin_size,
        )
        self.linear_constraint = LinearNoBias(
            self.constraint_encoding.num_bins, cfg.channel_z
        )

    def forward(
        self,
        f_input: FoldingInput,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of embedding module.
        See Section 3 Algorithm 1 and Algorithm 2 of AlphaFold3 paper.
        Algorithm 1 Line[1-5]

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.

        Returns
        -------
        s_inputs : torch.Tensor
            Tensor of shape (B, L, C_s) containing input single features
        z_init: torch.Tensor
            Tensor of shape (B, L, L, C_z) containing initial pair representation
            before trunk.
        """
        inplace = not self.training
        add = (lambda x, y: x.add_(y)) if inplace else (lambda x, y: x + y)  # noqa

        # Get input single representation
        s_inputs = self.input_embedder(f_input)  # [B, L, c_s]

        # Get initial pair representation
        z_init = (
            self.linear_z_init1(s_inputs)[..., None, :, :]
            + self.linear_z_init2(s_inputs)[..., :, None, :]
        )  # [B, L, L, c_z]
        dtype = z_init.dtype

        # Add relative positional encoding
        z_init = add(z_init, self.linear_rel_pos(self.rel_pos_encoding(f_input, dtype)))

        # Add bond adjacency matrix
        z_init = add(
            z_init, self.linear_bond(self.get_bond_adj(f_input, dtype).unsqueeze(-1))
        )

        # Add apo embedding
        z_init = add(z_init, self.linear_apo(self.apo_embedding(f_input)))

        # Add constraing embedding
        z_init = add(
            z_init, self.linear_constraint(self.constraint_encoding(f_input, dtype))
        )

        return s_inputs, z_init

    def get_bond_adj(
        self,
        f_input: FoldingInput,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get the adjacency bond matrix from the input features.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.

        Returns
        -------
        adj : torch.Tensor
            Tensor of shape (B, L, L) containing the adjacency matrix.
        """
        bond_index = f_input.bond.token_index  # [B, num_bonds, 2]
        mask = f_input.bond.pad_mask  # [B, num_bonds]

        B, L = f_input.batch_size, f_input.num_tokens
        dev = f_input.device

        # Masking; (0, 0) is padding index
        bond_index = bond_index.masked_fill(~mask[..., None], 0)

        # Create adjacency matrix
        src, dst = bond_index[:, :, 0], bond_index[:, :, 1]
        adj = torch.zeros((B, L, L), device=dev, dtype=dtype)
        b_idc = torch.arange(B, device=dev).unsqueeze(-1)
        adj[b_idc, src, dst] = 1.0
        adj[b_idc, dst, src] = 1.0  # undirected

        # Padding is always located at index (0,)
        adj[:, 0, 0] = 0.0
        return adj
