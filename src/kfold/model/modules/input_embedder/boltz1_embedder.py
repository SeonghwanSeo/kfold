import torch
import torch.nn as nn

from kfold.data.model_input import FoldingInput
from kfold.model.layers.boltz1.encoders import RelativePositionEncoder
from kfold.model.layers.boltz1.trunk import InputEmbedder
from kfold.utils.registry import INPUT_EMBEDDER, BaseConfig

from .base import BaseInputEmbedder


@INPUT_EMBEDDER.register()
class Boltz1InputEmbedder(BaseInputEmbedder):
    """Input embedding module based on Boltz1."""

    class Config(BaseConfig):
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
        atoms_per_window_queries: int,
            The number of atoms per window for queries.
        atoms_per_window_keys: int,
            The number of atoms per window for keys.
        atom_encoder_blocks: int,
            The atom encoder blocks.
        atom_encoder_heads: int,
            The atom encoder heads.
        max_relative_token : int
            The maximum relative residue distance for relative position encoding.
        max_relative_chain : int
            The maximum relative chain distance for relative position encoding.
        """

        channel_s: int = 384
        channel_z: int = 128
        channel_atom: int = 128
        channel_atompair: int = 16
        atoms_per_window_queries: int = 32
        atoms_per_window_keys: int = 128
        atom_encoder_blocks: int = 3
        atom_encoder_heads: int = 4
        max_relative_token: int = 32
        max_relative_chain: int = 2

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.channel_s = cfg.channel_s
        self.channel_z = cfg.channel_z
        self.channel_atom = cfg.channel_atom
        self.channel_atompair = cfg.channel_atompair

        token_s = cfg.channel_s
        token_z = cfg.channel_z
        atom_s = cfg.channel_atom
        atom_z = cfg.channel_atompair

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": 32,
            "atoms_per_window_keys": 128,
            "atom_feature_dim": 389,
            "no_atom_encoder": False,
            "atom_encoder_depth": 3,
            "atom_encoder_heads": 4,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)

        # Input projections
        s_input_dim = token_s + 2 * 33 + 1 + 4
        self.s_init = nn.Linear(s_input_dim, token_s, bias=False)
        self.z_init_1 = nn.Linear(s_input_dim, token_z, bias=False)
        self.z_init_2 = nn.Linear(s_input_dim, token_z, bias=False)

        self.rel_pos = RelativePositionEncoder(token_z)
        self.token_bonds = nn.Linear(1, token_z, bias=False)

    def forward(
        self,
        f_input: FoldingInput,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of embedding module.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.

        Returns
        -------
        s_inputs : torch.Tensor
            Tensor of shape (B, L, C_s) containing input single features
        s_init: torch.Tensor
            Tensor of shape (B, L, C_s) containing initial single representation
            before trunk.
        z_init: torch.Tensor
            Tensor of shape (B, L, L, C_s) containing initial pair representation
            before trunk.
        """

        s_inputs = self.input_embedder(f_input)

        # Initialize the sequence and pairwise embeddings
        s_init = self.s_init(s_inputs)
        z_init = self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        relative_position_encoding = self.rel_pos(f_input)
        z_init = z_init + relative_position_encoding

        # Line 5
        z_init = z_init + self.token_bonds(
            self.get_adjacency_matrix(
                f_input.bond.token_index, f_input.num_tokens, f_input.bond.pad_mask
            ).unsqueeze(-1)  # [B, L, L, 1]
        )  # [B, L, L, c_z]

        return s_inputs, s_init, z_init

    def get_adjacency_matrix(
        self, bond_index: torch.Tensor, num_tokens: int, mask: torch.Tensor
    ) -> torch.Tensor:
        """Get the adjacency bond matrix from the input features."""

        # Masking; (0, 0) is padding index
        bond_index = bond_index * mask.unsqueeze(-1)

        src, dst = bond_index[:, :, 0], bond_index[:, :, 1]

        batch_size = bond_index.shape[0]
        adj = torch.zeros(
            (batch_size, num_tokens, num_tokens),
            device=bond_index.device,
            dtype=torch.float32,
        )

        batch_indices = (
            torch.arange(batch_size, device=bond_index.device)
            .unsqueeze(-1)
            .expand_as(src)
        )

        adj[batch_indices, src, dst] = 1.0
        adj[batch_indices, dst, src] = 1.0  # undirected

        # Padding is always located at index (0,)
        adj[:, 0, 0] = 0
        return adj
