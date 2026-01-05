import torch

import kfold.constants as C
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.primitives import LinearNoBias

from .transformers import AtomAttentionEncoder


class InputFeatureEmbedder(torch.nn.Module):
    """Input embedding module based on AlphaFold3.
    See Section 3 Algorithm 2 of the AlphaFold3 paper.
    """

    def __init__(
        self,
        channel_s: int = 384,
        channel_atom: int = 128,
        channel_atompair: int = 16,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_encoder_blocks: int = 3,
        atom_encoder_heads: int = 4,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        """Initialize the Input feature embedding module.

        Parameters
        ----------
        channel_s : int
            The token single embedding size.
        channel_atom : int
            The atom single embedding size.
        channel_atompair : int
            The atom pairwise embedding size.
        atoms_per_window_queries: int,
            The number of atoms per window for queries.
        atoms_per_window_keys: int,
            The number of atoms per window for keys.
        atom_encoder_blocks: int,
            The number of blocks in atom encoder.
        atom_encoder_heads: int,
            The number of heads in atom encoder.
        """
        super().__init__()

        self.encoder = AtomAttentionEncoder(
            channel_s=channel_s,
            channel_z=None,
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            channel_token=channel_s,  # Same to channel_s
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            num_blocks=atom_encoder_blocks,
            num_heads=atom_encoder_heads,
            use_structure=False,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        # residue info
        self.num_res_types: int = C.NUM_RES_TYPES  # = 32
        assert self.num_res_types == 32, "Expected num_res_types to be 32."

        # out projection
        # NOTE: (SeonghwanSeo) I introduce additional linear layer to unify the dimension.
        s_input_dim = channel_s + self.num_res_types
        self.proj_s = LinearNoBias(s_input_dim, channel_s, init="default")

    def forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        input : FoldingInput
            Input features

        Returns
        -------
        Tensor
            The embedded tokens. [B, Lt, c_s]
        """
        # Atom attention encoder forward
        a, *_ = self.encoder(f_input, None, None, None)  # [B, Lt, c_s]

        # Concatenate additional token features
        res_type = f_input.token.res_type  # [B, Lt, 32]
        s = torch.cat([a, res_type], dim=-1)

        # Project to model dimension
        # NOTE: (SeonghwanSeo) I introduce additional linear layer to unify the dimension.
        s = self.proj_s(s)  # [B, Lt, c_s]
        return s
