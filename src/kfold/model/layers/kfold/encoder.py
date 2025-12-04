import torch

import kfold.constants as C
from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.input_encoder import InputFeatureEmbedder
from kfold.model.layers.kfold.transformers import AtomAttentionEncoderWithApo
from kfold.model.layers.primitives import LinearNoBias


class PretrainedInputEmbedder(InputFeatureEmbedder):
    """Input embedding module with pre-trained embeddings."""

    def __init__(
        self,
        channel_s: int = 384,
        channel_atom: int = 128,
        channel_atompair: int = 16,
        channel_seq_encoder: int | None = None,
        channel_struct_encoder: int | None = None,
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
            The token single embedding size.
        channel_atompair : int
            The token pairwise embedding size.
        channel_seq_encoder : int | None
            The pre-trained sequence encoder output channel size.
        channel_struct_encoder : int | None
            The pre-trained structure encoder output channel size.
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
        # pre-trained embedding projection

        self.use_seq_enc = channel_seq_encoder is not None and channel_seq_encoder > 0
        if channel_seq_encoder is not None:
            self.proj_seq_enc = LinearNoBias(channel_seq_encoder, channel_s, init="zero")
        self.use_struct_enc = (
            channel_struct_encoder is not None and channel_struct_encoder > 0
        )
        if channel_struct_encoder is not None:
            self.proj_struct_enc = LinearNoBias(
                channel_struct_encoder, channel_s, init="zero"
            )

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
        s = super().forward(f_input)  # [B, Lt, c_s]

        # Add pre-trained sequence embedding if available
        if self.use_seq_enc:
            assert f_input.pretrained.has_sequence_embedding, (
                "Pre-trained sequence embedding is not available in the input."
            )
            seq_enc = f_input.pretrained.sequence_embedding  # [B, Lt, c_seq_enc]
            s = s + self.proj_seq_enc(seq_enc)

        # Add pre-trained structure embedding if available
        if self.use_struct_enc:
            assert f_input.pretrained.has_structure_embedding, (
                "Pre-trained structure embedding is not available in the input."
            )
            struct_enc = f_input.pretrained.structure_embedding  # [B, Lt, c_struct_enc]
            s = s + self.proj_struct_enc(struct_enc)

        return s


class PretrainedInputEmbedderWithApo(torch.nn.Module):
    """Input embedding module with pre-trained embeddings."""

    def __init__(
        self,
        channel_s: int = 384,
        channel_atom: int = 128,
        channel_atompair: int = 16,
        channel_seq_encoder: int | None = None,
        channel_struct_encoder: int | None = None,
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
            The token single embedding size.
        channel_atompair : int
            The token pairwise embedding size.
        channel_seq_encoder : int | None
            The pre-trained sequence encoder output channel size.
        channel_struct_encoder : int | None
            The pre-trained structure encoder output channel size.
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

        self.encoder = AtomAttentionEncoderWithApo(
            channel_s=channel_s,
            channel_z=None,
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            channel_token=channel_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            num_blocks=atom_encoder_blocks,
            num_heads=atom_encoder_heads,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        # residue info
        self.num_res_types: int = C.NUM_RES_TYPES  # = 32
        assert self.num_res_types == 32, "Expected num_res_types to be 32."

        # out projection
        # NOTE: (SeonghwanSeo) I introduce additional linear layer to unify the dimension.
        s_input_dim = channel_s + self.num_res_types
        self.proj_s = LinearNoBias(s_input_dim, channel_s, init="default")

        # pre-trained embedding projection
        self.use_seq_enc = channel_seq_encoder is not None
        if channel_seq_encoder is not None:
            self.proj_seq_enc = LinearNoBias(channel_seq_encoder, channel_s, init="zero")
        self.use_struct_enc = channel_struct_encoder is not None
        if channel_struct_encoder is not None:
            self.proj_struct_enc = LinearNoBias(
                channel_struct_encoder, channel_s, init="zero"
            )

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
        a, *_ = self.encoder(f_input)  # [B, Lt, c_s]

        # Concatenate additional token features
        res_type = f_input.token.res_type  # [B, Lt, 32]
        s = torch.cat(
            [a, res_type],
            dim=-1,
        )

        # Project to model dimension
        # NOTE: (SeonghwanSeo) I introduce additional linear layer to unify the dimension.
        s = self.proj_s(s)  # [B, Lt, c_s]

        # Add pre-trained sequence embedding if available
        if self.use_seq_enc:
            assert f_input.pretrained.has_sequence_embedding, (
                "Pre-trained sequence embedding is not available in the input."
            )
            seq_enc = f_input.pretrained.sequence_embedding  # [B, Lt, c_seq_enc]
            s = s + self.proj_seq_enc(seq_enc)
        # Add pre-trained structure embedding if available
        if self.use_struct_enc:
            assert f_input.pretrained.has_structure_embedding, (
                "Pre-trained structure embedding is not available in the input."
            )
            struct_enc = f_input.pretrained.structure_embedding  # [B, Lt, c_struct_enc]
            s = s + self.proj_struct_enc(struct_enc)
        return s
