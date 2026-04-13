from kfold.model.layers.alphafold3.diffusion import DiffusionStack

from .atom_transformer import AtomEmbedderWithApo


class ApoConditionedDiffusionStack(DiffusionStack):
    """Modified AlphaFold3 diffusion module with apo structure conditioning."""

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        channel_atom: int = 128,
        channel_atompair: int = 16,
        channel_coords: int = 3,
        atom_encoder_blocks: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_blocks: int = 24,
        token_transformer_heads: int = 16,
        atom_decoder_blocks: int = 3,
        atom_decoder_heads: int = 4,
        blocks_per_ckpt: int | None = None,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        channel_s : int
            The single representation dimension.
        channel_z : int
            The pair representation dimension.
        channel_atom : int
            The atom single representation dimension.
        channel_atompair : int
            The atom pair representation dimension.
        channel_coords : int
            The atom coordinates dimension, by default 3.
        atom_encoder_blocks : int, optional
            The number of blocks in the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_blocks : int, optional
            The number of blocks in the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 16.
        atom_decoder_blocks : int, optional
            The number of blocks in the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint for gradient checkpointing,
            by default None.

        """
        super().__init__(
            channel_s,
            channel_z,
            channel_atom,
            channel_atompair,
            channel_coords,
            atom_encoder_blocks,
            atom_encoder_heads,
            token_transformer_blocks,
            token_transformer_heads,
            atom_decoder_blocks,
            atom_decoder_heads,
            blocks_per_ckpt,
        )
        # === Local atom-level attention encoder === #
        self.atom_embedder = AtomEmbedderWithApo(
            channel_s=channel_s,
            channel_z=channel_z,
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            use_structure=True,
        )
