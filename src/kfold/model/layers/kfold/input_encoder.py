from kfold.model.layers.alphafold3.input_encoder import InputFeatureEmbedder

from .atom_transformer import AtomEmbedderWithApo


class InputEmbedderWithApo(InputFeatureEmbedder):
    """Input embedding module with apo structure embedding"""

    def __init__(
        self,
        channel_s: int = 384,
        channel_atom: int = 128,
        channel_atompair: int = 16,
        atom_encoder_blocks: int = 3,
        atom_encoder_heads: int = 4,
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
        atom_encoder_blocks: int
            The number of blocks in atom encoder.
        atom_encoder_heads: int
            The number of heads in atom encoder.
        """
        super().__init__(
            channel_s,
            channel_atom,
            channel_atompair,
            atom_encoder_blocks,
            atom_encoder_heads,
        )
        # Replace the embedder with one that includes apo structure embedding
        self.embedder = AtomEmbedderWithApo(
            channel_s=channel_s,
            channel_z=None,
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            use_structure=False,
        )
