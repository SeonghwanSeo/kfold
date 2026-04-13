from kfold.model.layers.alphafold3.diffusion import DiffusionStack
from kfold.utils.registry import SCORE_MODEL, BaseConfig

from .base import AF3StyleDiffusionModule


@SCORE_MODEL.register()
class AF3DiffusionModule(AF3StyleDiffusionModule):
    """AF3 Diffusion module
    Section 3.7 Algorithm 20: Diffusion Module in the AF3 paper.
    """

    class Config(BaseConfig):
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
        atom_encoder_blocks : int, optional
            The number of blocks of the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_blocks : int, optional
            The number of blocks of the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 8.
        atom_decoder_blocks : int, optional
            The number of blocks of the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint, by default None.
        """

        channel_s: int = 384
        channel_z: int = 128
        channel_atom: int = 128
        channel_atompair: int = 16
        atom_encoder_blocks: int = 3
        atom_encoder_heads: int = 4
        token_transformer_blocks: int = 24
        token_transformer_heads: int = 16
        atom_decoder_blocks: int = 3
        atom_decoder_heads: int = 4
        blocks_per_ckpt: int | None = None

    def __init__(self, cfg: Config, kernel_config):
        super().__init__(cfg, kernel_config)
        self.diffusion_stack = DiffusionStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            channel_atom=cfg.channel_atom,
            channel_atompair=cfg.channel_atompair,
            channel_coords=3,
            atom_encoder_blocks=cfg.atom_encoder_blocks,
            atom_encoder_heads=cfg.atom_encoder_heads,
            token_transformer_blocks=cfg.token_transformer_blocks,
            token_transformer_heads=cfg.token_transformer_heads,
            atom_decoder_blocks=cfg.atom_decoder_blocks,
            atom_decoder_heads=cfg.atom_decoder_heads,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )
