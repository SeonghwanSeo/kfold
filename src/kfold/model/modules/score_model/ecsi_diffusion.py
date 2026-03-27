from kfold.model.layers.alphafold3.diffusion import DiffusionModule
from kfold.utils.registry import SCORE_MODEL

from .af3_diffusion import AF3DiffusionModule
from .base import BaseScoreModel


@SCORE_MODEL.register()
class ECSIDiffusionModule(AF3DiffusionModule):
    """Diffusion score model with apo structure conditioning."""

    class Config(AF3DiffusionModule.Config):
        """Configuration for the apo-conditioned diffusion module.

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
        use_prior_coords : bool, optional
            Whether to use prior coordinates (apo structure) in the score model.
             If True, the score model will take both x_t and x_T as input.
             If False, the score model will take only x_t as input.
             By default True.
        """

        use_prior_coords: bool = True

    def __init__(self, cfg: Config, kernel_config):
        BaseScoreModel.__init__(self, cfg, kernel_config)
        effective_channel_coords = 6 if cfg.use_prior_coords else 3
        self.diffusion_stack = DiffusionModule(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            channel_atom=cfg.channel_atom,
            channel_atompair=cfg.channel_atompair,
            channel_coords=effective_channel_coords,
            atom_encoder_blocks=cfg.atom_encoder_blocks,
            atom_encoder_heads=cfg.atom_encoder_heads,
            token_transformer_blocks=cfg.token_transformer_blocks,
            token_transformer_heads=cfg.token_transformer_heads,
            atom_decoder_blocks=cfg.atom_decoder_blocks,
            atom_decoder_heads=cfg.atom_decoder_heads,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )
