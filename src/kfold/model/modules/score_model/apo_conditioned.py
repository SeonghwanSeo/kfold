from kfold.model.layers.kfold.diffusion import ApoConditionedDiffusionStack
from kfold.utils.registry import SCORE_MODEL

from .af3_diffusion import AF3DiffusionModule
from .base import AF3StyleDiffusionModule


@SCORE_MODEL.register()
class ApoConditionedDiffusionModule(AF3DiffusionModule):
    """Apo-conditioned Diffusion module"""

    def __init__(self, cfg: AF3DiffusionModule.Config, kernel_config):
        AF3StyleDiffusionModule.__init__(self, cfg, kernel_config)
        self.diffusion_stack = ApoConditionedDiffusionStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            channel_atom=cfg.channel_atom,
            channel_atompair=cfg.channel_atompair,
            channel_coords=cfg.channel_coords,
            atom_encoder_blocks=cfg.atom_encoder_blocks,
            atom_encoder_heads=cfg.atom_encoder_heads,
            token_transformer_blocks=cfg.token_transformer_blocks,
            token_transformer_heads=cfg.token_transformer_heads,
            atom_decoder_blocks=cfg.atom_decoder_blocks,
            atom_decoder_heads=cfg.atom_decoder_heads,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )
        self.drop_rate: float = cfg.conditioning_drop_rate
        assert 0.0 <= self.drop_rate < 1.0, "Conditioning drop rate must be in [0, 1)."
