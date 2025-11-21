# started from code from https://github.com/jwohlwend/boltz, MIT License
import torch

from kfold.data.model_input import FoldingInput
from kfold.model.layers.boltz1.diffusion import DiffusionModule
from kfold.model.layers.boltz1.encoders import RelativePositionEncoder
from kfold.utils.registry import SCORE_MODEL, BaseConfig

from .base import BaseScoreModel


@SCORE_MODEL.register()
class Boltz1DiffusionModule(BaseScoreModel):
    """Boltz1 Diffusion module"""

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
        atoms_per_window_queries : int, optional
            The number of atoms per window for queries, by default 32.
        atoms_per_window_keys : int, optional
            The number of atoms per window for keys, by default 128.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.
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
        conditioning_transition_layers : int, optional
            The number of transition layers for conditioning, by default 2.
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint, by default None.
        """

        channel_s: int = 384
        channel_z: int = 128
        channel_atom: int = 128
        channel_atompair: int = 16
        atoms_per_window_queries: int = 32
        atoms_per_window_keys: int = 128
        dim_fourier: int = 256
        atom_encoder_blocks: int = 3
        atom_encoder_heads: int = 4
        token_transformer_blocks: int = 24
        token_transformer_heads: int = 8
        atom_decoder_blocks: int = 3
        atom_decoder_heads: int = 4
        conditioning_transition_layers: int = 2
        blocks_per_ckpt: int | None = None

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

        self.diffusion_stack = DiffusionModule(
            token_s=cfg.channel_s,
            token_z=cfg.channel_z,
            atom_s=cfg.channel_atom,
            atom_z=cfg.channel_atompair,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            atoms_per_window_keys=cfg.atoms_per_window_keys,
            dim_fourier=cfg.dim_fourier,
            atom_encoder_depth=cfg.atom_encoder_blocks,
            atom_encoder_heads=cfg.atom_encoder_heads,
            token_transformer_depth=cfg.token_transformer_blocks,
            token_transformer_heads=cfg.token_transformer_heads,
            atom_decoder_depth=cfg.atom_decoder_blocks,
            atom_decoder_heads=cfg.atom_decoder_heads,
            atom_feature_dim=389,
            conditioning_transition_layers=cfg.conditioning_transition_layers,
        )

        self.rel_pos_encoding = RelativePositionEncoder(cfg.channel_z)

    def forward(
        self,
        r_noisy: torch.Tensor,
        c_noise: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        model_cache: dict | None = None,
    ) -> torch.Tensor:
        """Forward pass of the AF3 diffusion module.
        See Section 3.7 Algorithm 20: Diffusion Module in the AF3 paper.

        Parameters
        ----------
        r_noisy : torch.Tensor
            The noisy atom positions, shape [B, N, La, 3],
            where N is number of diffusion samples and La is number of atoms.
        c_noise : torch.Tensor
            The diffusion noise level (or sigmas), shape [B, N].
            c_noise = 1/4 log(t_hat / sigma_data) (See Algorithm 21.)
            c_noise is computed outside of this class (See StructureModule).
        f_input : FoldingInput
            The folding input.
        s_inputs : torch.Tensor
            The input single representation, shape [B, Lt, c_s].
        s_trunk : torch.Tensor
            The trunk single representation, shape [B, Lt, c_s].
        z_trunk : torch.Tensor
            The trunk pair representation, shape [B, Lt, c_z].

        Returns
        -------
        r_update : torch.Tensor
            The denoised atom positions, shape [B, N, La, 3].
        """
        B, N, La, _ = r_noisy.shape
        r_noisy = r_noisy.view(B * N, La, 3)
        c_noise = c_noise.view(B * N)
        # s_inputs = s_inputs
        # s_trunk = s_trunk.repeat_interleave(N, dim=0)
        # z_trunk = z_trunk.repeat_interleave(N, dim=0)

        rel_pos_encoding = self.rel_pos_encoding(f_input)

        r_update = self.diffusion_stack(
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            r_noisy=r_noisy,
            times=c_noise,
            relative_position_encoding=rel_pos_encoding,
            multiplicity=N,
            f_input=f_input,
            model_cache=model_cache,
        )["r_update"]
        return r_update.view(B, N, La, 3)
