# started from code from https://github.com/jwohlwend/boltz, MIT License
import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.alphafold3.diffusion import DiffusionModule
from kfold.utils.registry import SCORE_MODEL, BaseConfig

from .base import BaseScoreModel


@SCORE_MODEL.register()
class ECSIDiffusionModule(BaseScoreModel):
    """Diffusion score model with apo structure conditioning."""

    class Config(BaseConfig):
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
        use_apo : bool
            Whether to use apo structure conditioning, by default True.
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
        use_prior_coords: bool = True
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

    def __init__(self, cfg: Config, kernel_config):
        super().__init__(cfg, kernel_config)

        diffusion_stack_class = DiffusionModule
        # NOTE:
        # - If use_prior_coords=True, score model expects r_noisy[..., 6]
        #   (x_t concat x_apo).
        # - If use_prior_coords=False, score model expects r_noisy[..., 3]
        #   (x_t only).
        effective_channel_coords = 6 if cfg.use_prior_coords else 3

        self.diffusion_stack = diffusion_stack_class(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            channel_atom=cfg.channel_atom,
            channel_atompair=cfg.channel_atompair,
            channel_coords=effective_channel_coords,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            atoms_per_window_keys=cfg.atoms_per_window_keys,
            dim_fourier=cfg.dim_fourier,
            atom_encoder_blocks=cfg.atom_encoder_blocks,
            atom_encoder_heads=cfg.atom_encoder_heads,
            token_transformer_blocks=cfg.token_transformer_blocks,
            token_transformer_heads=cfg.token_transformer_heads,
            atom_decoder_blocks=cfg.atom_decoder_blocks,
            atom_decoder_heads=cfg.atom_decoder_heads,
            conditioning_transition_layers=cfg.conditioning_transition_layers,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )

    def do_compile(self, mode: str = "default"):
        """Compile the trunk module."""
        self.diffusion_stack = torch.compile(
            self.diffusion_stack,
            mode="default",  # reduce-overhead mode has issues on DDP.
            dynamic=False,
            fullgraph=False,
        )  # type: ignore

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
        """Forward pass of the apo-conditioned diffusion score model.

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
        if self.training:
            assert model_cache is None, "model_cache is only used during evaluation."

        # Revert to uncompiled version for validation
        diffusion_stack: DiffusionModule
        if self.is_compiled and not self.training:
            diffusion_stack = self.diffusion_stack._orig_mod  # noqa: SLF001
        else:
            diffusion_stack = self.diffusion_stack

        return diffusion_stack(
            r_noisy,
            c_noise,
            f_input,
            s_inputs,
            s_trunk,
            z_trunk,
            model_cache,
            use_cuequiv_kernels=self.kernel_config.cuequivariance,
        )
