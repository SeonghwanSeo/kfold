# started from code from https://github.com/jwohlwend/boltz, MIT License

import torch

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.diffusion import DiffusionModule
from kfold.utils.registry import SCORE_MODEL, BaseConfig

from .base import BaseScoreModel


@SCORE_MODEL.register()
class AF3DiffusionModule(BaseScoreModel):
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
        atoms_per_window_queries : int, optional
            The number of atoms per window for queries, by default 32.
        atoms_per_window_keys : int, optional
            The number of atoms per window for keys, by default 128.
        sigma_data : int, optional
            The standard deviation of the data distribution, by default 16.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.
        atom_encoder_depth : int, optional
            The depth of the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_depth : int, optional
            The depth of the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 8.
        atom_decoder_depth : int, optional
            The depth of the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        conditioning_transition_layers : int, optional
            The number of transition layers for conditioning, by default 2.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.
        offload_to_cpu : bool, optional
            Whether to offload the activations to CPU, by default False.

        """

        channel_s: int = 384
        channel_z: int = 128
        channel_atom: int = 128
        channel_atompair: int = 16
        atoms_per_window_queries: int = 32
        atoms_per_window_keys: int = 128
        sigma_data: int = 16
        dim_fourier: int = 256
        atom_encoder_depth: int = 3
        atom_encoder_heads: int = 4
        token_transformer_depth: int = 24
        token_transformer_heads: int = 8
        atom_decoder_depth: int = 3
        atom_decoder_heads: int = 4
        conditioning_transition_layers: int = 2
        activation_checkpointing: bool = False
        offload_to_cpu: bool = False

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

        self.diffusion_stack = DiffusionModule(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            channel_atom=cfg.channel_atom,
            channel_atompair=cfg.channel_atompair,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            atoms_per_window_keys=cfg.atoms_per_window_keys,
            sigma_data=cfg.sigma_data,
            dim_fourier=cfg.dim_fourier,
            atom_encoder_depth=cfg.atom_encoder_depth,
            atom_encoder_heads=cfg.atom_encoder_heads,
            token_transformer_depth=cfg.token_transformer_depth,
            token_transformer_heads=cfg.token_transformer_heads,
            atom_decoder_depth=cfg.atom_decoder_depth,
            atom_decoder_heads=cfg.atom_decoder_heads,
            conditioning_transition_layers=cfg.conditioning_transition_layers,
            activation_checkpointing=cfg.activation_checkpointing,
            offload_to_cpu=cfg.offload_to_cpu,
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        times: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        model_cache=None,
    ) -> torch.Tensor:
        """Forward pass of the AF3 diffusion module.
        See Section 3.7 Algorithm 20: Diffusion Module in the AF3 paper.

        Parameters
        ----------
        x_noisy : torch.Tensor
            The noisy atom positions, shape [Nsample, La, 3],
            where Nsample is the number of diffusion samples.
        times : torch.Tensor
            The diffusion times, shape [Nsample].
        f_input : FoldingInput
            The folding input.
        s_inputs : torch.Tensor
            The input single representation, shape [Lt, c_s].
        s_trunk : torch.Tensor
            The trunk single representation, shape [Lt, c_s].
        z_trunk : torch.Tensor
            The trunk pair representation, shape [Lt, c_z].
        """
        x_denoised = self.diffusion_stack(
            x_noisy,
            times,
            f_input,
            s_inputs,
            s_trunk,
            z_trunk,
            model_cache=model_cache,
        )
        return x_denoised
