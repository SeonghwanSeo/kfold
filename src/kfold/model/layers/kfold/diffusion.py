import torch
import torch.nn as nn

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.diffusion import (
    DiffusionConditioning,
)
from kfold.model.layers.alphafold3.transformers import (
    AtomAttentionDecoder,
    DiffusionTransformer,
)
from kfold.model.layers.primitives import LayerNorm, LinearNoBias

from .transformers import AtomAttentionEncoderWithApo


class DiffusionModuleWithApo(nn.Module):
    """Diffusion module with apo structure conditioning."""

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        channel_atom: int = 128,
        channel_atompair: int = 16,
        channel_coords: int = 3,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        dim_fourier: int = 256,
        atom_encoder_blocks: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_blocks: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_blocks: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
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
        atoms_per_window_queries : int, optional
            The number of atoms per window for queries, by default 32.
        atoms_per_window_keys : int, optional
            The number of atoms per window for keys, by default 128.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.
        atom_encoder_blocks : int, optional
            The number of blocks in the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_blocks : int, optional
            The number of blocks in the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 8.
        atom_decoder_blocks : int, optional
            The number of blocks in the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        conditioning_transition_layers : int, optional
            The number of transition layers for conditioning, by default 2.
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint for gradient checkpointing,
            by default None.

        """
        super().__init__()

        self.atoms_per_window_queries: int = atoms_per_window_queries
        self.atoms_per_window_keys: int = atoms_per_window_keys

        channel_token = channel_s * 2

        # === Diffusion conditioning === #
        self.diffusion_conditioning = DiffusionConditioning(
            channel_s=channel_s,
            channel_z=channel_z,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        # === Local atom-level attention encoder === #
        # NOTE: Encode local apo structure information
        self.atom_attention_encoder = AtomAttentionEncoderWithApo(
            channel_s=channel_s,
            channel_z=channel_z,
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            channel_token=channel_token,
            channel_coords=channel_coords,
            num_blocks=atom_encoder_blocks,
            num_heads=atom_encoder_heads,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            blocks_per_ckpt=blocks_per_ckpt,
            use_structure=True,
        )

        # === Full token-level attention === #
        # Projection from s to a
        self.layernorm_s = LayerNorm(channel_s, create_offset=False)
        self.linear_s_to_a = LinearNoBias(channel_s, channel_token, init="final")
        self.token_transformer = DiffusionTransformer(
            channel_a=channel_token,
            channel_s=channel_s,
            channel_z=channel_z,
            num_blocks=token_transformer_blocks,
            num_heads=token_transformer_heads,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_a = LayerNorm(channel_token, create_offset=False)

        # === Local token-level attention decoder === #
        self.atom_attention_decoder = AtomAttentionDecoder(
            channel_a=channel_token,
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            num_blocks=atom_decoder_blocks,
            num_heads=atom_decoder_heads,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )

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
        """Forward pass of diffusion score model.

        Parameters
        ----------
        r_noisy : torch.Tensor
            The scaled noisy atom positions, shape [B, N, La, 3],
            where B is the batch size and Nsample is the number of diffusion samples.
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
            The scaled updated atom positions, shape [B, N, La, 3].
        """
        # B: batch size, N: number of diffusion samples

        # === Diffusion conditioning === #
        s, z = self.diffusion_conditioning(
            c_noise=c_noise,
            f_input=f_input,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            model_cache=model_cache,
        )  # [B, N, Lt, c_s], [B, Lt, Lt, c_z]

        s_trunk = s_trunk.unsqueeze(-3)  # [B, 1, Lt, c_s]
        z = z.unsqueeze(-4)  # [B, 1, Lt, Lt, c_z]

        # Shape:
        # - s_trunk: [B, 1, Lt, c_s] (time-independent)
        # - s: [B, N, Lt, c_s] where N is number of diffusion samples
        # - z: [B, 1, Lt, Lt, c_z] (time-independent)

        # === Local attention on atom-level and aggregate to coarse-grained token === #
        a, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            f_input=f_input,
            r_noisy=r_noisy,  # [B, N, La, 3]
            s_trunk=s_trunk,  # [B, 1, Lt, c_s], broadcasted to [B, N, Lt, c_s]
            z_trunk=z,  # [B, 1, Lt, Lt, c_z], broadcasted to [B, N, Lt, Lt, c_z]
            model_cache=model_cache,
        )
        # Shape:
        # - a: [B, N, Lt, c_token]
        # - q_skip: [B, N, La, c_atom]
        # - c_skip: [B, N, La, c_atom]
        # - p_skip: [B, N, Lq, Lk, c_atompair]

        # === Full attention on token-level === #
        a = a + self.linear_s_to_a(self.layernorm_s(s))  # [B, N, La, c_token]
        token_mask = f_input.token.pad_mask[..., None, :]  # [B, 1, Lt]
        a = self.token_transformer(
            a,  # [B, N, Lt, c_token]
            s=s,  # [B, N, Lt, c_s]
            z=z,  # [B, 1, Lt, Lt, c_z], broadcasted to [B, N, Lt, Lt, c_z]
            attn_mask=token_mask,  # [B, 1, Lt], broadcasted to [B, N, Lt]
        )
        a = self.layernorm_a(a)

        # === Local attention decoder to update atom positions === #
        r_update = self.atom_attention_decoder(
            a=a,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip,
            f_input=f_input,
        )
        return r_update
