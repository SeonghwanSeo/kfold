import torch
import torch.nn as nn

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.diffusion import (
    FourierEmbedding,
)
from kfold.model.layers.alphafold3.embeddings import RelativePositionEncoding
from kfold.model.layers.alphafold3.transformers import (
    AtomAttentionDecoder,
    DiffusionTransformer,
)
from kfold.model.layers.alphafold3.transition import Transition
from kfold.model.layers.primitives import LayerNorm, LinearNoBias

from .transformers import AtomAttentionEncoderWithApo


class DiffusionConditioningWithApo(nn.Module):
    """Diffusion conditioning layer with apo structure information."""

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        dim_fourier: int = 256,
        num_transitions: int = 2,
        transition_expansion_factor: int = 2,
        eps: float = 1e-20,
    ):
        """Initialize the single conditioning layer.

        Parameters
        ----------
        channel_s : int
            The single representation dimension, by default 384.
        channel_z : int
        The pair representation dimension, by default 128.
        dim_fourier : int
            The fourier embeddings dimension, by default 256.
        num_transitions : int
            The number of transitions layers, by default 2.
        transition_expansion_factor : int
            The transition expansion factor, by default 2.
        """
        super().__init__()

        # Pair representation conditioning
        self.rel_pos_encoding = RelativePositionEncoding()
        rel_pos_dim = self.rel_pos_encoding.dimension
        self.layernorm_z = LayerNorm(channel_z + rel_pos_dim + 1, create_offset=False)
        self.linear_z = LinearNoBias(
            channel_z + rel_pos_dim + 1, channel_z, init="default"
        )

        self.transitions_z = nn.ModuleList(
            [
                Transition(channel_z, expansion_factor=transition_expansion_factor)
                for _ in range(num_transitions)
            ]
        )

        # Single representation conditioning
        self.layernorm_s = LayerNorm(channel_s * 2, create_offset=False)
        self.linear_s = LinearNoBias(channel_s * 2, channel_s, init="default")

        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.layernorm_fourier = LayerNorm(dim_fourier, create_offset=False)
        self.linear_fourier = LinearNoBias(dim_fourier, channel_s, init="default")

        self.transitions_s = nn.ModuleList(
            [
                Transition(channel_s, expansion_factor=transition_expansion_factor)
                for _ in range(num_transitions)
            ]
        )

    def forward(
        self,
        c_noise: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        model_cache: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of diffusion conditioning.

        Parameters
        ----------
        c_noise : torch.Tensor
            Tensor of shape (B, N) containing diffusion noise level (or sigma).
            > c_noise = 1/4 log(t_hat / sigma_data) according to EDM
        f_input : FoldingInput
            The folding input.
        s_inputs : torch.Tensor
            Tensor of shape (B, Lt, c_s) for input single representation.
        s_trunk : torch.Tensors
            Tensor of shape (B, Lt, c_s) for trunk single representation.
        z_trunk : torch.Tensor
            Tensor of shape (B, Lt, Lt, c_z) for trunk pair representation.
        model_cache : dict | None
            The model cache for storing intermediate representations, by default None.

        Returns
        -------
        s : torch.Tensor
            Tensor of shape (B, N, Lt, c_s) for time-dependent single conditioning.
        z : torch.Tensor
            Tensor of shape (B, Lt, Lt, c_z) for time-independent pair conditioning.
        """

        if model_cache is not None:
            cache_prefix = "diffusion_conditioning"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]
        else:
            layer_cache = {}

        # For time-independent pair representation z, we cache the result
        if "z" in layer_cache:
            z = layer_cache["z"]
        else:
            z = self.compute_pair_conditioning(f_input, z_trunk)  # [B, Lt, Lt, c_z]
            layer_cache["z"] = z

        # Compute time-dependent single representation s
        s = self.compute_single_conditioning(
            c_noise, s_inputs, s_trunk
        )  # [B, N, Lt, c_s]

        return s, z

    def compute_pair_conditioning(
        self, f_input: FoldingInput, z_trunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute only the pair conditioning z.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        z_trunk : torch.Tensor
            Tensor of shape (B, Lt, Lt, c_z) containing trunk pair embeddings.

        Returns
        -------
        z : torch.Tensor
            Tensor of shape (B, Lt, Lt, c_z) containing conditioned pair embeddings.
        """
        # Add relative position encoding
        rel_pos_feats = self.rel_pos_encoding(f_input, z_trunk.dtype)  # [B, Lt, Lt, c_z]

        # Add apo distance encoding
        apo_dist = self.compute_apo_distance_map(f_input, p=-0.5)  # [B, Lt, Lt]
        apo_dist = apo_dist[..., None]  # [B, Lt, Lt, 1]

        z = torch.cat((z_trunk, rel_pos_feats, apo_dist), dim=-1)
        z = self.linear_z(self.layernorm_z(z))  # [B, Lt, Lt, c_z]

        for transition in self.transitions_z:
            z = z + transition(z)
        return z

    def compute_single_conditioning(
        self, c_noise: torch.Tensor, s_inputs: torch.Tensor, s_trunk: torch.Tensor
    ) -> torch.Tensor:
        s = torch.cat((s_trunk, s_inputs), dim=-1)  # [B, Lt, 2*c_s]
        s = self.linear_s(self.layernorm_s(s))  # [B, Lt, c_s]

        # NOTE: 1/4 log(t_hat / sigma_data) is computed outside of this class.
        # See StructureModule for more details.
        fourier_embed = self.fourier_embed(c_noise)  # [B, N, d_fourier]
        fourier_embed = self.linear_fourier(self.layernorm_fourier(fourier_embed))
        s = s[:, None, :, :] + fourier_embed[:, :, None, :]  # [B, N, Lt, c_s]

        for transition in self.transitions_s:
            s = transition(s) + s
        return s

    @staticmethod
    def compute_apo_distance_map(f_input: FoldingInput, p: float = -0.5) -> torch.Tensor:
        """Compute the apo distance map from the folding input.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        p : float
            The exponent for distance transformation, by default -0.5.

        Returns
        -------
        d : torch.Tensor
            The apo distance map, shape [B, Lt, Lt].
        """
        batch_index = torch.arange(f_input.batch_size, device=f_input.device)[:, None]
        center_index = f_input.token.center_index

        # Extract apo C-alpha coordinates and mask
        apo_coords = f_input.atom.apo_coords[batch_index, center_index]  # [B, L, 3]
        mask = f_input.atom.apo_mask[batch_index, center_index]  # [B, L]
        pair_mask = mask[:, :, None] & mask[:, None, :]

        # Chain identity mask (no inter-chain apo distances)
        asym_id = f_input.token.asym_id  # [B, L]
        chain_mask = asym_id[:, :, None] == asym_id[:, None, :]

        pair_mask = pair_mask & chain_mask

        # Compute distance features
        with torch.autocast("cuda", enabled=False):
            # NOTE: use d_inv instead of d_sq_inv(used for ref_pos in AF3) since
            # d_inv has better numerical stability for large distances.
            pdist = torch.cdist(apo_coords, apo_coords, p=2)  # [B, L, L]

            if p >= 0:
                d = torch.pow(pdist, p)
            else:
                d = 1.0 / (1 + torch.pow(pdist, -p))

        d = d * pair_mask
        return d


class DiffusionModuleWithApo(nn.Module):
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
        self.diffusion_conditioning = DiffusionConditioningWithApo(
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
        """Forward pass of diffusion score model

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
        # - p_skip: [B, N, La, La, c_atompair]

        # === Full attention on token-level === #
        a = a + self.linear_s_to_a(self.layernorm_s(s))  # [Nsample, La, c_token]
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
