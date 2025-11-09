"""Section 3.7 Diffusion Module in the AF3 paper."""
# started from code from https://github.com/jwohlwend/boltz, MIT License

import math

import torch
import torch.nn as nn

from kfold.data.model_input import FoldingInput

from . import initialize as init
from .embeddings import RelativePositionEncoding
from .primitives import LinearNoBias, Transition
from .transformers import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)


class FourierEmbedding(nn.Module):
    """Fourier embedding layer.
    Section 3.7 Algorithm 22 Fourier Embedding
    """

    def __init__(self, channel: int, seed: int = 42):
        """Initialize the Fourier Embeddings.

        Parameters
        ----------
        channel : int
            The fourier embedding dimension.
        seed : int, optional
            The random seed, by default 42

        """
        super().__init__()

        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Line 1: Randomly generate weight/bias once before training
        w = torch.randn(size=(1, channel), generator=generator)
        b = torch.randn(size=(1, channel), generator=generator)
        self.w = nn.Parameter(w, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        See Section 3.7 Algorithm 22 of AlphaFold3 paper.

        Parameters
        ----------
        times : torch.Tensor
            The input times. Shape (B, N,)

        Returns
        -------
        torch.Tensor
            The Fourier embeddings. Shape (B, N, channel)
        """
        # Line 2
        return torch.cos((2 * math.pi) * times[..., None] * self.w + self.b)


class DiffusionModule(nn.Module):
    """AF3 Diffusion module
    Section 3.7 Algorithm 20: Diffusion Module in the AF3 paper.
    """

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        channel_atom: int = 128,
        channel_atompair: int = 16,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_blocks: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_blocks: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_blocks: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
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
        atoms_per_window_queries : int, optional
            The number of atoms per window for queries, by default 32.
        atoms_per_window_keys : int, optional
            The number of atoms per window for keys, by default 128.
        sigma_data : int, optional
            The standard deviation of the data distribution, by default 16.
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
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.
        offload_to_cpu : bool, optional
            Whether to offload the activations to CPU, by default False.

        """
        super().__init__()

        self.atoms_per_window_queries: int = atoms_per_window_queries
        self.atoms_per_window_keys: int = atoms_per_window_keys
        self.sigma_data: int = sigma_data

        channel_token = channel_s * 2

        # === Diffusion conditioning === #
        self.diffusion_conditioning = DiffusionConditioning(
            channel_s=channel_s,
            channel_z=channel_z,
            sigma_data=sigma_data,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        # === Local atom-level attention encoder === #
        self.atom_attention_encoder = AtomAttentionEncoder(
            channel_s=channel_s,
            channel_z=channel_z,
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            channel_token=channel_token,
            num_blocks=atom_encoder_blocks,
            num_heads=atom_encoder_heads,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            activation_checkpointing=activation_checkpointing,
            use_structure=True,
        )

        # === Full token-level attention === #
        self.layernorm_s = nn.LayerNorm(channel_token)
        self.trans_s_to_a = LinearNoBias(channel_token, channel_token)
        init.final_init_(self.trans_s_to_a.weight)

        self.token_transformer = DiffusionTransformer(
            channel_a=channel_token,
            channel_s=channel_s,
            channel_z=channel_z,
            num_blocks=token_transformer_blocks,
            num_heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

        self.layernorm_a = nn.LayerNorm(2 * channel_s)

        # === Local token-level attention decoder === #
        self.atom_attention_decoder = AtomAttentionDecoder(
            channel_a=channel_token,
            channel_s=channel_s,
            channel_atom=channel_atom,
            channel_atompair=channel_atompair,
            num_blocks=atom_decoder_blocks,
            num_heads=atom_decoder_heads,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            activation_checkpointing=activation_checkpointing,
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
            The noisy atom positions, shape [B, N, La, 3],
            where B is the batch size and Nsample is the number of diffusion samples.
        times : torch.Tensor
            The diffusion times, shape [B, N].
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
        x_update : torch.Tensor
            The updated atom positions, shape [B, N, La, 3].
        """

        # B: batch size, N: number of diffusion samples
        B, N = x_noisy.shape[:2]  # noqa

        # Line 1
        s, z = self.diffusion_conditioning(
            f_input=f_input,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            times=times,
            model_cache=model_cache,
        )  # [B, Lt, Cs], [B, Lt, Lt, Cz]

        # Scale positions
        c_in = 1 / torch.sqrt(times**2 + self.sigma_data**2)  # [B, N]
        r_noisy = x_noisy * c_in[:, None, None]  # [B, N, La, 3]

        # Compute Atom Attention Encoder and aggregation to coarse-grained tokens
        # Shape:
        # - a: [B, N, Lt, 2*Cs]
        # - q_skip: [B, N, La, Ca]
        # - c_skip: [B, N, La, Ca]
        # - p_skip: [B, N, La, La, Cap]

        a, q_skip, c_skip, p_skip, local_attn_indexer = self.atom_attention_encoder(
            f_input=f_input,
            r=r_noisy,  # [B, N, La, 3]
            s_trunk=s_trunk,  # [B, Lt, Cs]
            z=z,  # [B, Lt, Lt, Cz]
            model_cache=model_cache,
        )

        # Full self-attention on token level
        a = a + self.trans_s_to_a(self.layernorm_s(s))  # [Nsample, La, Cs]

        mask = f_input.token.pad_mask.float()  # [B, Lt]
        a = self.token_transformer(
            a,  # [B, N, Lt, Cs]
            s=s,  # [B, N, Lt, Cs]
            z=z,  # [B, Lt, Lt, Cz]
            attn_mask=mask[:, None, None],  # [B, 1, 1, Lt], broadcasted to [B, N, Lt, Lt]
            model_cache=model_cache,
        )
        a = self.layernorm_a(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            p=p_skip,
            f_input=f_input,
            local_attn_indexer=local_attn_indexer,
            model_cache=model_cache,
        )

        # Rescale positions and update
        # NOTE: I simply use AF3 formula instead of Boltz1's
        # TODO: add hparams to control.
        c_skip = (self.sigma_data**2) / (self.sigma_data**2 + times**2)
        c_out = self.sigma_data * times / torch.sqrt(self.sigma_data**2 + times**2)
        x_update = c_skip.view(B, N, 1, 1) * x_noisy + c_out.view(B, N, 1, 1) * r_update

        return x_update


class DiffusionConditioning(nn.Module):
    """Diffusion conditioning layer.

    NOTE(seonghwanseo):
    According to AlphaFold3 paper, s is time-dependent single representation,
    while z is time-independent pair representation.

    For model efficiency, I implement this function to return batched s and
    single z, where the batch size is the number of diffusion samples.
    Input:
        times: [B, N] - diffusion times
        s_inputs: [B, Lt, c_s] - input single representation
        s_trunk: [B, Lt, c_s] - trunk single representation
        z_trunk: [B, Lt, Lt, c_z] - trunk pair representation
    Output:
        s: [B, N, Lt, c_s] - time-dependent single conditioning
        z: [B, Lt, Lt, c_z] - time-independent pair conditioning

    Due to this reason, in Boltz2, they remove the time term from conditioning,
    i.e., return time-independent s and z. (see Boltz2's diffusion_conditioning.py)
    """

    def __init__(
        self,
        channel_s: int = 384,
        channel_z: int = 128,
        sigma_data: float = 16.0,
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
        sigma_data : float
            The data sigma.
        dim_fourier : int
            The fourier embeddings dimension, by default 256.
        num_transitions : int
            The number of transitions layers, by default 2.
        transition_expansion_factor : int
            The transition expansion factor, by default 2.
        """
        super().__init__()
        self.sigma_data = sigma_data

        # Pair representation conditioning
        self.rel_pos_encoding = RelativePositionEncoding(channel_z=channel_z)
        self.layernorm_pair = nn.LayerNorm(channel_z * 2)
        self.linear_no_bias_pair = LinearNoBias(channel_z * 2, channel_z)

        self.transitions_pair = nn.ModuleList(
            [
                Transition(channel_z, expansion_factor=transition_expansion_factor)
                for _ in range(num_transitions)
            ]
        )

        # Single representation conditioning
        self.layernorm_single = nn.LayerNorm(channel_s * 2)
        self.linear_no_bias_single = LinearNoBias(channel_s * 2, channel_s)

        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.layernorm_fourier = nn.LayerNorm(dim_fourier)
        self.linear_no_bias_fourier = LinearNoBias(dim_fourier, channel_s)

        self.transitions_single = nn.ModuleList(
            [
                Transition(channel_s, expansion_factor=transition_expansion_factor)
                for _ in range(num_transitions)
            ]
        )

    def forward(
        self,
        times: torch.Tensor,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        model_cache: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """See Section 3.7 Algorithm 21 Diffusion Conditioning in the AF3 paper.

        Parameters
        ----------
        times : torch.Tensor
            Tensor of shape (B, N) containing diffusion times.
        f_input : FoldingInput
            The folding input.
        s_inputs : torch.Tensor
            Tensor of shape (B, Lt, c_s) containing input single embeddings.
        s_trunk : torch.Tensors
            Tensor of shape (B, Lt, c_s) containing trunk single embeddings.
        z_trunk : torch.Tensor
            Tensor of shape (B, Lt, Lt, c_z) containing trunk pair embeddings.
        model_cache : dict | None
            The model cache for storing intermediate representations, by default None.

        Returns
        -------
        s : torch.Tensor
            Tensor of shape (B, N, Lt, c_s) containing conditioned single embeddings.
        z : torch.Tensor
            Tensor of shape (B, Lt, Lt, c_z) containing conditioned pair embeddings.
        """

        if model_cache is not None:
            cache_prefix = "diffusion_conditioning"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]
        else:
            layer_cache = {}

        if "z" not in layer_cache:
            # For time-independent pair representation z, we cache the result
            # Line 1
            rel_pos_feats = self.rel_pos_encoding(
                f_input, model_cache
            )  # [B, Lt, Lt, c_z]
            z = torch.cat((z_trunk, rel_pos_feats), dim=-1)

            # Line 2
            z = self.linear_no_bias_pair(self.layernorm_pair(z))  # [B, Lt, Lt, c_z]

            # Line 3-5
            for transition in self.transitions_pair:
                z = z + transition(z)
            layer_cache["z"] = z
        else:
            z = layer_cache["z"]

        # Line 6
        s = torch.cat((s_trunk, s_inputs), dim=-1)  # [B, Lt, 2*c_s]

        # Line 7
        s = self.linear_no_bias_single(self.layernorm_single(s))  # [B, Lt, c_s]

        # Line 8
        c_noise = (times / self.sigma_data).clamp(1e-20).log() * 0.25
        fourier_embed = self.fourier_embed(c_noise)  # [B, N, d_fourier]

        # Line 9
        fourier_embed = self.linear_no_bias_fourier(self.layernorm_fourier(fourier_embed))
        s = s[:, None, :, :] + fourier_embed[:, :, None, :]  # [B, N, Lt, c_s]

        # Line 10-12
        for transition in self.transitions_single:
            s = transition(s) + s

        # Line 13
        return s, z
