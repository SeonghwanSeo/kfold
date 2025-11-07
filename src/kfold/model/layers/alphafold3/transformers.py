# Started from code from https://github.com/jwohlwend/boltz, MIT License

from collections.abc import Callable
from functools import lru_cache, partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from kfold.data.model_input import FoldingInput

from . import initialize as init
from .embeddings import AtomEmbedding
from .primitives import AdaLN, LinearNoBias
from .utils import expand_batch


# === Helper functions for local atom attention === #
@lru_cache(maxsize=2)
def get_indexing_matrix(K: int, W: int, H: int, device: torch.device) -> torch.Tensor:
    assert W % 2 == 0
    assert H % (W // 2) == 0

    h = H // (W // 2)
    assert h % 2 == 0

    arange = torch.arange(2 * K, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(min=0, max=h + 1)
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = F.one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()


def single_to_keys(
    single: torch.Tensor,
    indexing_matrix: torch.Tensor,
    W: int,
    H: int,
) -> torch.Tensor:
    """Convert single tensor to keys tensor using indexing matrix.
    [..., K, W, D] -> [..., K, H, D]
    """
    if single.ndim == 2:
        L, D = single.shape
        K = L // W
        single = single.view(2 * K, W // 2, D)
        return torch.einsum("j i d, j k -> k i d", single, indexing_matrix).reshape(
            K, H, D
        )
    elif single.ndim == 3:
        B, L, D = single.shape
        K = L // W
        single = single.view(B, 2 * K, W // 2, D)
        return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
            B, K, H, D
        )
    elif single.ndim == 4:
        B, N, L, D = single.shape
        K = L // W
        single = single.view(B, N, 2 * K, W // 2, D)
        return torch.einsum(
            "b n j i d, j k -> b n k i d", single, indexing_matrix
        ).reshape(B, N, K, H, D)
    else:
        raise ValueError("Invalid single tensor shape.")


class AttentionPairBias(nn.Module):
    """Attention pair bias layer.
    Section 3.7 Algorithm 24 Attention with Pair Bias
    """

    def __init__(
        self,
        channel_a: int,  # c_atom (atom-attn) or c_token (token-attn)
        channel_s: int,  # c_atom (atom-attn) or c_s (token-attn)
        channel_z: int,  # c_atompair (atom-attn) or c_z (token-attn)
        num_heads: int,
        use_s: bool = True,
        inf: float = 1e6,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_a : int
            The atom/token dimension.
        c_s : int
            The input single dimension.
        c_z : int
            The input pair dimension.
        num_heads : int
            The number of heads.
        use_s : bool
            whether s is None or not, stated in Algorithm 24 Line 1.
        inf : float, optional
            The inf value, by default 1e6
        """
        super().__init__()

        assert channel_a % num_heads == 0

        self.channel_a: int = channel_a
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z
        self.num_heads: int = num_heads
        self.head_dim: int = channel_a // num_heads
        self.inf: float = inf

        self.use_s: bool = use_s
        if self.use_s:
            assert self.channel_s > 0, "channel_s must be positive if use_s is True"
            self.adaln = AdaLN(channel_a, channel_s)  # Defined below (Algorithm 26)
        else:
            assert self.channel_s == 0, "channel_s must be 0 if use_s is False"
            self.norm_a = nn.LayerNorm(channel_a)

        self.proj_q = nn.Linear(channel_a, channel_a)
        self.proj_k = LinearNoBias(channel_a, channel_a)
        self.proj_v = LinearNoBias(channel_a, channel_a)
        self.proj_g = LinearNoBias(channel_a, channel_a)

        self.proj_z = nn.Sequential(
            nn.LayerNorm(channel_z),
            LinearNoBias(channel_z, num_heads),
            Rearrange("b ... h -> b h ..."),
        )

        self.proj_out = LinearNoBias(channel_a, channel_a)
        init.final_init_(self.proj_out.weight)

        if self.use_s:
            self.linear_s = nn.Linear(channel_s, channel_a)
            nn.init.zeros_(self.linear_s.weight)
            nn.init.constant_(self.linear_s.bias, -2.0)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor | None,
        z: torch.Tensor,
        mask: torch.Tensor,
        to_keys=None,
        model_cache=None,
        use_kernels: bool = True,
    ) -> torch.Tensor:
        """Forward pass.
        See Section 3.7 Algorithm 24 of AlphaFold3 paper.

        Parameters
        ----------
        a : torch.Tensor
            The input atom/token tensor (B, N, c_a)
        s : torch.Tensor | None
            The input single tensor (B, N, c_s), can be None if use_s is False
        z : torch.Tensor
            The input pairwise tensor (B, N, N, c_z) or (1, N, N, c_z)
        mask : torch.Tensor
            The pairwise mask tensor (B, N) or (1, N)
        to_keys : Callable, optional
            A function to transform s to keys, by default None

        Returns
        -------
        a : torch.Tensor
            The output sequence tensor. (B, N, c_a)

        """
        B = a.shape[0]

        # === Input projection === #
        if self.use_s:
            # Line 1-2
            assert s is not None, "s cannot be None if use_s is True"
            a = self.adaln(a, s)
        else:
            # Line 3-4
            assert s is None, "s must be None if use_s is False"
            a = self.norm_a(a)

        if to_keys is not None:
            q_in = a
            k_in = to_keys(a)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            q_in = a
            k_in = a

        # Line 6
        q = self.proj_q(q_in).view(B, -1, self.num_heads, self.head_dim)

        # Line 7
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        # Line 8
        # Caching attention bias during diffusion roll-out
        if model_cache is None or "attn_bias" not in model_cache:
            attn_bias = self.proj_z(z)
            # The pairwise mask (B, N) is broadcasted to (B, 1, 1, N) and (B, H, N, N)
            attn_bias = attn_bias.masked_fill(~mask[:, None, None, :], -self.inf)

            if model_cache is not None:
                model_cache["attn_bias"] = attn_bias
        else:
            attn_bias = model_cache["attn_bias"]

        # Line 9
        g = self.proj_g(a).sigmoid()

        # === Attention === #
        # Line 10-11
        if use_kernels:
            # Use torch scaled_dot_product_attention for efficiency
            Av = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_bias,
                scale=None,
            )
            Av = Av.reshape(B, -1, self.channel_s)
        else:
            with torch.autocast("cuda", enabled=False):
                # Compute attention weights
                attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
                # Add attention bias
                attn = attn / (self.head_dim**0.5) + attn_bias
                attn = attn.softmax(dim=-1)

                # Compute output
                Av = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
                Av = Av.reshape(B, -1, self.channel_s)

        # Line 11
        a = self.proj_out(g * a)

        # === Output projection === #
        # Line 12-14
        if self.use_s:
            assert s is not None, "s cannot be None if use_s is True"
            a = torch.sigmoid(self.linear_s(s)) * a
        return a


class DiffusionTransformer(nn.Module):
    """Diffusion Transformer Stack
    Section 3.7 Algorithm 23 Diffusion Transformer [Line 1,4]
    """

    def __init__(
        self,
        channel_a: int,  # c_atom (atom-attn) or c_token (token-attn)
        channel_s: int,  # c_atom (atom-attn) or c_s (token-attn)
        channel_z: int,  # c_atompair (atom-attn) or c_z (token-attn)
        num_blocks: int,
        num_heads: int,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
    ):
        """Initialize the diffusion transformer.

        Parameters
        ----------
        channel_a : int
            The atom/token dimension.
        channel_s : int
            The single dimension.
        channel_z : int
            The pairwise dimension.
        num_blocks : int
            The number of blocks.
        num_heads : int
            The number of heads.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False
        offload_to_cpu : bool, optional
            Whether to offload to CPU, by default False

        """
        super().__init__()
        self.activation_checkpointing = activation_checkpointing

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if activation_checkpointing:
                self.blocks.append(
                    checkpoint_wrapper(
                        DiffusionTransformerBlock(
                            channel_a,
                            channel_s,
                            channel_z,
                            num_heads,
                        ),
                        offload_to_cpu=offload_to_cpu,
                    )
                )
            else:
                self.blocks.append(
                    DiffusionTransformerBlock(
                        channel_a,
                        channel_s,
                        channel_z,
                        num_heads,
                    )
                )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None,
        to_keys: Callable | None = None,
        model_cache=None,
    ):
        """See Section 3.7 Algorithm 23 Diffusion Transformer"""
        # Line 1, 4
        for i, block in enumerate(self.blocks):
            if model_cache is not None:
                prefix_cache = "layer_" + str(i)
                if prefix_cache not in model_cache:
                    model_cache[prefix_cache] = {}
                block_cache = model_cache[prefix_cache]
            else:
                block_cache = None
            a = block(
                a,
                s,
                z,
                mask=mask,
                to_keys=to_keys,
                layer_cache=block_cache,
            )
        return a


class DiffusionTransformerBlock(nn.Module):
    """Diffusion Transformer Block
    Section 3.7 Algorithm 23 Diffusion Transformer [Line 2-3]
    """

    def __init__(
        self,
        channel_a: int,  # c_atom (atom-attn) or c_token (token-attn)
        channel_s: int,  # c_atom (atom-attn) or c_s (token-attn)
        channel_z: int,  # c_atompair (atom-attn) or c_z (token-attn)
        num_heads: int,
    ):
        """Initialize the diffusion transformer block.

        Parameters
        ----------
        channel_a : int
            The atom/token dimension.
        channel_s : int
            The single dimension.
        channel_z : int
            The pairwise dimension.
        heads : int
            The number of heads.

        """
        super().__init__()
        self.pair_bias_attn = AttentionPairBias(
            channel_a, channel_s, channel_z, num_heads, use_s=True
        )

        self.transition = ConditionedTransitionBlock(
            channel_a, channel_s, expansion_factor=2
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        to_keys: Callable | None = None,
        block_cache: dict | None = None,
    ) -> torch.Tensor:
        """See Section 3.7 Algorithm 23 Diffusion Transformer"""
        # Line 2
        b = self.pair_bias_attn(
            a=a,
            s=s,
            z=z,
            mask=mask,
            to_keys=to_keys,
            model_cache=block_cache,
        )
        # Line 3
        a = a + self.transition(b, s)
        return a


class ConditionedTransitionBlock(nn.Module):
    """Conditioned Transition Block
    Section 3.7 Algorithm 25 Conditioned Transition Block
    """

    def __init__(self, channel_a: int, channel_s: int, expansion_factor: int = 2):
        """Initialize the conditioned transition block.

        Parameters
        ----------
        channel_a : int
            The atom/token dimension.
        channel_s : int
            The single conditioning dimension.
        expansion_factor : int, optional
            The expansion factor, by default 2

        """
        super().__init__()

        self.adaln = AdaLN(channel_a, channel_s)

        model_dim = int(channel_a * expansion_factor)  # Line 2
        self.linear_no_bias_a1 = LinearNoBias(channel_a, model_dim)
        self.linear_no_bias_a2 = LinearNoBias(channel_a, model_dim)
        self.linear_no_bias_b = LinearNoBias(model_dim, channel_a)

        self.linear_s = nn.Linear(channel_s, channel_a)
        nn.init.zeros_(self.linear_s.weight)
        nn.init.constant_(self.linear_s.bias, -2.0)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """See Section 3.7 Algorithm 25 Conditioned Transition Block"""
        # Line 1
        a = self.adaln(a, s)

        # Line 2
        b = F.silu(self.linear_no_bias_a1(a)) * self.linear_no_bias_a2(a)

        # Line 3
        a = torch.sigmoid(self.linear_s(s)) * self.linear_no_bias_b(b)
        return a


class AtomTransformer(nn.Module):
    """Atom Transformer
    See Section 3.2 Algorithm 7 Atom Transformer
    """

    def __init__(
        self,
        channel_a: int,  # c_atom (atom-attn) or c_token (token-attn)
        channel_s: int,  # c_atom (atom-attn) or c_s (token-attn)
        channel_z: int,  # c_atompair (atom-attn) or c_z (token-attn)
        num_blocks: int,
        num_heads: int,
        attn_window_queries: int,
        attn_window_keys: int,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
    ):
        """Initialize the atom transformer.

        Parameters
        ----------
        attn_window_queries : int
            The attention window queries, by default None
        attn_window_keys : int
            The attention window keys, by default None
        diffusion_transformer_kwargs : dict
            The diffusion transformer keyword arguments

        """
        super().__init__()
        self.attn_window_queries: int = attn_window_queries
        self.attn_window_keys: int = attn_window_keys
        self.diffusion_transformer = DiffusionTransformer(
            channel_a=channel_a,
            channel_s=channel_s,
            channel_z=channel_z,
            num_blocks=num_blocks,
            num_heads=num_heads,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        mask: torch.Tensor,
        to_keys,
        model_cache=None,
    ) -> torch.Tensor:
        """See Section 3.2 Algorithm 7 Atom Transformer"""

        W: int = self.attn_window_queries
        H: int = self.attn_window_keys

        B, N, D = q.shape
        NW = N // W

        # reshape tokens
        q = q.view((B * NW, W, -1))
        c = c.view((B * NW, W, -1))
        p = p.view((B * NW, W, H, -1))
        mask = mask.view(B * NW, W)

        to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)  # noqa

        # main transformer
        q = self.diffusion_transformer(
            a=q,
            s=c,
            z=p,
            mask=mask.float(),
            to_keys=to_keys_new,
            model_cache=model_cache,
        )

        if W is not None:
            q = q.view((B, NW * W, D))

        return q


class AtomAttentionEncoder(nn.Module):
    """Atom attention encoder.
    Section 3.2 Algorithm 6 Atom Attention Encoder
    """

    def __init__(
        self,
        channel_s: int,  # 384 in AF3
        channel_z: int,  # 128 in AF3
        channel_atom: int,  # 128 in AF3
        channel_atompair: int,  # 16 in AF3
        channel_token: int,  # 384 (InputEmbedder) or 768 (Diffusion) in AF3
        num_blocks=3,
        num_heads=4,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        use_structure: bool = True,
        activation_checkpointing=False,
    ):
        """Initialize the atom attention encoder.

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
        channel_token: int
            The token single representation dimension.
        num_blocks : int, optional
            The number of transformer blocks, by default 3.
        num_heads : int, optional
            The number of transformer heads, by default 4.
        atoms_per_window_queries : int
            The number of atoms per window for queries.
        atoms_per_window_keys : int
            The number of atoms per window for keys.
        use_structure : bool, optional
            Whether to use structure information, by default True.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.

        """
        super().__init__()
        self.atoms_per_window_queries: int = atoms_per_window_queries
        self.atoms_per_window_keys: int = atoms_per_window_keys

        self.embed_atom = AtomEmbedding(channel_atom)
        self.embed_atompair_ref_pos = LinearNoBias(3, channel_atompair)
        self.embed_atompair_ref_dist = LinearNoBias(1, channel_atompair)
        self.embed_atompair_mask = LinearNoBias(1, channel_atompair)

        self.use_structure = use_structure
        if use_structure:
            self.s_to_c_trans = nn.Sequential(
                nn.LayerNorm(channel_s), LinearNoBias(channel_s, channel_atom)
            )
            init.final_init_(self.s_to_c_trans[1].weight)

            self.z_to_p_trans = nn.Sequential(
                nn.LayerNorm(channel_z), LinearNoBias(channel_z, channel_atompair)
            )
            init.final_init_(self.z_to_p_trans[1].weight)

            self.r_to_q_trans = LinearNoBias(3, channel_atom)
            init.final_init_(self.r_to_q_trans.weight)

        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(channel_atom, channel_atompair),
        )
        init.final_init_(self.c_to_p_trans_k[1].weight)

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(channel_atom, channel_atompair),
        )
        init.final_init_(self.c_to_p_trans_q[1].weight)

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(channel_atompair, channel_atompair),
            nn.ReLU(),
            LinearNoBias(channel_atompair, channel_atompair),
            nn.ReLU(),
            LinearNoBias(channel_atompair, channel_atompair),
        )
        init.final_init_(self.p_mlp[5].weight)

        self.atom_encoder = AtomTransformer(
            channel_a=channel_atom,
            channel_s=channel_atom,
            channel_z=channel_atompair,
            num_blocks=num_blocks,
            num_heads=num_heads,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            activation_checkpointing=activation_checkpointing,
        )

        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(channel_atom, channel_token),
            nn.ReLU(),
        )

    def forward(
        self,
        f_input: FoldingInput,
        s_trunk: torch.Tensor | None,
        z: torch.Tensor | None,
        r: torch.Tensor | None,
        model_cache: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Callable]:
        """Forward pass of the atom attention encoder.
        See Algorithm 5 in the AF3 paper for more details.

        q: Atom single representation
        c: Atom single conditioning
        p: Atom pair representation

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        s_trunk : torch.Tensor | None
            The trunk single representation, shape [Nt, c_s].
        z : torch.Tensor | None
            The conditioning pair representation, shape [Nsample, Nt, c_z].
        r : torch.Tensor | None
            The noised structures' positions, shape [Nsample, Na, c_r],
            where Nsample is the number of diffusion samples.
        model_cache : dict | None
            The model cache for storing intermediate representations, by default None.

        Returns
        -------
        a : torch.Tensor
            The token single representation, shape [Nsample, Nt, c_token].
        q_skip : torch.Tensor
            The atom single representation, shape [Nsample, Na, c_atom].
        c_skip : torch.Tensor
            The atom single conditioning, shape [Nsample, Na, c_atom].
        p_skip : torch.Tensor
            The atom pair representation, shape [Nsample, K, W, H, c_atompair].
        to_keys : Callable
            The function to convert single representation to keys representation.
        """
        atom_mask = f_input.atom.pad_mask

        if model_cache is not None:
            # NOTE: (seonghwanseo) Since atom encoder is used in both InputEmbedder and
            # DiffusionModule, I constrain caching only for DiffusionModule usage
            # according to Boltz's implementation.
            assert self.use_structure, "Caching is only supported when using structure."
            cache_prefix = "atom_attn_encoder"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]
        else:
            layer_cache = {}

        if len(layer_cache) == 0:
            # Cache the representation for structure-independent components

            # Get indexing matrix for single to keys conversion
            Na = len(f_input.atom)
            W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
            K = Na // W
            indexing_matrix = get_indexing_matrix(K, W, H, f_input.device)
            to_keys = partial(single_to_keys, indexing_matrix=indexing_matrix, W=W, H=H)

            # Initialize single conditioning and pair representations
            # Line 1
            c = self.get_atom_single_conditioning(f_input)  # [Na, c_atom]
            # Line 2-6
            p = self.get_atom_pair_representation(
                f_input, to_keys
            )  # [K, W, H, c_atompair]
            # Line 7
            q = c

            if self.use_structure:
                # Add trunk embedding
                assert s_trunk is not None and z is not None and r is not None
                # Line 9
                c = self.add_trunk_single_conditioning(c, s_trunk, f_input.atom_to_token)
                # Line 10
                p = self.add_trunk_pair_embedding(p, z, f_input.atom_to_token, to_keys)

            # Line 13-14
            p = p + self.c_to_p_trans_q(c.view(K, W, 1, c.shape[-1]))
            p = p + self.c_to_p_trans_k(to_keys(c).view(K, 1, H, c.shape[-1]))
            p = p + self.p_mlp(p)  # [K, W, H, c_atompair]

            layer_cache["q"] = q
            layer_cache["c"] = c
            layer_cache["p"] = p
            layer_cache["to_keys"] = to_keys
        else:
            q = layer_cache["q"]
            c = layer_cache["c"]
            p = layer_cache["p"]
            to_keys = layer_cache["to_keys"]

        # Shapes at this point:
        # q: [Na, c_atom]
        # c: [Na, c_atom]
        # p: [K, W, H, c_atompair]

        if self.use_structure:
            # NOTE: here we repeat the conditioning for each diffused sample
            # batch size = num_diffusion_samples (number of diffused samples per input)
            assert r is not None
            Nsample = r.shape[0]
        else:
            # Else, use batch size 1
            Nsample = 1

        c = expand_batch(c, Nsample)  # [Nsample, Na, c_atom]
        q = expand_batch(q, Nsample)  # [Nsample, Na, c_atom]
        p = expand_batch(p, Nsample)  # [Nsample, K, W, H, c_atompair]
        atom_mask = expand_batch(atom_mask, Nsample)  # [Nsample, Na]

        # Shapes at this point:
        # q: [Nsample, Na, c_atom]
        # c: [Nsample, Na, c_atom]
        # p: [Nsample, K, W, H, c_atompair]

        # Line 11
        if self.use_structure:
            assert r is not None
            q = self.add_noise_position(q, r)

        # Line 15
        q = self.atom_encoder(
            q=q,
            c=c,
            p=p,
            mask=atom_mask,
            to_keys=to_keys,
            model_cache=layer_cache,
        )

        # Aggregate atom representations to token representations
        # [Nsample, Na, c_atom] -> [Nsample, Nt, c_token]
        # NOTE that c_token can be different from c_s (channel_s)
        # Line 16
        q_to_a = self.atom_to_token_trans(q)
        atom_to_token = f_input.atom_to_token.float()
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=0, keepdim=True) + 1e-6
        )  # [Na, Nt]
        a = torch.bmm(atom_to_token_mean.T.unsqueeze(0), q_to_a)  # [Nsample, Nt, c_token]

        # Line 17
        q_skip, c_skip, p_skip = q, c, p

        return a, q_skip, c_skip, p_skip, to_keys

    def get_atom_single_conditioning(
        self,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Algorithm 5, Line 1
        Get atom single conditioning.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.

        Returns
        -------
        c: torch.Tensor
            The atom single conditioning, shape [Na, c_atom].
        """
        return self.embed_atom(f_input)  # [Na, c_atom]

    def get_atom_pair_representation(
        self,
        f_input: FoldingInput,
        to_keys: Callable,
    ) -> torch.Tensor:
        """Algorithm 5, Line 2-4
        Get atom pair representation.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        to_keys : Callable
            The function to convert single representation to keys representation.

        Returns
        -------
        p: torch.Tensor
            The atom pair representation, shape [K, W, H, c_atompair].
        """
        # Create atom-pairwise representation with AtomTransformer
        # NOTE(seonghwanseo): Message passing is only performed between atoms in
        # same residues (ref_space_uid: residue unique id)
        ref_pos = f_input.atom.ref_pos.unsqueeze(0)  # [Na, 3]
        residue_uid = f_input.atom.ref_space_uid.unsqueeze(0)  # [Na]
        mask = f_input.atom.pad_mask

        Na = len(f_input.atom)
        W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
        K = Na // W

        ref_pos_queries = ref_pos.view(K, W, 1, 3)
        ref_pos_keys = to_keys(ref_pos).view(K, 1, H, 3)

        d = ref_pos_keys - ref_pos_queries  # [K, W, H, 3]
        d_norm = torch.sum(d * d, dim=-1, keepdim=True)
        d_norm = 1 / (1 + d_norm)  # [K, W, H, 1]

        # Create pair mask based on uids
        mask_queries = mask.view(K, W, 1)
        mask_keys = to_keys(mask.unsqueeze(-1).float()).view(K, 1, H).bool()
        uid_queries = residue_uid.view(K, W, 1)
        uid_keys = to_keys(residue_uid.unsqueeze(-1).float()).view(K, 1, H).long()
        pair_mask = (
            (mask_queries & mask_keys & (uid_queries == uid_keys)).float().unsqueeze(-1)
        )  # [K, W, H, 1]

        p = self.embed_atompair_ref_pos(d)
        p = p + self.embed_atompair_ref_dist(d_norm)
        p = p + self.embed_atompair_mask(pair_mask)
        p = p * pair_mask  # [K, W, H, c_atompair]
        return p  # [K, W, H, c_atompair]

    def add_trunk_single_conditioning(
        self,
        c: torch.Tensor,
        s_trunk: torch.Tensor,
        atom_to_token: torch.Tensor,
    ) -> torch.Tensor:
        """Algorithm 5, Line 9
        Add trunk single embedding to atom single conditioning.

        Parameters
        ----------
        c : torch.Tensor
            The atom single conditioning, shape [Na, c_atom].
        s_trunk : torch.Tensor
            The trunk single representation, shape [Nt, c_s].
        atom_to_token : torch.Tensor
        The atom to token mapping, shape [Na, Nt].
        """
        # [Nt, c_s] -> [Na, c_atom]
        s_to_c = self.s_to_c_trans(s_trunk)
        s_to_c = torch.bmm(atom_to_token, s_to_c)
        return c + s_to_c  # [Na, c_atom]

    def add_trunk_pair_embedding(
        self,
        p: torch.Tensor,
        z_trunk: torch.Tensor,
        atom_to_token: torch.Tensor,
        to_keys: Callable,
    ) -> torch.Tensor:
        """Algorithm 5, Line 10
        Add trunk pair embedding to atom pair representation.

        Parameters
        ----------
        p : torch.Tensor
            The atom pair representation, shape [K, W, H, c_atompair].
        to_keys : Callable
            The function to convert single representation to keys representation.
        z_trunk : torch.Tensor
            The trunk pair representation, shape [Nt, c_z].
        atom_to_token : torch.Tensor
            The atom to token mapping, shape [Na, Nt].
        """
        # [Nt, Nt, c_z] -> [K, W, H, c_atompair]
        Na, Nt = p.shape[0], z_trunk.shape[0]
        W = self.atoms_per_window_queries
        K = Na // W
        atom_to_token_queries = atom_to_token.view(K, W, Nt)
        atom_to_token_keys = to_keys(atom_to_token)
        z_to_p = self.z_to_p_trans(z_trunk)
        z_to_p = torch.einsum(
            "ijd,wki,wlj->wkld",
            z_to_p,
            atom_to_token_queries,
            atom_to_token_keys,
        )
        return p + z_to_p  # [K, W, H, c_atompair]

    def add_noise_position(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Algorithm 5, Line 11
        Add noise position to atom single representation.

        Parameters
        ----------
        q : torch.Tensor
            The atom single representation, shape [Nsample, Na, c_atom].
        r : torch.Tensor
            The noised structures' positions, shape [Nsample, Na, c_r].
        """
        r_to_q = self.r_to_q_trans(r)
        return q + r_to_q  # [Nsample, Na, c_atom]


class AtomAttentionDecoder(nn.Module):
    """Atom attention decoder.
    Section 3.2 Algorithm 7 Atom Attention Encoder
    """

    def __init__(
        self,
        channel_s: int,
        channel_atom: int,
        channel_atompair: int,
        num_blocks: int = 3,
        num_heads: int = 4,
        attn_window_queries: int = 32,
        attn_window_keys: int = 128,
        activation_checkpointing=False,
    ):
        """Initialize the atom attention decoder.

        Parameters
        ----------
        channel_s : int
            The single representation dimension.
        channel_atom : int
            The atom single representation dimension.
        channel_atompair : int
            The atom pair representation dimension.
        num_blocks : int, optional
            The number of transformer blocks, by default 3.
        num_heads : int, optional
            The number of transformer heads, by default 4.
        attn_window_queries : int
            The number of atoms per window for queries.
        attn_window_keys : int
            The number of atoms per window for keys.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.

        """
        super().__init__()

        self.a_to_q_trans = LinearNoBias(2 * channel_s, channel_atom)
        init.final_init_(self.a_to_q_trans.weight)

        self.atom_decoder = AtomTransformer(
            channel_a=channel_atom,
            channel_s=channel_atom,
            channel_z=channel_atompair,
            num_blocks=num_blocks,
            num_heads=num_heads,
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            activation_checkpointing=activation_checkpointing,
        )

        self.atom_feat_to_atom_pos_update = nn.Sequential(
            nn.LayerNorm(channel_atom), LinearNoBias(channel_atom, 3)
        )
        init.final_init_(self.atom_feat_to_atom_pos_update[1].weight)

    def forward(
        self,
        a: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        f_input: FoldingInput,
        to_keys,
        num_diffusion_samples: int = 1,
        model_cache=None,
    ):
        atom_mask = f_input.atom.pad_mask  # [Na]

        a_to_q = self.a_to_q_trans(a)  # [Ndiff, Nt, c_atom]
        a_to_q = torch.bmm(f_input.atom_to_token, a_to_q)  # [Ndiff, Na, c_atom]
        q = q + a_to_q  # [Ndiff, Na, c_atom]

        layer_cache = None
        if model_cache is not None:
            cache_prefix = "atom_attn_decoder"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]

        q = self.atom_decoder(
            q=q,
            mask=atom_mask,
            c=c,
            p=p,
            num_diffusion_samples=num_diffusion_samples,
            to_keys=to_keys,
            model_cache=layer_cache,
        )

        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update
