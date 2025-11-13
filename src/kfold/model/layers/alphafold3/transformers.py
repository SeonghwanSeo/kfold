# Started from code from https://github.com/jwohlwend/boltz, MIT License

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from kfold.data.model_input import FoldingInput

from . import initialize as init
from .embeddings import AtomEmbedding
from .primitives import AdaLN, LinearNoBias
from .utils import LocalAttentionIndexer, expand_dim

# === Helper functions for local atom attention === #


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

        self.proj_q = nn.Sequential(
            nn.Linear(channel_a, channel_a),
            Rearrange("b ... l (h d) -> b ... h l d", h=num_heads),
        )
        self.proj_k = nn.Sequential(
            LinearNoBias(channel_a, channel_a),
            Rearrange("b ... l (h d) -> b ... h l d", h=num_heads),
        )
        self.proj_v = nn.Sequential(
            LinearNoBias(channel_a, channel_a),
            Rearrange("b ... l (h d) -> b ... h l d", h=num_heads),
        )
        self.proj_g = LinearNoBias(channel_a, channel_a)

        self.proj_z = nn.Sequential(
            nn.LayerNorm(channel_z),
            LinearNoBias(channel_z, num_heads),
            Rearrange("... l1 l2 h -> ... h l1 l2"),
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
        attn_mask: torch.Tensor,
        local_attn_indexer: LocalAttentionIndexer | None = None,
        model_cache=None,
        use_kernels: bool = True,
    ) -> torch.Tensor:
        """Forward pass.
        See Section 3.7 Algorithm 24 of AlphaFold3 paper.

        Parameters
        ----------
        a : torch.Tensor
            The input atom/token tensor (..., L, c_a)
        s : torch.Tensor | None
            The input single tensor (..., L, c_s), can be None if use_s is False
        z : torch.Tensor
            The input pairwise tensor (..., Lq, Lk, c_z)
        attn_mask : torch.Tensor
            The pairwise mask tensor (..., Lq, Lk)
        local_attn_indexer : LocalAttentionIndexer | None
            The local attention indexer, by default None

        Returns
        -------
        a : torch.Tensor
            The output sequence tensor. (B, N, c_a)

        """
        # === Input projection === #
        if self.use_s:
            # Line 1-2
            assert s is not None, "s cannot be None if use_s is True"
            a = self.adaln(a, s)
        else:
            # Line 3-4
            assert s is None, "s must be None if use_s is False"
            a = self.norm_a(a)

        if local_attn_indexer is not None:
            q_in = local_attn_indexer.to_query(a)  # [..., W, Lq, c_a]
            k_in = local_attn_indexer.to_key(a)  # [..., W, Lk, c_a]
        else:
            q_in = a  # [..., L, C_a]
            k_in = a  # [..., L, C_a]

        # Line 6
        q = self.proj_q(q_in)  # [..., H, Lq, Dh]

        # Line 7
        k = self.proj_k(k_in)  # [..., H, Lk, Dh]
        v = self.proj_v(k_in)  # [..., H, Lk, Dh]

        # Line 8
        # Caching attention bias during diffusion roll-out
        if model_cache is None or "attn_bias" not in model_cache:
            attn_bias = self.proj_z(z)

            # The pairwise mask (..., Lq, Lk) is broadcasted to (..., H, Lq, Lk)
            attn_bias = attn_bias - self.inf * (1.0 - attn_mask.unsqueeze(-3))

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
            )
        else:
            with torch.autocast("cuda", enabled=False):
                # Compute attention weights
                attn = torch.einsum("bhid,bhjd->bhij", q.float(), k.float())
                # Add attention bias
                attn = attn / (self.head_dim**0.5) + attn_bias
                attn = attn.softmax(dim=-1)

                # Compute output
                Av = torch.einsum("bhij,bhjd->bhid", attn, v.float()).to(v.dtype)

        Av = rearrange(Av, "... h l d -> ... l (h d)")
        Av = Av.reshape(a.shape)  # [..., L, c_a]

        # Line 11
        a = self.proj_out(g * Av)

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
        attn_mask: torch.Tensor,
        local_attn_indexer: LocalAttentionIndexer | None = None,
        model_cache=None,
    ):
        """See Section 3.7 Algorithm 23 Diffusion Transformer"""
        # Line 1, 4
        for i, block in enumerate(self.blocks):
            if model_cache is not None:
                prefix_cache = "layer_" + str(i)
                block_cache = model_cache.setdefault(prefix_cache, {})
            else:
                block_cache = None

            if self.activation_checkpointing and self.training:
                a = torch.utils.checkpoint.checkpoint(
                    block,
                    a,
                    s,
                    z,
                    attn_mask,
                    local_attn_indexer,
                    use_reentrant=False,
                )

            else:
                a = block(
                    a,
                    s,
                    z,
                    attn_mask=attn_mask,
                    local_attn_indexer=local_attn_indexer,
                    block_cache=block_cache,
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
        self.attention = AttentionPairBias(
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
        attn_mask: torch.Tensor,
        local_attn_indexer: LocalAttentionIndexer | None = None,
        block_cache: dict | None = None,
    ) -> torch.Tensor:
        """See Section 3.7 Algorithm 23 Diffusion Transformer

        Parameters
        ----------
        a : torch.Tensor
            The input single representation tensor (..., L, c_a)
        s : torch.Tensor
            The input single condition tensor (..., L, c_s)
        z : torch.Tensor
            The input pair representation tensor (..., Lq, Lk, c_z)
        attn_mask : torch.Tensor
            The pairwise mask tensor (..., Lq, Lk)

        Returns
        -------
        a : torch.Tensor
            The output single representation tensor (..., L, c_a)

        NOTE
        ----
        Below is the original Algorithm 23:
        b = AttentionPairBias(a, s, z, attn_mask)  # Line 2
        a = b + ConditionedTransitionBlock(a, s)  # Line 3

        However, its official implementation uses residual connections:
        a = a + AttentionPairBias(a, s, z, attn_mask)
        a = a + ConditionedTransitionBlock(a, s)

        See https://github.com/google-deepmind/alphafold3/blob/f3e86f27dfac16559d16f470bb2f9323eb357f1f/src/alphafold3/model/network/diffusion_transformer.py#L209-L226

        """
        # Line 2
        a = a + self.attention(
            a=a,
            s=s,
            z=z,
            attn_mask=attn_mask,
            local_attn_indexer=local_attn_indexer,
            model_cache=block_cache,
        )
        # Line 3
        a = a + self.transition(a, s)
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
        model_cache: dict | None = None,
    ) -> torch.Tensor:
        """See Section 3.2 Algorithm 7 Atom Transformer

        Parameters
        ----------
        q : torch.Tensor
            The single representation, shape [..., La, c_atom].
        c : torch.Tensor
            The single conditioning, shape [..., La, c_atom].
        p : torch.Tensor
            The pair representation, shape [..., W, Lq, Lk, c_atompair]
            where K = L / W.
        mask : torch.Tensor
            The padding mask, shape [..., La].

        Returns
        -------
        q : torch.Tensor
            The output single representation, shape [..., La, c_atom].
        """

        original_shape = q.shape  # [..., La, D]

        q = q.flatten(0, -3)  # [B, L, c_atom]
        c = c.flatten(0, -3)  # [B, L, c_atom]
        p = p.flatten(0, -5)  # [B, W, Lq, Lk, c_atompair]
        mask = mask.flatten(0, -2)  # [B, La]

        local_attn_indexer = LocalAttentionIndexer(
            num_atoms=mask.shape[-1],
            atoms_per_window_queries=self.attn_window_queries,
            atoms_per_window_keys=self.attn_window_keys,
            device=q.device,
        )

        mask = mask.float().unsqueeze(-1)  # [B, La, 1]
        attn_mask = local_attn_indexer.to_key(mask).squeeze(-1)  # [B, W, Lk]
        attn_mask = attn_mask.unsqueeze(-2)  # [B, W, 1, Lk]

        # main transformer
        q = self.diffusion_transformer(
            a=q,  # [B, L, c_atom]
            s=c,  # [B, L, c_atom]
            z=p,  # [B, W, Lq, Lk, c_atompair]
            attn_mask=attn_mask,  # [B, W, Lq, Lk], broadcastable(Lq=1)
            local_attn_indexer=local_attn_indexer,
            model_cache=model_cache,
        )

        q = q.view(original_shape)

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
                nn.LayerNorm(channel_z),
                LinearNoBias(channel_z, channel_atompair),
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            The trunk single representation, shape [B, Lt, c_s].
        z : torch.Tensor | None
            The conditioning pair representation, shape [B, Lt, c_z].
        r : torch.Tensor | None
            The noised structures' positions, shape [B, N, La, c_r],
            where Nsample is the number of diffusion samples.
        model_cache : dict | None
            The model cache for storing intermediate representations, by default None.

        Returns
        -------
        a : torch.Tensor
            The token single representation, shape [B, N, Lt, c_token]
        q_skip : torch.Tensor
            The atom single representation, shape [B, N, La, c_atom]
        c_skip : torch.Tensor
            The atom single conditioning, shape [B, N, La, c_atom]
        p_skip : torch.Tensor
            The atom pair representation, shape [B, N, W, Lq, Lk, c_atompair]
        """
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
            local_attn_indexer = LocalAttentionIndexer(
                num_atoms=f_input.num_atoms,
                atoms_per_window_queries=self.atoms_per_window_queries,
                atoms_per_window_keys=self.atoms_per_window_keys,
                device=f_input.device,
            )

            # Initialize single conditioning and pair representations
            # Line 1
            c = self.embed_atom(f_input)  # [B, La, c_atom]

            # Line 2
            ref_pos = f_input.atom.ref_pos  # [B, La, 3]
            ref_pos_q = local_attn_indexer.to_query(ref_pos)  # [B, W, Lq, 3]
            ref_pos_k = local_attn_indexer.to_key(ref_pos)  # [B, W, Lk, 3]
            d = ref_pos_q.unsqueeze(-2) - ref_pos_k.unsqueeze(-3)  # [B, W, Lq, Lk, 3]

            # Line 3
            residue_uid = f_input.atom.ref_space_uid.unsqueeze(-1)  # [B, La, 1]
            uid_q = local_attn_indexer.to_query(residue_uid)  # [B, W, Lq, 1]
            uid_k = local_attn_indexer.to_key(residue_uid)  # [B, W, Lk, 1]
            v = (uid_q.unsqueeze(-2) == uid_k.unsqueeze(-3)).float()  # [B, W, Lq, Lk, 1]

            # Line 4, skip masking
            p = self.embed_atompair_ref_pos(d)

            # Line 5, skip masking
            d_sq = d.pow(2).sum(-1, keepdim=True)
            p = p + self.embed_atompair_ref_dist(1 / (1 + d_sq))

            # Line 6, skip masking
            p = p + self.embed_atompair_mask(v)

            # Line 4-6, mask at once
            p = p * v  # [B, W, Lq, Lk, c_atompair]

            # Line 7
            q = c  # [B, La, c_atom]

            mask = f_input.atom.pad_mask.float()  # [B, La]

            if self.use_structure:
                # Add trunk embedding
                assert s_trunk is not None and z is not None and r is not None
                # Line 9
                c = self.add_trunk_single_conditioning(c, s_trunk, f_input.atom_to_token)
                # Line 10
                p = self.add_trunk_pair_embedding(
                    p, z, f_input.atom_to_token, local_attn_indexer
                )

            # Line 13-14
            c_q = local_attn_indexer.to_query(c)  # [B, W, Lq, c_atom]
            c_k = local_attn_indexer.to_key(c)  # [B, W, Lk, c_atom]
            p = p + self.c_to_p_trans_q(c_q).unsqueeze(-2)
            p = p + self.c_to_p_trans_k(c_k).unsqueeze(-3)
            p = p + self.p_mlp(p)  # [B, W, Lq, Lk, c_atompair]

            layer_cache["q"] = q  # [B, La, c_atom]
            layer_cache["c"] = c  # [B, La, c_atom]
            layer_cache["p"] = p  # [B, W, Lq, Lk, c_atompair]
            layer_cache["mask"] = mask  # [B, La]
        else:
            q = layer_cache["q"]
            c = layer_cache["c"]
            p = layer_cache["p"]
            mask = layer_cache["mask"]

        # Shapes at this point:
        # q: [B, La, c_atom]
        # c: [B, La, c_atom]
        # p: [B, W, Lq, Lk, c_atompair]
        # mask: [B, La]

        # Repeat for diffusion samples
        if self.use_structure:
            assert r is not None, "r cannot be None when use_structure is True"
            N = r.shape[1]  # number of diffusion samples
        else:
            N = 1

        q = expand_dim(q, dim=1, n=N, add_dim=True)
        c = expand_dim(c, dim=1, n=N, add_dim=True)
        p = expand_dim(p, dim=1, n=N, add_dim=True)
        mask = expand_dim(mask, dim=1, n=N, add_dim=True)

        # Shapes at this point:
        # q: [B, N, La, c_atom]
        # c: [B, N, La, c_atom]
        # p: [B, N, W, Lq, Lk, c_atompair]
        # mask: [B, N, La]

        # Line 11
        if self.use_structure:
            assert r is not None
            q = self.add_noise_position(q, r)

        # Line 15
        q = self.atom_encoder(
            q=q,
            c=c,
            p=p,
            mask=mask,
            model_cache=layer_cache,
        )

        # Aggregate atom representations to token representations
        # [B, N, La, c_atom] -> [B, N, Lt, c_token]
        # NOTE that c_token can be different from c_s (channel_s)
        # Line 16
        atom_to_token = f_input.atom_to_token  # [B, La, Lt]
        atom_to_token_mean = (
            atom_to_token / atom_to_token.sum(dim=-2, keepdim=True).clamp(1)
        ).permute(0, 2, 1)  # [B, Lt, La]
        q_to_a = self.atom_to_token_trans(q)  # [B, N, La, c_token]
        a = torch.einsum("btl, bnlc -> bntc", atom_to_token_mean, q_to_a)

        # Line 17
        q_skip, c_skip, p_skip = q, c, p

        return a, q_skip, c_skip, p_skip

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
            The atom single conditioning, shape [B, La, c_atom].
        s_trunk : torch.Tensor
            The trunk single representation, shape [B,Lt, c_s].
        atom_to_token : torch.Tensor
        The atom to token mapping, shape [B, La, Lt].
        """
        # [B, Lt, c_s] -> [B, La, c_atom]
        s_to_c = self.s_to_c_trans(s_trunk)
        s_to_c = torch.bmm(atom_to_token, s_to_c)
        return c + s_to_c  # [B, La, c_atom]

    def add_trunk_pair_embedding(
        self,
        p: torch.Tensor,
        z_trunk: torch.Tensor,
        atom_to_token: torch.Tensor,
        local_attn_indexer: LocalAttentionIndexer,
    ) -> torch.Tensor:
        """Algorithm 5, Line 10
        Add trunk pair embedding to atom pair representation.

        Parameters
        ----------
        p : torch.Tensor
            The atom pair representation, shape [B, W, Lq, Lk, c_atompair].
        z_trunk : torch.Tensor
            The trunk pair representation, shape [B, Lt, c_z].
        atom_to_token : torch.Tensor
            The atom to token mapping, shape [B, La, Lt].
        local_attn_indexer : LocalAttentionIndexer
            The local attention indexer for atom attention.
        """
        # [B, Lt, Lt, c_z] -> [B, W, Lq, Lk, c_atompair]

        atom_to_token_q = local_attn_indexer.to_query(atom_to_token)  # [B, W, Lq, Lt]
        atom_to_token_k = local_attn_indexer.to_key(atom_to_token)  # [B, W, Lk, Lt]

        z_to_p = self.z_to_p_trans(z_trunk)  # [B, Lt, Lt, c_atompair]
        z_to_p = torch.einsum(
            "bijd,bwki,bwlj->bwkld",
            z_to_p,  # [B, Lt, Lt, c_atompair]
            atom_to_token_q,  # [B, W, Lq, Lt]
            atom_to_token_k,  # [B, W, Lk, Lt]
        )
        return p + z_to_p  # [B, W, Lq, Lk, c_atompair]

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
            The atom single representation, shape [B, N, La, c_atom].
        r : torch.Tensor
            The noised structures' positions, shape [B, N, La, 3].
        """
        r_to_q = self.r_to_q_trans(r)  # [B, N, La, c_atom]
        return q + r_to_q  # [B, N, La, c_atom]


class AtomAttentionDecoder(nn.Module):
    """Atom attention decoder.
    Section 3.2 Algorithm 6 Atom Attention Decoder
    """

    def __init__(
        self,
        channel_a: int,
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
        channel_a : int
            The token single representation dimension.
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

        self.a_to_q_trans = LinearNoBias(channel_a, channel_atom)
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
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        f_input: FoldingInput,
        model_cache=None,
    ):
        """Forward pass of the atom attention decoder.
        See Algorithm 6 in the AF3 paper for more details.

        Parameters
        ----------
        a : torch.Tensor
            The token single representation, shape [B, N, Lt, c_token].
        q_skip : torch.Tensor
            The atom single representation, shape [B, N, La, c_atom].
        c_skip : torch.Tensor
            The atom single conditioning, shape [B, N, La, c_atom].
        p_skip : torch.Tensor
            The atom pair representation, shape [B, N, W, Lq, Lk, c_atompair].
        f_input : FoldingInput
            The folding input.

        Returns
        -------
        r_update : torch.Tensor
            The atom position updates, shape [B, N, La, 3].
        """
        a_to_q = self.a_to_q_trans(a)  # [B, N, Lt, c_atom]
        atom_to_token = f_input.atom_to_token  # [B, La, Lt]
        a_to_q = torch.einsum(
            "bat, bntc -> bnac", atom_to_token, a_to_q
        )  # [B, N, La, c_atom]
        q = q_skip + a_to_q  # [B, N, La, c_atom]

        mask = f_input.atom.pad_mask.float().unsqueeze(-2)  # [B, N, La]

        layer_cache = None
        if model_cache is not None:
            cache_prefix = "atom_attn_decoder"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]

        q = self.atom_decoder(
            q=q,  # [B, N, La, c_atom]
            c=c_skip,  # [B, N, La, c_atom]
            p=p_skip,  # [B, N, W, Lq, Lk, c_atompair]
            mask=mask,  # [B, N, La], where N is broadcastable(=1)
            model_cache=layer_cache,
        )

        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update
