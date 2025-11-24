# Started from code from https://github.com/jwohlwend/boltz, MIT License

import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from kfold.data.model_input import FoldingInput
from kfold.model.layers.primitives import (
    AdaLN,
    LayerNorm,
    Linear,
    LinearNoBias,
    SwiGLU,
    attention,
)
from kfold.utils.checkpointing import checkpoint_blocks

from .embeddings import AtomEmbedding
from .utils import (
    LocalAttentionIndex,
    aggregate_atoms_to_tokens,
    broadcast_tokens_to_atoms,
    expand_dim,
)

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
            self.layernorm_a = LayerNorm(channel_a)

        self.proj_q = nn.Sequential(
            Linear(channel_a, channel_a, init="default"),
            Rearrange("b ... l (h d) -> b ... h l d", h=num_heads),
        )
        self.proj_k = nn.Sequential(
            LinearNoBias(channel_a, channel_a, init="default"),
            Rearrange("b ... l (h d) -> b ... h l d", h=num_heads),
        )
        self.proj_v = nn.Sequential(
            LinearNoBias(channel_a, channel_a, init="default"),
            Rearrange("b ... l (h d) -> b ... h l d", h=num_heads),
        )
        self.proj_g = LinearNoBias(channel_a, channel_a, init="gating")

        self.proj_z = nn.Sequential(
            LayerNorm(channel_z),
            LinearNoBias(channel_z, num_heads, init="default"),
            Rearrange("b ... l1 l2 h -> b ... h l1 l2"),
        )

        self.proj_out = LinearNoBias(channel_a, channel_a, init="final")

        if self.use_s:
            self.linear_s = Linear(channel_s, channel_a, init="gating_ada_zero")

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor | None,
        z: torch.Tensor,
        attn_mask: torch.Tensor,
        local_attn_index: LocalAttentionIndex | None = None,
        use_high_precision: bool = False,
        inplace: bool = False,
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
            The attention mask tensor (..., Lk)
            NOTE: We only mask key positions as in the official implementation.
        local_attn_index : LocalAttentionIndex | None
            The local attention indexer, by default None
        use_high_precision : bool
            Whether to use high precision for attention computation, by default True

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
            a = self.layernorm_a(a)

        if local_attn_index is not None:
            q_in = local_attn_index.to_query(a)  # [..., W, Lq, c_a]
            k_in = local_attn_index.to_key(a)  # [..., W, Lk, c_a]
        else:
            q_in = a  # [..., L, C_a]
            k_in = a  # [..., L, C_a]

        # Line 6
        q = self.proj_q(q_in)  # [..., H, Lq, Dh]

        # Line 7
        k = self.proj_k(k_in)  # [..., H, Lk, Dh]
        v = self.proj_v(k_in)  # [..., H, Lk, Dh]

        # Line 8
        attn_bias = self.proj_z(z)  # [..., H, Lq, Lk]
        attn_bias = attn_bias - self.inf * (1 - attn_mask.float())[..., None, None, :]

        # Line 9
        g = self.proj_g(a).sigmoid()

        # === Attention === #
        # Line 10-11
        Av = attention(
            q,
            k,
            v,
            bias=attn_bias,
            scale=math.sqrt(self.head_dim),
            use_high_precision=use_high_precision,
        )
        Av = rearrange(Av, "... h l d -> ... l (h d)")
        Av = Av.reshape(a.shape)

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
        blocks_per_ckpt: int | None = None,
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
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint

        """
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks_per_ckpt: int | None = blocks_per_ckpt
        for _ in range(num_blocks):
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
        local_attn_index: LocalAttentionIndex | None = None,
    ):
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
            The pairwise mask tensor (..., Lk)
            NOTE: We only mask key positions as in the official implementation.
        """
        # Line 1, 4

        blocks = [
            partial(
                b,
                attn_mask=attn_mask,
                local_attn_index=local_attn_index,
                block_cache=None,
            )
            for b in self.blocks
        ]

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        a, s, z = checkpoint_blocks(
            blocks,
            args=(a, s, z),
            blocks_per_ckpt=blocks_per_ckpt,
            use_reentrant=False,
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
        local_attn_index: LocalAttentionIndex | None = None,
        block_cache: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            local_attn_index=local_attn_index,
            use_high_precision=True,  # High precision for attention computation
        )
        # Line 3
        a = a + self.transition(a, s)

        # Return updated a, s, z (s and z are unchanged)
        # This is to maintain compatibility with checkpoint_blocks
        return a, s, z


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
        self.swiglu = SwiGLU(channel_a, model_dim)

        self.linear_g = Linear(channel_s, channel_a, init="gating_ada_zero")
        self.linear_out = LinearNoBias(model_dim, channel_a, init="default")
        self.sigmoid = nn.Sigmoid()

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """See Section 3.7 Algorithm 25 Conditioned Transition Block"""
        # Line 1
        a = self.adaln(a, s)

        # Line 2
        b = self.swiglu(a)

        # Line 3
        a = self.sigmoid(self.linear_g(s)) * self.linear_out(b)
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
        blocks_per_ckpt: int | None = None,
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
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        mask: torch.Tensor,
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

        local_attn_index = LocalAttentionIndex(
            num_atoms=mask.shape[-1],
            atoms_per_window_queries=self.attn_window_queries,
            atoms_per_window_keys=self.attn_window_keys,
            device=q.device,
        )

        # NOTE: mask the key positions only (masking query is not required)
        mask = mask.float()  # [B, La, 1]
        attn_mask = local_attn_index.to_key(mask[..., None]).squeeze(-1)  # [B, W, Lk]

        # main transformer
        q = self.diffusion_transformer(
            a=q,  # [B, L, c_atom]
            s=c,  # [B, L, c_atom]
            z=p,  # [B, W, Lq, Lk, c_atompair]
            attn_mask=attn_mask,  # [B, W, Lk]
            local_attn_index=local_attn_index,
        )

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
        channel_coords: int = 3,
        num_blocks=3,
        num_heads=4,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        use_structure: bool = True,
        blocks_per_ckpt: int | None = None,
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
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint, by default None.

        """
        super().__init__()
        self.atoms_per_window_queries: int = atoms_per_window_queries
        self.atoms_per_window_keys: int = atoms_per_window_keys

        self.embed_atom = AtomEmbedding(channel_atom)
        self.embed_atompair_ref_pos = LinearNoBias(3, channel_atompair, init="default")
        self.embed_atompair_ref_dist = LinearNoBias(1, channel_atompair, init="default")
        self.embed_atompair_mask = LinearNoBias(1, channel_atompair, init="default")

        self.use_structure = use_structure
        if use_structure:
            self.linear_s_to_c = nn.Sequential(
                LayerNorm(channel_s, bias=False),
                LinearNoBias(channel_s, channel_atom, init="final"),
            )
            self.linear_z_to_p = nn.Sequential(
                LayerNorm(channel_z, bias=False),
                LinearNoBias(channel_z, channel_atompair, init="final"),
            )
            self.linear_r_to_q = LinearNoBias(
                channel_coords, channel_atom, init="default"
            )

        self.linear_key = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(channel_atom, channel_atompair, init="default"),
        )

        self.linear_query = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(channel_atom, channel_atompair, init="default"),
        )

        self.mlp_pair = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(channel_atompair, channel_atompair, init="relu"),
            nn.ReLU(),
            LinearNoBias(channel_atompair, channel_atompair, init="relu"),
            nn.ReLU(),
            LinearNoBias(channel_atompair, channel_atompair, init="final"),
        )

        self.atom_encoder = AtomTransformer(
            channel_a=channel_atom,
            channel_s=channel_atom,
            channel_z=channel_atompair,
            num_blocks=num_blocks,
            num_heads=num_heads,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        self.linear_q_to_a = nn.Sequential(
            LinearNoBias(channel_atom, channel_token, init="default"),
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
            local_attn_index = LocalAttentionIndex(
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
            ref_pos_q = local_attn_index.to_query(ref_pos)  # [B, W, Lq, 3]
            ref_pos_k = local_attn_index.to_key(ref_pos)  # [B, W, Lk, 3]
            d = ref_pos_q.unsqueeze(-2) - ref_pos_k.unsqueeze(-3)  # [B, W, Lq, Lk, 3]

            # Line 3
            residue_uid = f_input.atom.ref_space_uid  # [B, La]
            mask = f_input.atom.pad_mask  # [B, La]
            uid_q = local_attn_index.to_query(residue_uid, dim=-1)  # [B, W, Lq]
            uid_k = local_attn_index.to_key(residue_uid, dim=-1)  # [B, W, Lk]
            mask_q = local_attn_index.to_query(mask, dim=-1)  # [B, W, Lq]
            mask_k = local_attn_index.to_key(mask, dim=-1)  # [B, W, Lk]
            v = uid_q[..., :, None] == uid_k[..., None, :]  # [B, W, Lq, Lk]
            v = v & mask_q[..., :, None] & mask_k[..., None, :]  # [B, W, Lq, Lk]
            v = v.float().unsqueeze(-1)  # [B, W, Lq, Lk, 1]

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

            if self.use_structure:
                assert s_trunk is not None and z is not None and r is not None
                # Add trunk embedding
                atom_mask = f_input.atom.pad_mask
                token_index = f_input.atom.token_index
                # Line 9
                c = self.add_trunk_single_conditioning(c, s_trunk, token_index, atom_mask)
                # Line 10
                p = self.add_trunk_pair_embedding(
                    p, z, token_index, atom_mask, local_attn_index
                )

            # Line 13-14
            c_q = local_attn_index.to_query(c)  # [B, W, Lq, c_atom]
            c_k = local_attn_index.to_key(c)  # [B, W, Lk, c_atom]
            p = p + self.linear_query(c_q)[..., :, None, :]
            p = p + self.linear_key(c_k)[..., None, :, :]
            p = p + self.mlp_pair(p)  # [B, W, Lq, Lk, c_atompair]

            layer_cache["q"] = q  # [B, La, c_atom]
            layer_cache["c"] = c  # [B, La, c_atom]
            layer_cache["p"] = p  # [B, W, Lq, Lk, c_atompair]
        else:
            q = layer_cache["q"]
            c = layer_cache["c"]
            p = layer_cache["p"]

        # Shapes at this point:
        # q: [B, La, c_atom]
        # c: [B, La, c_atom]
        # p: [B, W, Lq, Lk, c_atompair]

        # Repeat for diffusion samples
        if self.use_structure:
            assert r is not None, "r cannot be None when use_structure is True"
            N = r.shape[1]  # number of diffusion samples
        else:
            N = 1

        # [B, ...] -> [B, N, ...]
        mask = f_input.atom.pad_mask
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
        q = self.atom_encoder(q, c, p, mask)

        # NOTE that c_token can be different from c_s (channel_s)
        # Line 16
        q_to_a = self.linear_q_to_a(q)  # [B, N, La, c_token]

        # Aggregate atom representations to token representations
        # [B, N, La, c_atom] -> [B, N, Lt, c_token]
        a = aggregate_atoms_to_tokens(
            q_to_a,  # [B, N, La, c_token]
            token_index=f_input.atom.token_index[:, None, :],  # [B, 1, La]
            num_tokens=f_input.num_tokens,
            atom_mask=f_input.atom.pad_mask[:, None, :],  # [B, 1, La]
            aggr="mean",
        )

        # Line 17
        q_skip, c_skip, p_skip = q, c, p

        return a, q_skip, c_skip, p_skip

    def add_trunk_single_conditioning(
        self,
        c: torch.Tensor,
        s_trunk: torch.Tensor,
        token_index: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Algorithm 5, Line 9
        Add trunk single embedding to atom single conditioning.

        Parameters
        ----------
        c : torch.Tensor
            The atom single conditioning, shape [B, La, c_atom].
        s_trunk : torch.Tensor
            The trunk single representation, shape [B,Lt, c_s].
        token_index: torch.Tensor
            The atom to token mapping, shape [B, La].
        atom_mask : torch.Tensor
            The atom padding mask, shape [B, La]
        """
        # Case 2
        s_trunk = self.linear_s_to_c(s_trunk)  # [B, Lt, c_atom]
        s_to_c = broadcast_tokens_to_atoms(s_trunk, token_index)  # [B, La, c_atom]
        s_to_c = s_to_c * atom_mask.unsqueeze(-1)  # [B, La, c_atom]
        return c + s_to_c  # [B, La, c_atom]

    def add_trunk_pair_embedding(
        self,
        p: torch.Tensor,
        z_trunk: torch.Tensor,
        token_index: torch.Tensor,
        atom_mask: torch.Tensor,
        local_attn_index: LocalAttentionIndex,
    ) -> torch.Tensor:
        """Algorithm 5, Line 10
        Add trunk pair embedding to atom pair representation.

        Parameters
        ----------
        p : torch.Tensor
            The atom pair representation, shape [B, W, Lq, Lk, c_atompair].
        z_trunk : torch.Tensor
            The trunk pair representation, shape [B, Lt, Lt, c_z].
        token_index : torch.Tensor
            The atom to token mapping, shape [B, La].
        local_attn_index : LocalAttentionIndex
            The local attention indexer for atom attention.
        """
        B = token_index.shape[0]

        # 1. Project Trunk features
        # [B, Lt, Lt, c_z] -> [B, Lt, Lt, c_atompair]
        z_trunk = self.linear_z_to_p(z_trunk)

        # 2. Get Windowed Indices
        # [B, La] -> [B, W, Lq], [B, W, Lk]
        q_idx = local_attn_index.to_query(token_index, dim=-1)  # [B, W, Lq]
        k_idx = local_attn_index.to_key(token_index, dim=-1)  # [B, W, Lk]

        # NOTE: safe indexing: Although the pad value of token_index is 0,
        # we clamp indices to be at least 0 to avoid run-time error.
        q_idx, k_idx = q_idx.clamp(min=0), k_idx.clamp(min=0)

        # 3. Create Batch Indices
        # Broadcast Trunk Pair Embedding to Atom Pair Representation
        # [B, Ntoken, c_atom_pair] -> [B, W, Lq, Lk, c_atompair]
        # batch_idx: [B, 1, 1, 1]
        # q_idx: [B, W, Lq] -> [B, W, Lq, 1]
        # k_idx: [B, W, Lk] -> [B, W, 1, Lk]
        batch_idx = torch.arange(B, device=p.device).view(B, 1, 1, 1)

        # Token pair embedding to atom pair representation
        z_to_p = z_trunk[batch_idx, q_idx[..., None], k_idx[..., None, :]]

        # 5. Apply Padding Mask
        q_mask = local_attn_index.to_query(atom_mask, dim=-1)  # [B, W, Lq]
        k_mask = local_attn_index.to_key(atom_mask, dim=-1)  # [B, W, Lk]
        pair_mask = q_mask[..., :, None] & k_mask[..., None, :]  # [B, W, Lq, Lk]
        z_to_p.masked_fill_(~pair_mask[..., None], 0.0)

        return p + z_to_p

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
        r_to_q = self.linear_r_to_q(r)  # [B, N, La, c_atom]
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
        blocks_per_ckpt: int | None = None,
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
        blocks_per_ckpt : int | None, optional
            The number of blocks per checkpoint, by default None.

        """
        super().__init__()

        self.linear_a_to_q = LinearNoBias(channel_a, channel_atom, init="default")

        self.atom_decoder = AtomTransformer(
            channel_a=channel_atom,
            channel_s=channel_atom,
            channel_z=channel_atompair,
            num_blocks=num_blocks,
            num_heads=num_heads,
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        self.linear_q_to_r = nn.Sequential(
            LayerNorm(channel_atom, bias=False),
            LinearNoBias(channel_atom, 3, init="final"),
        )

    def forward(
        self,
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        f_input: FoldingInput,
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
        N = a.shape[1]  # number of diffusion samples
        token_index = f_input.atom.token_index  # [B, La]
        mask = f_input.atom.pad_mask  # [B, 1, La]

        a_to_q = self.linear_a_to_q(a)  # [B, N, Lt, c_atom]
        a_to_q = broadcast_tokens_to_atoms(
            a_to_q,  # [B, N, Lt, c_atom]
            token_index[..., None, :].expand(-1, N, -1),  # [B, N, La]
        )  # [B, N, La, c_atom]
        q = q_skip + a_to_q  # [B, N, La, c_atom]

        q = self.atom_decoder(
            q=q,  # [B, N, La, c_atom]
            c=c_skip,  # [B, N, La, c_atom]
            p=p_skip,  # [B, N, W, Lq, Lk, c_atompair]
            mask=mask[..., None, :],  # [B, 1, La], broadcasted to [B, N, La]
        )

        r_update = self.linear_q_to_r(q)
        return r_update
