import math
from collections.abc import Callable

import torch
import torch.nn as nn

from kfold.data.types.model_input import FoldingInput
from kfold.model.primitives import LayerNorm, LinearNoBias

from .diffusion_transformer import LocalTransformerStack
from .utils import (
    aggregate_atoms_to_tokens,
    broadcast_tokens_to_atoms,
    build_atom_to_qk_fn,
)


class AtomEmbedder(nn.Module):
    """Input embedding module for atom attention.
    See Section 3.7 Algorithm 5 AtomAttentionEncoder in the AF3 paper.
    """

    def __init__(
        self,
        channel_z: int | None,
        channel_atom: int,
        channel_atompair: int,
        use_structure: bool = False,
    ):
        """Initialize the atom attention encoder.

        Parameters
        ----------
        channel_z : int | None
            The pair representation dimension.
        channel_atom : int
            The atom single representation dimension.
        channel_atompair : int
            The atom pair representation dimension.
        use_structure : bool, optional
            Whether to use structure information, by default True.

        """
        super().__init__()
        # Embeddings atom features `c`
        self.embed_atom_pos = LinearNoBias(3, channel_atom, init="default", precision=32)
        self.embed_atom_charge = LinearNoBias(1, channel_atom, init="default")
        self.embed_atom_mask = LinearNoBias(1, channel_atom, init="default")
        self.embed_atom_element = LinearNoBias(128, channel_atom, init="default")
        self.embed_atom_name_chars = LinearNoBias(4 * 64, channel_atom, init="default")

        # Embeddings for atom pair features `p`
        # Reference position embeddings
        self.embed_ref_offset = LinearNoBias(
            3, channel_atompair, init="default", precision=32
        )
        self.embed_ref_inv_dist = LinearNoBias(1, channel_atompair, init="default")
        self.embed_ref_mask = LinearNoBias(1, channel_atompair, init="default")

        self.use_structure: bool = use_structure

        if use_structure:
            assert channel_z is not None, (
                "channel_z must be provided if use_structure is True"
            )
            self.linear_z_to_p = nn.Sequential(
                LayerNorm(channel_z, create_offset=False),
                LinearNoBias(channel_z, channel_atompair, init="final", precision=32),
            )
        else:
            assert channel_z is None, "channel_z must be None if use_structure is False"

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

    def forward(
        self,
        f_input: FoldingInput,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the atom attention input embedder.
        See Section 3.7 Algorithm 5 AtomAttentionEncoder

        q: Atom single representation
        c: Atom single conditioning
        p: Atom pair representation

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        z : torch.Tensor | None
            The pair conditioning, shape [B, Lt, c_z].

        Returns
        -------
        q : torch.Tensor
            The atom single representation
            shape [B, La, c_atom]
        c : torch.Tensor
            The atom single conditioning
            shape [B, La, c_atom]
        p : torch.Tensor
            The atom pair representation
            shape [B, W, Lq, Lk, c_atompair]
        """
        to_qk = build_atom_to_qk_fn(f_input.num_atoms, f_input.device)

        # Initialize single conditioning
        c = self.embed_atom(f_input)  # [B, La, c_atom]

        # Initialize pair representation
        # [B, W, Lq, Lk, c_atompair]
        p = self.embed_atom_pairs(f_input, to_qk)

        # Initialize atom single representation
        q = c  # [B, La, c_atom]

        # Add trunk embedding and noise position
        if self.use_structure:
            assert z is not None
            token_index = f_input.atom.token_index  # [B, La]
            p = p + self.get_trunk_pair_conditioning(z, token_index, to_qk)
        else:
            assert z is None

        # Add atom-wise contributions to pair representation
        c_q, c_k = to_qk(c, -2)  # [B, W, Lq|Lk, c_atom]
        p = p + self.linear_query(c_q)[..., :, None, :]
        p = p + self.linear_key(c_k)[..., None, :, :]
        p = p + self.mlp_pair(p)  # [*, W, Lq, Lk, c_atompair]

        return q, c, p

    def embed_atom(self, f_input: FoldingInput) -> torch.Tensor:
        c = self.embed_atom_pos(f_input.atom.ref_pos)
        c = c + self.embed_atom_charge(f_input.atom.ref_charge.unsqueeze(-1))
        c = c + self.embed_atom_mask(f_input.atom.pad_mask.float().unsqueeze(-1))
        c = c + self.embed_atom_element(f_input.atom.ref_element)
        c = c + self.embed_atom_name_chars(f_input.atom.ref_atom_name_chars.flatten(-2))
        return c

    def embed_atom_pairs(self, f_input: FoldingInput, to_qk: Callable) -> torch.Tensor:
        """Get atom pair representation from reference molecule conformer.

        Parameters
        ----------
        f_input : FoldingInputk
            The folding input.
        atom_to_query: tuple[torch.Tensor, torch.Tensor]
            The pre-computed atom to query indices for window attention.

        Returns
        -------
        p : torch.Tensor
            The atom pair representation, shape [B, W, Lq, Lk, c_atompair]
        """

        # Mask with residue identity
        uid = f_input.atom.ref_space_uid  # [B, La]
        uid_q, uid_k = to_qk(uid, dim=-1)  # [B, W, Lq|Lk]
        v = uid_q[..., :, None] == uid_k[..., None, :]  # [B, W, Lq, Lk]
        v = v.float().unsqueeze(-1)  # [B, W, Lq, Lk, 1]

        ref_pos = f_input.atom.ref_pos  # [B, La, 3]
        ref_pos_q, ref_pos_k = to_qk(ref_pos, dim=-2)  # [B, W, Lq|Lk, 3]
        # Shape: [B, W, Lq, Lk, 3], [B, W, Lq, Lk, 1]
        ref_d_offset = ref_pos_q[..., :, None, :] - ref_pos_k[..., None, :, :]
        ref_dsq_inv = 1.0 / (1.0 + ref_d_offset.pow(2).sum(-1))

        # Shape: [B, W, Lq, Lk, c_atompair]
        p = self.embed_ref_offset(ref_d_offset)
        p = p + self.embed_ref_inv_dist(ref_dsq_inv.unsqueeze(-1))
        p = p + self.embed_ref_mask(v)
        p = p * v
        return p

    def get_trunk_pair_conditioning(
        self,
        z: torch.Tensor,
        token_index: torch.Tensor,
        to_qk: Callable,
    ) -> torch.Tensor:
        """Add trunk pair conditioning to atom pair representation.
        Parameters
        ----------
        p: torch.Tensor
            The atom pair representation, shape [*, W, Lq, Lk, c_atompair].
        z: torch.Tensor
            The trunk pair representation, shape [*, Lt, Lt, c_z].
        token_index: torch.Tensor
            The token index for each atom, shape [*, La].

        Returns
        -------
        p: torch.Tensor
            The updated atom pair representation, shape [*, W, Lq, Lk, c_atompair].
        """
        batch_shape = z.shape[:-3]  # [*]
        B = math.prod(batch_shape)

        # 1. Project Trunk features
        # [*, Lt, Lt, c_z] -> [*, Lt, Lt, c_atompair]
        cond = self.linear_z_to_p(z.float())

        # 2. Token pair embedding to atom pair representation
        # [*, Ntoken, c_atom_pair] -> [*, W, Lq, Lk, c_atompair]
        b_idx = torch.arange(B, device=z.device)

        # NOTE: safe indexing: Although the pad value of token_index is 0,
        # we clamp indices to be at least 0 to avoid run-time error.
        token_index = token_index.view(B, -1)  # [B, La]
        token_index = token_index.clamp(min=0)  # [B, La]
        # [B, La] -> [B, W, Lq], [B, W, Lk]
        q_idx, k_idx = to_qk(token_index, dim=-1)

        cond = cond.flatten(0, -4)  # [B, Lt, Lt, c_atompair]
        cond = cond[
            b_idx.view(B, 1, 1, 1),
            q_idx[..., :, None],
            k_idx[..., None, :],
        ].unflatten(0, batch_shape)  # [*, W, Lq, Lk, c_atompair]
        return cond


class AtomAttentionEncoder(nn.Module):
    """Atom attention encoder
    See Section 3.7 Algorithm 5 AtomAttentionEncoder in the AF3 paper.
    """

    def __init__(
        self,
        channel_atom: int,  # 128 in AF3
        channel_atompair: int,  # 16 in AF3
        channel_token: int,  # 384 (InputEmbedder) or 768 (Diffusion) in AF3
        channel_coords: int = 3,
        num_blocks=3,
        num_heads=4,
        use_structure: bool = False,
    ):
        """Initialize the atom attention encoder.

        Parameters
        ----------
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
        use_structure : bool, optional
            Whether to use structure information, by default True.
        """
        super().__init__()
        self.use_structure: bool = use_structure
        if use_structure:
            self.linear_r_to_q = LinearNoBias(
                channel_coords, channel_atom, init="default", precision=32
            )
        self.transformer = LocalTransformerStack(
            channel_a=channel_atom,
            channel_s=channel_atom,
            channel_z=channel_atompair,
            num_blocks=num_blocks,
            num_heads=num_heads,
        )
        self.linear_q_to_a = LinearNoBias(channel_atom, channel_token, init="default")
        self.relu = nn.ReLU()

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        token_index: torch.Tensor,
        mask: torch.Tensor,
        num_tokens: int,
        r_noisy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the atom attention encoder.
        See Section 3.7 Algorithm 5 AtomAttentionEncoder in the AF3 paper.

        Parameters
        ----------
        q : torch.Tensor
            The atom single representation, shape [*, La, c_atom].
        c : torch.Tensor
            The atom single conditioning, shape [*, La, c_atom].
        p : torch.Tensor
            The atom pair conditioning, shape [*, W, Lq, Lk, c_atompair].
        token_index : torch.Tensor
            The token index for each atom, shape [*, La].
        mask : torch.Tensor
            The atom mask, shape [*, La].
        num_tokens : int
            The number of tokens in the input.
        r_noisy : torch.Tensor | None
            The noisy atom coordinates for diffusion, shape [*, N, La, 3].

        Returns
        -------
        a : torch.Tensor
            The token single representation, shape [*, Lt, c_token].
        q_skip : torch.Tensor
            The atom single representation before atom transformer, for skip connection.
        c_skip : torch.Tensor
            The atom single conditioning before atom transformer, for skip connection.
        p_skip : torch.Tensor
            The atom pair conditioning before atom transformer, for skip connection.
        """
        if self.use_structure:
            # Line 11: Add noise position
            assert r_noisy is not None, (
                "r_noisy must be provided if use_structure is True"
            )
            q = q + self.linear_r_to_q(r_noisy)
        else:
            assert r_noisy is None, "r_noisy must be None if use_structure is False"

        # Run Transformer
        q = self.transformer(q, c, p, mask)

        # Aggregate atom representations to token representations
        # [*, La, c_atom] -> [*, Lt, c_token]
        q_to_a = self.relu(self.linear_q_to_a(q))  # [*, La, c_token]
        a = aggregate_atoms_to_tokens(q_to_a, token_index, mask, num_tokens=num_tokens)

        # Save skip connections for decoder
        q_skip, c_skip, p_skip = q, c, p
        return a, q_skip, c_skip, p_skip


class AtomAttentionDecoder(nn.Module):
    """Atom attention decoder.
    Section 3.2 Algorithm 6 AtomAttentionDecoder
    """

    def __init__(
        self,
        channel_a: int,
        channel_atom: int,
        channel_atompair: int,
        num_blocks: int = 3,
        num_heads: int = 4,
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
        """
        super().__init__()

        self.linear_a_to_q = LinearNoBias(channel_a, channel_atom, init="default")
        self.transformer = LocalTransformerStack(
            channel_a=channel_atom,
            channel_s=channel_atom,
            channel_z=channel_atompair,
            num_blocks=num_blocks,
            num_heads=num_heads,
        )
        self.layernorm_q = LayerNorm(channel_atom, create_offset=False)
        self.linear_q_to_r = LinearNoBias(channel_atom, 3, init="final", precision=32)

    def forward(
        self,
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        token_index: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Forward pass of the atom attention decoder.
        See Algorithm 6 in the AF3 paper for more details.

        Parameters
        ----------
        a : torch.Tensor
            The token single representation, shape [*, Lt, c_token].
        q_skip : torch.Tensor
            The atom single representation, shape [*, La, c_atom].
        c_skip : torch.Tensor
            The atom single conditioning, shape [*, La, c_atom].
        p_skip : torch.Tensor
            The atom pair representation, shape [*, W, Lq, Lk, c_atompair].
        token_index : torch.Tensor
            The token index for each atom, shape [*, La].
        mask : torch.Tensor
            The atom mask, shape [*, La].

        Returns
        -------
        r_update : torch.Tensor
            The atom position updates, shape [*, La, 3].
        """
        # Convert token representation to atom representation
        a_to_q = self.linear_a_to_q(a)  # [*, Lt, c_atom]
        q = broadcast_tokens_to_atoms(a_to_q, token_index)  # -> [*, La, c_atom]
        # Skip-connection
        q = q + q_skip  # [*, La, c_atom]

        #  Run Transformer
        q = self.transformer(q, c_skip, p_skip, mask)

        # Project atom representation to updated coordinates
        r_update = self.linear_q_to_r(self.layernorm_q(q.float()))  # [*, N, La, 3]
        return r_update
