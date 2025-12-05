import math

import torch
import torch.nn as nn

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.embeddings import AtomEmbedding
from kfold.model.layers.alphafold3.transformers import AtomTransformer
from kfold.model.layers.alphafold3.utils import (
    LocalAttentionIndex,
    aggregate_atoms_to_tokens,
    broadcast_tokens_to_atoms,
)
from kfold.model.layers.primitives import LayerNorm, LinearNoBias


class AtomAttentionEncoderWithApo(nn.Module):
    """Atom attention encoder with apo embedding."""

    def __init__(
        self,
        channel_s: int,  # 384 in AF3
        channel_z: int | None,  # 128 in AF3
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
        channel_z : int | None
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

        # Embeddings atom features `c`
        self.embed_atom = AtomEmbedding(channel_atom)

        # Embeddings for atom pair features `p`
        # Reference position embeddings
        self.embed_ref_offset = LinearNoBias(3, channel_atompair, init="default")
        self.embed_ref_inv_dist = LinearNoBias(1, channel_atompair, init="default")
        self.embed_ref_mask = LinearNoBias(1, channel_atompair, init="default")
        # Apo position embeddings
        self.embed_apo_offset = LinearNoBias(3, channel_atompair, init="default")
        self.embed_apo_inv_dist = LinearNoBias(1, channel_atompair, init="default")
        self.embed_apo_mask = LinearNoBias(1, channel_atompair, init="default")

        self.use_structure = use_structure
        if use_structure:
            assert channel_z is not None, (
                "channel_z must be provided if use_structure is True"
            )
            self.linear_s_to_c = nn.Sequential(
                LayerNorm(channel_s, create_offset=False),
                LinearNoBias(channel_s, channel_atom, init="final"),
            )
            self.linear_z_to_p = nn.Sequential(
                LayerNorm(channel_z, create_offset=False),
                LinearNoBias(channel_z, channel_atompair, init="final"),
            )
            self.linear_r_to_q = LinearNoBias(
                channel_coords, channel_atom, init="default"
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
        z_trunk: torch.Tensor | None,
        r_noisy: torch.Tensor | None,
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
            The folding input. (batch size B)
        s_trunk : torch.Tensor | None
            The trunk single representation, shape [B, N, Lt, c_s].
            where Nsample is the number of diffusion samples.
        z_trunk : torch.Tensor | None
            The conditioning pair representation, shape [B, N, Lt, c_z].
        r_noisy : torch.Tensor | None
            The noised structures' positions, shape [B, N, La, c_r],
        model_cache : dict | None
            The model cache for storing intermediate representations, by default None.

        Returns
        -------
        a : torch.Tensor
            The token single representation
            Shape: [B, Lt, c_token] or [B, N, Lt, c_token]
        q_skip : torch.Tensor
            The atom single representation
            shape [B, La, c_atom] or [B, N, La, c_atom]
        c_skip : torch.Tensor
            The atom single conditioning
            shape [B, La, c_atom] or [B, N, La, c_atom]
        p_skip : torch.Tensor
            The atom pair representation
            shape [B, W, Lq, Lk, c_atompair] or [B, N, W, Lq, Lk, c_atompair]
        """
        if model_cache is not None:
            assert self.use_structure, "Caching is only supported when using structure."
            cache_prefix = "atom_attn_encoder"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]
        else:
            layer_cache = {}

        local_attn_index = LocalAttentionIndex(
            num_atoms=f_input.num_atoms,
            atoms_per_window_queries=self.atoms_per_window_queries,
            atoms_per_window_keys=self.atoms_per_window_keys,
            device=f_input.device,
        )
        if "qcp" in layer_cache:
            q, c, p = layer_cache["qcp"]
        else:
            q, c, p = self.initialize_atom_representation(f_input, local_attn_index)
            layer_cache["qcp"] = (q, c, p)

        # Shapes at this point:
        # q: [B, La, c_atom]
        # c: [B, La, c_atom]
        # p: [B, W, Lq, Lk, c_atompair]

        mask = f_input.atom.pad_mask  # [B, La]
        token_index = f_input.atom.token_index  # [B, La]

        # Add trunk embedding and noise position
        if self.use_structure:
            assert s_trunk is not None and z_trunk is not None and r_noisy is not None
            # Broadcast for multiple diffusion samples
            # [B, ...] -> [B, N, ...]
            q = q.unsqueeze(1)  # [B, 1, La, c_atom]
            c = c.unsqueeze(1)  # [B, 1, La, c_atom]
            p = p.unsqueeze(1)  # [B, 1, W, Lq, Lk, c_atompair]
            mask = mask.unsqueeze(1)  # [B, 1, La]
            token_index = token_index.unsqueeze(1)  # [B, 1, La]

            c = self.add_trunk_single_conditioning(c, s_trunk, token_index, mask)
            p = self.add_trunk_pair_embedding(
                p, z_trunk, token_index, mask, local_attn_index
            )
            q = self.add_noise_position(q, r_noisy)

        # Add atom-wise contributions to pair representation
        c_q = local_attn_index.to_query(c)  # [B, *, W, Lq, c_atom]
        c_k = local_attn_index.to_key(c)  # [B, *, W, Lk, c_atom]
        p = p + self.linear_query(c_q)[..., :, None, :]
        p = p + self.linear_key(c_k)[..., None, :, :]
        p = p + self.mlp_pair(p)  # [B, *, W, Lq, Lk, c_atompair]

        # Run Atom Transformer
        q = self.atom_encoder(q, c, p, mask)

        # Aggregate atom representations to token representations
        # [B, *, La, c_atom] -> [B, *, Lt, c_token]
        q_to_a = self.linear_q_to_a(q)  # [B, *, La, c_token]
        a = aggregate_atoms_to_tokens(
            q_to_a,  # [*, La, c_token]
            token_index=token_index,  # [*, La]
            num_tokens=f_input.num_tokens,
            atom_mask=mask,  # [*, La]
            aggr="mean",
        )

        q_skip, c_skip, p_skip = q, c, p

        return a, q_skip, c_skip, p_skip

    def initialize_atom_representation(
        self,
        f_input: FoldingInput,
        local_attn_index: LocalAttentionIndex,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize atom representations.

        Parameters
        ----------
        f_input : FoldingInput
            The folding input.
        local_attn_index : LocalAttentionIndex
            The local attention indexer for atom attention.

        Returns
        -------
        q : torch.Tensor
            The atom single representation, shape [B, La, c_atom].
        c : torch.Tensor
            The atom single conditioning, shape [B, La, c_atom].
        p : torch.Tensor
            The atom pair representation, shape [B, W, Lq, Lk, c_atompair].
        """
        # === Initialize single conditioning === #
        c = self.embed_atom(f_input)  # [B, La, c_atom]

        # === Initialize atom single representation === #
        q = c  # [B, La, c_atom]

        # === Initialize pair representation === #
        # 1. Embed Reference conformers (residue-level pairwise embeddings)
        # Mask with residue identity
        uid = f_input.atom.ref_space_uid
        uid_q, uid_k = local_attn_index.to_qk(uid, dim=-1)
        v_ref = uid_q[..., :, None] == uid_k[..., None, :]  # [B, W, Lq, Lk]
        v_ref = v_ref.to(dtype=c.dtype).unsqueeze(-1)  # [B, W, Lq, Lk, 1]

        # Shape: [B, W, Lq], [B, W, Lk]
        ref_pos = f_input.atom.ref_pos  # [B, La, 3]
        ref_pos_q = local_attn_index.to_query(ref_pos)  # [B, W, Lq, 3]
        ref_pos_k = local_attn_index.to_key(ref_pos)  # [B, W, Lk, 3]

        # Shape: [B, W, Lq, Lk, 3], [B, W, Lq, Lk, 1]
        ref_d_offset = ref_pos_q[..., :, None, :] - ref_pos_k[..., None, :, :]
        ref_dsq_inv = 1.0 / (1.0 + ref_d_offset.pow(2).sum(-1, keepdim=True))

        # Shape: [B, W, Lq, Lk, c_atompair]
        p_ref = self.embed_ref_offset(ref_d_offset)
        p_ref = p_ref + self.embed_ref_inv_dist(ref_dsq_inv)
        p_ref = p_ref + self.embed_ref_mask(v_ref)
        p_ref = p_ref * v_ref

        # 2. Embed Apo chain structures (chain-level pairwise embeddings)
        # Mask with chain identity
        asym_id = broadcast_tokens_to_atoms(
            f_input.token.asym_id.unsqueeze(-1), f_input.atom.token_index
        ).squeeze(-1)  # [B, La]
        asym_id_q, asym_id_k = local_attn_index.to_qk(asym_id, dim=-1)
        v_apo = asym_id_q[..., :, None] == asym_id_k[..., None, :]  # [B, W, Lq, Lk]

        # Mask unresolved apo atoms (this doesn't mean unresolved atoms in holo)
        apo_mask = f_input.atom.apo_mask  # [B, La]
        apo_mask_q, apo_mask_k = local_attn_index.to_qk(apo_mask, dim=-1)
        v_apo &= apo_mask_q[..., :, None] & apo_mask_k[..., None, :]  # [B, W, Lq, Lk]

        # Final apo mask
        v_apo = v_apo.to(c.dtype).unsqueeze(-1)  # [B, W, Lq, Lk, 1]

        with torch.autocast(c.device.type, enabled=False):
            # NOTE: (SeonghwanSeo) Since apo structure is much larger than ref_pos,
            # d_inv is adopted instead of d_inv_sq for better representation.
            apo_pos = f_input.atom.apo_coords  # [B, La, 3]
            # Shape: [B, La, 3] -> [B, W, Lq, 3], [B, W, Lk, 3]
            apo_pos_q, apo_pos_k = local_attn_index.to_qk(apo_pos)
            # Shape: [B, W, Lq, Lk, 3], [B, W, Lq, Lk, 1]
            apo_d_offset = apo_pos_q[..., :, None, :] - apo_pos_k[..., None, :, :]
            apo_d_inv = 1.0 / (1.0 + apo_d_offset.norm(dim=-1, keepdim=True))

        # Shape: [B, W, Lq, Lk, c_atompair]
        p_apo = self.embed_apo_offset(apo_d_offset)
        p_apo = p_apo + self.embed_apo_inv_dist(apo_d_inv)
        p_apo = p_apo + self.embed_apo_mask(v_apo)
        p_apo = p_apo * v_apo

        # 3. Combine reference and apo position embeddings
        p = p_ref + p_apo  # [B, W, Lq, Lk, c_atompair]

        # 4. Masking with padding mask
        mask_q, mask_k = local_attn_index.to_qk(f_input.atom.pad_mask, dim=-1)
        pair_mask = mask_q[..., :, None] & mask_k[..., None, :]

        p = p * pair_mask[..., None]  # [B, W, Lq, Lk, c_atompair]

        return q, c, p

    def add_trunk_single_conditioning(
        self,
        c: torch.Tensor,
        s_trunk: torch.Tensor,
        token_index: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Add trunk single embedding to atom single conditioning.

        Parameters
        ----------
        c : torch.Tensor
            The atom single conditioning, shape [*, La, c_atom].
        s_trunk : torch.Tensor
            The trunk single representation, shape [*, Lt, c_s].
        token_index: torch.Tensor
            The atom to token mapping, shape [*, La].
        atom_mask : torch.Tensor
            The atom padding mask, shape [*, La]
        """
        # Case 2
        s_trunk = self.linear_s_to_c(s_trunk)  # [*, Lt, c_atom]
        s_to_c = broadcast_tokens_to_atoms(s_trunk, token_index)  # [*, La, c_atom]
        s_to_c = s_to_c * atom_mask.unsqueeze(-1)  # [*, La, c_atom]
        return c + s_to_c  # [*, La, c_atom]

    def add_trunk_pair_embedding(
        self,
        p: torch.Tensor,
        z_trunk: torch.Tensor,
        token_index: torch.Tensor,
        atom_mask: torch.Tensor,
        local_attn_index: LocalAttentionIndex,
    ) -> torch.Tensor:
        """Add trunk pair embedding to atom pair representation.

        Parameters
        ----------
        p : torch.Tensor
            The atom pair representation, shape [*, W, Lq, Lk, c_atompair].
        z_trunk : torch.Tensor
            The trunk pair representation, shape [*, Lt, Lt, c_z].
        token_index : torch.Tensor
            The atom to token mapping, shape [*, La].
        atom_mask : torch.Tensor
            The atom padding mask, shape [*, La].
        local_attn_index : LocalAttentionIndex
            The local attention indexer for atom attention.
        """
        batch_shape = z_trunk.shape[:-3]  # [*]
        W, Lq, Lk = p.shape[-4:-1]

        # 1. Project Trunk features
        # [*, Lt, Lt, c_z] -> [*, Lt, Lt, c_atompair]
        z_trunk = self.linear_z_to_p(z_trunk)

        # 2. Get Windowed Indices
        # [*, La] -> [*, W, Lq], [*, W, Lk]
        idx_q, idx_k = local_attn_index.to_qk(token_index, dim=-1)
        idx_q = idx_q.expand(*batch_shape, W, Lq)
        idx_k = idx_k.expand(*batch_shape, W, Lk)

        # NOTE: safe indexing: Although the pad value of token_index is 0,
        # we clamp indices to be at least 0 to avoid run-time error.
        idx_q, idx_k = idx_q.clamp(min=0), idx_k.clamp(min=0)

        # 3. Token pair embedding to atom pair representation
        # [*, Ntoken, c_atom_pair] -> [*, W, Lq, Lk, c_atompair]
        B = math.prod(batch_shape)
        z_trunk = z_trunk.flatten(0, -4)  # [B, Lt, Lt, c_atompair]
        batch_indices = torch.arange(B, device=p.device)
        z_to_p = z_trunk[
            batch_indices.view(B, 1, 1, 1),  # [B, 1, 1, 1]
            idx_q.view(B, W, Lq, 1),
            idx_k.view(B, W, 1, Lk),
        ].unflatten(0, batch_shape)  # [*, W, Lq, Lk, c_atompair]

        # 5. Apply Padding Mask
        mask_q, mask_k = local_attn_index.to_qk(atom_mask, dim=-1)  # [*, W, Lq|Lk]
        pair_mask = mask_q[..., :, None] & mask_k[..., None, :]  # [*, W, Lq, Lk]
        z_to_p = z_to_p * pair_mask[..., None]  # [*, W, Lq, Lk, c_atompair]

        return p + z_to_p

    def add_noise_position(
        self,
        q: torch.Tensor,
        r_noisy: torch.Tensor,
    ) -> torch.Tensor:
        """Algorithm 5, Line 11
        Add noise position to atom single representation.

        Parameters
        ----------
        q : torch.Tensor
            The atom single representation, shape [*, La, c_atom].
        r_noisy : torch.Tensor
            The noised structures' positions, shape [*, La, 3].
        """
        with torch.autocast(q.device.type, enabled=False):
            assert r_noisy.dtype == torch.float32, "r_noisy must be float32"
            r_to_q = self.linear_r_to_q(r_noisy)  # [*, La, c_atom]
        return q + r_to_q  # [*, La, c_atom]
