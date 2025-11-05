# Started from code from https://github.com/jwohlwend/boltz, MIT License
from functools import lru_cache, partial
from math import pi

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import Module, ModuleList

import kfold.constants as C
from kfold.data.model_input import FoldingInput

from ..layers import initialize as init
from ..layers.transition import Transition
from .transformers import AtomTransformer
from .utils import LinearNoBias


class FourierEmbedding(Module):
    """Fourier embedding layer."""

    def __init__(self, dim):
        """Initialize the Fourier Embeddings.

        Parameters
        ----------
        dim : int
            The dimension of the embeddings.

        """
        super().__init__()
        self.proj = nn.Linear(1, dim)
        torch.nn.init.normal_(self.proj.weight, mean=0, std=1)
        torch.nn.init.normal_(self.proj.bias, mean=0, std=1)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,
    ):
        times = rearrange(times, "b -> b 1")
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)


class RelativePositionEncoder(Module):
    """Relative position encoder."""

    def __init__(self, channel_z: int, r_max=32, s_max=2):
        """Initialize the relative position encoder.

        Parameters
        ----------
        channel_z : int
            The pair representation dimension.
        r_max : int, optional
            The maximum index distance, by default 32.
        s_max : int, optional
            The maximum chain distance, by default 2.

        """
        super().__init__()
        self.r_max: int = r_max
        self.s_max: int = s_max
        self.linear_layer = LinearNoBias(4 * (r_max + 1) + 2 * (s_max + 1) + 1, channel_z)

    def forward(self, input: FoldingInput) -> torch.Tensor:
        asym_id = input.token.asym_id
        entity_id = input.token.entity_id
        sym_id = input.token.sym_id
        residue_index = input.token.residue_index
        token_index = input.token.token_index
        cyclic_period = input.token.cyclic_period

        if torch.any(cyclic_period != 0):
            raise NotImplementedError("Cyclic periods not supported yet.")

        b_same_chain = torch.eq(asym_id[:, :, None], asym_id[:, None, :])
        b_same_residue = torch.eq(residue_index[:, :, None], residue_index[:, None, :])
        b_same_entity = torch.eq(entity_id[:, :, None], entity_id[:, None, :])
        rel_pos = residue_index[:, :, None] - residue_index[:, None, :]

        d_residue = torch.clip(rel_pos, min=-self.r_max, max=self.r_max)
        d_residue = d_residue + self.r_max  # [0, 2*r_max]
        d_residue = torch.where(
            b_same_chain, d_residue, torch.full_like(d_residue, 2 * self.r_max + 1)
        )
        a_rel_pos = F.one_hot(d_residue, 2 * self.r_max + 2)

        d_token = torch.clip(
            token_index[:, :, None] - token_index[:, None, :],
            min=-self.r_max,
            max=self.r_max,
        )
        d_token = d_token + self.r_max  # [0, 2*r_max]
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            torch.full_like(d_token, 2 * self.r_max + 1),
        )
        a_rel_token = F.one_hot(d_token, 2 * self.r_max + 2)

        d_chain = torch.clip(
            sym_id[:, :, None] - sym_id[:, None, :],
            min=-self.s_max,
            max=self.s_max,
        )
        d_chain = d_chain + self.s_max  # [0, 2*s_max]
        d_chain = torch.where(
            b_same_entity, d_chain, torch.full_like(d_chain, 2 * self.s_max + 1)
        )
        a_rel_chain = F.one_hot(d_chain, 2 * self.s_max + 2)

        p = self.linear_layer(
            torch.cat(
                [
                    a_rel_pos.float(),
                    a_rel_token.float(),
                    b_same_entity.unsqueeze(-1).float(),
                    a_rel_chain.float(),
                ],
                dim=-1,
            )
        )
        return p


class SingleConditioning(Module):
    """Single conditioning layer."""

    def __init__(
        self,
        sigma_data: float,
        channel_s=384,
        dim_fourier=256,
        num_transitions=2,
        transition_expansion_factor=2,
        eps=1e-20,
    ):
        """Initialize the single conditioning layer.

        Parameters
        ----------
        sigma_data : float
            The data sigma.
        channel_s : int, optional
            The single representation dimension, by default 384.
        dim_fourier : int, optional
            The fourier embeddings dimension, by default 256.
        num_transitions : int, optional
            The number of transitions layers, by default 2.
        transition_expansion_factor : int, optional
            The transition expansion factor, by default 2.
        eps : float, optional
            The epsilon value, by default 1e-20.

        """
        super().__init__()
        self.eps = eps
        self.sigma_data = sigma_data

        input_dim = 2 * channel_s + 2 * C.NUM_RES_TYPES + 1 + C.NUM_POCKET_CONTACT_TYPES
        self.norm_single = nn.LayerNorm(input_dim)
        self.single_embed = nn.Linear(input_dim, 2 * channel_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.fourier_to_single = LinearNoBias(dim_fourier, 2 * channel_s)

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(
                dim=2 * channel_s, hidden=transition_expansion_factor * 2 * channel_s
            )
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        *,
        times,
        s_trunk,
        s_inputs,
    ):
        s = torch.cat((s_trunk, s_inputs), dim=-1)
        s = self.single_embed(self.norm_single(s))
        fourier_embed = self.fourier_embed(times)
        normed_fourier = self.norm_fourier(fourier_embed)
        fourier_to_single = self.fourier_to_single(normed_fourier)

        s = rearrange(fourier_to_single, "b d -> b 1 d") + s

        for transition in self.transitions:
            s = transition(s) + s

        return s, normed_fourier


class PairwiseConditioning(Module):
    """Pairwise conditioning layer."""

    def __init__(
        self,
        channel_z,
        dim_token_rel_pos_feats,
        num_transitions=2,
        transition_expansion_factor=2,
    ):
        """Initialize the pairwise conditioning layer.

        Parameters
        ----------
        channel_z : int
            The pair representation dimension.
        dim_token_rel_pos_feats : int
            The token relative position features dimension.
        num_transitions : int, optional
            The number of transitions layers, by default 2.
        transition_expansion_factor : int, optional
            The transition expansion factor, by default 2.

        """
        super().__init__()

        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(channel_z + dim_token_rel_pos_feats),
            LinearNoBias(channel_z + dim_token_rel_pos_feats, channel_z),
        )

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(
                dim=channel_z, hidden=transition_expansion_factor * channel_z
            )
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        z_trunk,
        token_rel_pos_feats,
    ):
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)

        for transition in self.transitions:
            z = transition(z) + z

        return z


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
    single: torch.Tensor, indexing_matrix: torch.Tensor, W: int, H: int
) -> torch.Tensor:
    if single.ndim == 2:
        N, D = single.shape
        K = N // W
        single = single.view(2 * K, W // 2, D)
        return torch.einsum("j i d, j k -> k i d", single, indexing_matrix).reshape(
            K, H, D
        )
    elif single.ndim == 3:
        B, N, D = single.shape
        K = N // W
        single = single.view(B, 2 * K, W // 2, D)
        return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
            B, K, H, D
        )
    else:
        raise ValueError("Invalid single tensor shape.")


class AtomAttentionEncoder(Module):
    """Atom attention encoder."""

    def __init__(
        self,
        channel_atom: int,
        channel_atompair: int,
        channel_s: int,
        channel_z: int,
        atom_feature_dim: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        structure_prediction=True,
        activation_checkpointing=False,
    ):
        """Initialize the atom attention encoder.

        Parameters
        ----------
        channel_atom : int
            The atom single representation dimension.
        channel_atompair : int
            The atom pair representation dimension.
        channel_s : int
            The single representation dimension.
        channel_z : int
            The pair representation dimension.
        atom_feature_dim : int
            The atom feature dimension.
        atoms_per_window_queries : int
            The number of atoms per window for queries.
        atoms_per_window_keys : int
            The number of atoms per window for keys.
        atom_encoder_depth : int, optional
            The number of transformer layers, by default 3.
        atom_encoder_heads : int, optional
            The number of transformer heads, by default 4.
        structure_prediction : bool, optional
            Whether it is used in the diffusion module, by default True.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.

        """
        super().__init__()
        self.num_atom_elements = C.NUM_ATOM_ELEMENTS
        self.num_atom_name_chars = C.NUM_ATOM_NAME_CHARS
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys

        # Atom feature embeddings
        self.embed_atom_pos = LinearNoBias(3, channel_atom)
        self.embed_atom_element = nn.Embedding(self.num_atom_elements, channel_atom)
        self.embed_atom_charge = LinearNoBias(1, channel_atom)
        self.embed_atom_name = LinearNoBias(4 * self.num_atom_name_chars, channel_atom)

        self.embed_atompair_ref_pos = LinearNoBias(3, channel_atompair)
        self.embed_atompair_ref_dist = LinearNoBias(1, channel_atompair)
        self.embed_atompair_mask = LinearNoBias(1, channel_atompair)

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.s_to_c_trans = nn.Sequential(
                nn.LayerNorm(channel_s), LinearNoBias(channel_s, channel_atom)
            )
            init.final_init_(self.s_to_c_trans[1].weight)

            self.z_to_p_trans = nn.Sequential(
                nn.LayerNorm(channel_z), LinearNoBias(channel_z, channel_atompair)
            )
            init.final_init_(self.z_to_p_trans[1].weight)

            self.r_to_q_trans = LinearNoBias(10, channel_atom)
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
            dim=channel_atom,
            dim_single_cond=channel_atom,
            dim_pairwise=channel_atompair,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(
                channel_atom, 2 * channel_s if structure_prediction else channel_s
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        input: FoldingInput,
        s_trunk: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
        multiplicity: int = 1,
        model_cache: dict | None = None,
    ):
        Na = len(input.atom)
        Nt = len(input.token)
        atom_mask = input.atom.pad_mask

        layer_cache = None
        if model_cache is not None:
            cache_prefix = "atomencoder"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]

        if model_cache is None or len(layer_cache) == 0:  # type: ignore
            # either model is not using the cache or it is the first time running it

            dtype = input.atom.ref_pos.dtype

            atom_feats = self.embed_atom_pos(input.atom.ref_pos)
            atom_feats = atom_feats + self.embed_atom_element(input.atom.ref_element)
            atom_feats = atom_feats + self.embed_atom_charge(
                input.atom.ref_charge.to(dtype).unsqueeze(-1)
            )
            atom_feats = atom_feats + self.embed_atom_name(
                F.one_hot(input.atom.ref_atom_name_chars, self.num_atom_name_chars)
                .to(dtype)
                .flatten(-2)
            )

            c = atom_feats  # [Na, channel_channel_atom]
            atom_ref_pos = input.atom.ref_pos.unsqueeze(0)  # [Na, 3]
            atom_uid = input.atom.ref_space_uid.unsqueeze(0)  # [Na]

            # NOTE: we are already creating the windows to make it more efficient
            W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
            Na = c.shape[:2]
            K = Na // W
            keys_indexing_matrix = get_indexing_matrix(K, W, H, c.device)
            to_keys = partial(
                single_to_keys, indexing_matrix=keys_indexing_matrix, W=W, H=H
            )

            atom_ref_pos_queries = atom_ref_pos.view(K, W, 1, 3)
            atom_ref_pos_keys = to_keys(atom_ref_pos).view(K, 1, H, 3)

            d = atom_ref_pos_keys - atom_ref_pos_queries  # [K, W, H, 3]
            d_norm = torch.sum(d * d, dim=-1, keepdim=True)
            d_norm = 1 / (1 + d_norm)  # [K, W, H, 1]

            # Create attention mask based on atom uids
            atom_mask_queries = atom_mask.view(K, W, 1)
            atom_mask_keys = to_keys(atom_mask.unsqueeze(-1).float()).view(K, 1, H).bool()
            atom_uid_queries = atom_uid.view(K, W, 1)
            atom_uid_keys = to_keys(atom_uid.unsqueeze(-1).float()).view(K, 1, H).long()
            v = (
                (atom_mask_queries & atom_mask_keys & (atom_uid_queries == atom_uid_keys))
                .float()
                .unsqueeze(-1)
            )  # [K, W, H, 1]

            p = self.embed_atompair_ref_pos(d) * v
            p = p + self.embed_atompair_ref_dist(d_norm) * v
            p = p + self.embed_atompair_mask(v) * v  # [K, W, H, channel_atompair]

            q = c

            if self.structure_prediction:
                # run only in structure model not in initial encoding
                s_to_c = self.s_to_c_trans(s_trunk)
                s_to_c = torch.bmm(input.atom_to_token, s_to_c)
                c = c + s_to_c

                atom_to_token_queries = input.atom_to_token.view(K, W, Nt)
                atom_to_token_keys = to_keys(input.atom_to_token)
                z_to_p = self.z_to_p_trans(z)
                z_to_p = torch.einsum(
                    "ijd,wki,wlj->wkld",
                    z_to_p,
                    atom_to_token_queries,
                    atom_to_token_keys,
                )
                p = p + z_to_p

            p = p + self.c_to_p_trans_q(c.view(K, W, 1, c.shape[-1]))
            p = p + self.c_to_p_trans_k(to_keys(c).view(K, 1, H, c.shape[-1]))
            p = p + self.p_mlp(p)  # [K, W, H, channel_atompair]

            if model_cache is not None:
                layer_cache["q"] = q
                layer_cache["c"] = c
                layer_cache["p"] = p
                layer_cache["to_keys"] = to_keys

        else:
            q = layer_cache["q"]
            c = layer_cache["c"]
            p = layer_cache["p"]
            to_keys = layer_cache["to_keys"]

        # Shapes:
        # q: [Na, channel_channel_atom]
        # c: [Na, channel_channel_atom]
        # p: [K, W, H, channel_atompair]

        if self.structure_prediction:
            # only here the multiplicity kicks in because we use the different positions r
            # NOTE: here we use multiple batches for different positions where
            # batch size = multiplicity (number of diffused samples per input)
            assert r is not None, "r must be provided for structure prediction"
            r_input = torch.cat(
                [r, torch.zeros((multiplicity, Na, 7), dtype=r.dtype, device=r.device)],
                dim=-1,
            )
            r_to_q = self.r_to_q_trans(r_input)
            q = q.unsqeeze(0) + r_to_q  # [B, Na, channel_channel_atom]
        else:
            q = q.unsqueeze(0)  # [1, Na, channel_channel_atom]
        B = q.shape[0]

        # Shapes:
        # q: [B, Na, channel_channel_atom]
        # c: [B, Na, channel_channel_atom]
        # p: [B, K, W, H, channel_atompair]

        c = c.unsqueeze(0).expand(B, -1, -1)  # [B, Na, channel_channel_atom]
        p = p.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, K, W, H, channel_atompair]
        atom_mask = atom_mask.unsqueeze(0).expand(B, -1)  # [B, Na]

        q = self.atom_encoder(
            q=q,
            mask=atom_mask,
            c=c,
            p=p,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=layer_cache,
        )

        q_to_a = self.atom_to_token_trans(q)
        atom_to_token = input.atom_to_token.float()
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        return a, q, c, p, to_keys


class AtomAttentionDecoder(Module):
    """Atom attention decoder."""

    def __init__(
        self,
        channel_s: int,
        channel_atom: int,
        channel_atompair: int,
        attn_window_queries,
        attn_window_keys,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
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
        attn_window_queries : int
            The number of atoms per window for queries.
        attn_window_keys : int
            The number of atoms per window for keys.
        atom_decoder_depth : int, optional
            The number of transformer layers, by default 3.
        atom_decoder_heads : int, optional
            The number of transformer heads, by default 4.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.

        """
        super().__init__()

        self.a_to_q_trans = LinearNoBias(2 * channel_s, channel_atom)
        init.final_init_(self.a_to_q_trans.weight)

        self.atom_decoder = AtomTransformer(
            dim=channel_atom,
            dim_single_cond=channel_atom,
            dim_pairwise=channel_atompair,
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
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
        input: FoldingInput,
        to_keys,
        multiplicity: int = 1,
        model_cache=None,
    ):
        atom_mask = input.atom.pad_mask  # [Na]

        a_to_q = self.a_to_q_trans(a)  # [Ndiff, Nt, channel_atom]
        a_to_q = torch.bmm(input.atom_to_token, a_to_q)  # [Ndiff, Na, channel_atom]
        q = q + a_to_q  # [Ndiff, Na, channel_atom]

        layer_cache = None
        if model_cache is not None:
            cache_prefix = "atomdecoder"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]

        q = self.atom_decoder(
            q=q,
            mask=atom_mask,
            c=c,
            p=p,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=layer_cache,
        )

        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update
