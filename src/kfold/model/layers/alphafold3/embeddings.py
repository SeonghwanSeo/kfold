# Started from code from https://github.com/jwohlwend/boltz, MIT License
import torch
import torch.nn.functional as F
from torch import nn

import kfold.constants as C
from kfold.data.model_input import FoldingInput

from .primitives import LinearNoBias


class RelativePositionEncoding(nn.Module):
    """Relative position encoder."""

    def __init__(self, channel_z: int, r_max: int = 32, s_max: int = 2):
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

    def forward(
        self,
        f_input: FoldingInput,
        model_cache: dict | None = None,
    ) -> torch.Tensor:
        """See Section 3.1.2 Algorithm 3: Relative position encoding in the AF3 paper."""
        # All shape: [B, Lt]
        asym_id = f_input.token.asym_id
        entity_id = f_input.token.entity_id
        sym_id = f_input.token.sym_id
        residue_index = f_input.token.residue_index
        token_index = f_input.token.token_index

        if model_cache is not None:
            cache_prefix = "relative_position_encoding"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]
        else:
            layer_cache = {}

        if len(layer_cache) == 0:
            with torch.no_grad():
                # Line 1
                b_same_chain = torch.eq(asym_id[:, :, None], asym_id[:, None, :])
                # Line 2
                b_same_residue = torch.eq(
                    residue_index[:, :, None], residue_index[:, None, :]
                )
                # Line 3
                b_same_entity = torch.eq(entity_id[:, :, None], entity_id[:, None, :])

                # Line 4
                d_residue = torch.clip(
                    residue_index[:, :, None] - residue_index[:, None, :] + self.r_max,
                    min=0,
                    max=2 * self.r_max,
                )
                d_residue = torch.where(
                    b_same_chain,
                    d_residue,
                    2 * self.r_max + 1,
                )
                # Line 5
                a_rel_pos = F.one_hot(d_residue, 2 * self.r_max + 2)

                # Line 6
                d_token = torch.clip(
                    token_index[:, :, None] - token_index[:, None, :] + self.r_max,
                    min=0,
                    max=2 * self.r_max,
                )
                d_token = torch.where(
                    b_same_chain & b_same_residue,
                    d_token,
                    2 * self.r_max + 1,
                )
                # Line 7
                a_rel_token = F.one_hot(d_token, 2 * self.r_max + 2)

                # Line 8
                d_chain = torch.clip(
                    sym_id[:, :, None] - sym_id[:, None, :] + self.s_max,
                    min=0,
                    max=2 * self.s_max,
                )
                # NOTE: (seonghwanseo) In the original paper and Boltz implementation,
                # it is written as b_same_chain.
                # However, it is implemented as b_same_entity according to AF3 official
                # implementation.
                d_chain = torch.where(
                    b_same_entity,
                    d_chain,
                    2 * self.s_max + 1,
                )
                # Line 9
                a_rel_chain = F.one_hot(d_chain, 2 * self.s_max + 2)

                # Line 10:1 (concat)
                rel_position_encoding = torch.cat(
                    [
                        a_rel_pos.float(),
                        a_rel_token.float(),
                        b_same_entity.unsqueeze(-1).float(),
                        a_rel_chain.float(),
                    ],
                    dim=-1,
                )
            layer_cache["rel_pos_encoding"] = rel_position_encoding
        else:
            rel_position_encoding = layer_cache["rel_pos_encoding"]

        # Line 10:2 (lienar)
        p = self.linear_layer(rel_position_encoding)  # [B, L, L, c_z]

        return p  # [L, L, c_z]


class AtomEmbedding(nn.Module):
    """Atom embedding.
    See Section 3.2 Algorithm 5 AtomAttentionEncoder: Line 1
    """

    def __init__(self, channel_atom: int):
        """Initialize the atom attention encoder.

        Parameters
        ----------
        channel_atom : int
            The atom single representation dimension.
        """
        super().__init__()
        self.num_atom_elements: int = C.NUM_ATOM_ELEMENTS
        self.num_atom_name_chars: int = C.NUM_ATOM_NAME_CHARS

        # Atom feature embeddings
        self.embed_atom_pos = LinearNoBias(3, channel_atom)
        self.embed_atom_charge = LinearNoBias(1, channel_atom)
        self.embed_atom_mask = LinearNoBias(1, channel_atom)
        self.embed_atom_element = nn.Embedding(self.num_atom_elements, channel_atom)
        self.embed_atom_name_chars = LinearNoBias(
            4 * self.num_atom_name_chars, channel_atom
        )

    def forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Embed atom features.
        Line 1:
        c = LinearNoBias(concat(ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars)))
        """  # noqa: E501

        atom_layout = f_input.atom
        ref_pos = atom_layout.ref_pos
        ref_charge = atom_layout.ref_charge
        ref_mask = atom_layout.pad_mask
        ref_element = atom_layout.ref_element
        ref_atom_name_chars = atom_layout.ref_atom_name_chars

        atom_feats = self.embed_atom_pos(ref_pos)
        atom_feats = atom_feats + self.embed_atom_charge(ref_charge.float().unsqueeze(-1))
        atom_feats = atom_feats + self.embed_atom_mask(ref_mask.float().unsqueeze(-1))
        atom_feats = atom_feats + self.embed_atom_element(ref_element)
        atom_feats = atom_feats + self.embed_atom_name_chars(
            F.one_hot(ref_atom_name_chars, self.num_atom_name_chars).flatten(-2).float()
        )
        return atom_feats
