"""
Code adopted from La-Proteina (https://github.com/NVIDIA-Digital-Bio/la-proteina).
"""

import torch
from torch import nn

from .atom_triangular_update import TriangleMultiplicativeUpdate


class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super().__init__()

        self.c_z = c_z
        self.n = n

        self.layer_norm = nn.LayerNorm(self.c_z)
        self.linear_1 = nn.Linear(self.c_z, self.n * self.c_z)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.n * self.c_z, self.c_z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        z = self.layer_norm(z)
        z = self.linear_1(z)
        z = self.relu(z)
        z = self.linear_2(z)
        return z


class PairReprUpdate(nn.Module):
    """Layer to update the pair representation."""

    def __init__(
        self,
        token_dim,
        pair_dim,
        expansion_factor_transition=2,
        use_tri_mult=False,
        tri_mult_c=196,
    ):
        super().__init__()
        self.use_tri_mult = use_tri_mult
        self.layer_norm_in = torch.nn.LayerNorm(token_dim)
        self.linear_x = torch.nn.Linear(token_dim, int(2 * pair_dim), bias=False)

        if use_tri_mult:
            tri_mult_c = min(pair_dim, tri_mult_c)
            self.tri_mult_out = TriangleMultiplicativeUpdate(
                c_z=pair_dim, c_hidden=tri_mult_c, _outgoing=True
            )
            self.tri_mult_in = TriangleMultiplicativeUpdate(
                c_z=pair_dim, c_hidden=tri_mult_c, _outgoing=False
            )
        self.transition_out = PairTransition(c_z=pair_dim, n=expansion_factor_transition)

    def forward(self, x, pair_rep, mask):
        """
        Args:
            x: Input sequence, shape [B, L, A, token_dim]
            pair_rep: Input pair representation, shape [B, L, L, A, pair_dim]
            mask: binary mask, shape [B, L, A]

        Returns:
            Updated pair representation, shape [B, L, A, A, pair_dim].
        """
        x_proj_1, x_proj_2 = self.linear_x(self.layer_norm_in(x)).chunk(2, dim=-1)
        pair_rep = pair_rep + x_proj_1[:, :, None, :, :] + x_proj_2[:, :, :, None, :]
        if self.use_tri_mult:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(3)  # [B, L, A, A]
            pair_mask = pair_mask.to(pair_rep.dtype)
            pair_rep = pair_rep + self.tri_mult_out(pair_rep, pair_mask)
            pair_rep = pair_rep + self.tri_mult_in(pair_rep, pair_mask)
        pair_rep = pair_rep + self.transition_out(pair_rep)
        return pair_rep
