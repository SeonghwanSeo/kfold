"""
Code adopted from La-Proteina (https://github.com/NVIDIA-Digital-Bio/la-proteina).
"""

import torch
import torch.nn as nn

from .tri_mul import TriangleMultiplicativeUpdate


class PairTransition(nn.Module):
    """Implements Algorithm 15."""

    def __init__(self, c_z: int, n: int):
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
        self.linear_2 = nn.Linear(self.n * self.c_z, c_z)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)

        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z)
        return z


class PairReprUpdate(torch.nn.Module):
    """Layer to update the pair representation."""

    def __init__(self, c_s: int, c_z: int):
        super().__init__()
        self.layer_norm_in = torch.nn.LayerNorm(c_s)
        self.linear_x = torch.nn.Linear(c_s, 2 * c_z, bias=False)
        self.tri_mult_out = TriangleMultiplicativeUpdate(c_z, _outgoing=True)
        self.tri_mult_in = TriangleMultiplicativeUpdate(c_z, _outgoing=False)
        self.transition_out = PairTransition(c_z, n=2)

    def forward(self, s, z, mask):
        """
        Args:
            s: Input sequence, shape [B, L, A, c_s]
            z: Input pair representation, shape [B, L, L, A, c_z]
            mask: binary mask, shape [B, L, A]

        Returns:
            Updated pair representation, shape [B, L, A, A, c_z].
        """
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(3)  # [B, L, A, A]
        pair_mask = pair_mask.to(z.dtype)

        si, sj = self.linear_x(self.layer_norm_in(s)).chunk(2, dim=-1)
        z = z + si[..., None, :, :] + sj[..., :, None, :]
        z = z + self.tri_mult_out(z, pair_mask)
        z = z + self.tri_mult_in(z, pair_mask)
        z = z + self.transition_out(z, pair_mask)
        return z
