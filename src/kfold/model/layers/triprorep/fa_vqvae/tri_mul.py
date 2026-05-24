"""
Code adopted from La-Proteina (https://github.com/NVIDIA-Digital-Bio/la-proteina).
"""

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn


def permute_final_dims(tensor: torch.Tensor, inds: Sequence[int]):
    num_first_dims = len(tensor.shape) - len(inds)
    first_inds = list(range(num_first_dims))
    return tensor.permute(first_inds + [num_first_dims + i for i in inds])


class BaseTriangleMultiplicativeUpdate(nn.Module, ABC):
    """
    Implements Algorithms 11 and 12.
    """

    @abstractmethod
    def __init__(self, c_z, _outgoing):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super().__init__()
        self.c_z = c_z
        self._outgoing = _outgoing

        self.linear_g = nn.Linear(self.c_z, self.c_z)
        self.linear_z = nn.Linear(self.c_z, self.c_z)

        self.layer_norm_in = nn.LayerNorm(self.c_z)
        self.layer_norm_out = nn.LayerNorm(self.c_z)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        p = torch.matmul(a, b)
        return permute_final_dims(p, (1, 2, 0))

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_atom, N_atom, C_z] input tensor
            mask:
                [*, N_atom, N_atom] input mask
        Returns:
            [*, N_atom, N_atom, C_z] output tensor
        """
        pass


class TriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements Algorithms 11 and 12.
    """

    def __init__(self, c_z, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super().__init__(c_z, _outgoing=_outgoing)

        self.linear_a_p = nn.Linear(self.c_z, self.c_z)
        self.linear_a_g = nn.Linear(self.c_z, self.c_z)
        self.linear_b_p = nn.Linear(self.c_z, self.c_z)
        self.linear_b_g = nn.Linear(self.c_z, self.c_z)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_atom, N_atom, C_z] input tensor
            mask:
                [*, N_atom, N_atom] input mask
        Returns:
            [*, N_atom, N_atom, C_z] output tensor
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask
        a = a * self.sigmoid(self.linear_a_g(z))
        a = a * self.linear_a_p(z)
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)

        # Prevents overflow of torch.matmul in combine projections in
        # reduced-precision modes
        x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x
