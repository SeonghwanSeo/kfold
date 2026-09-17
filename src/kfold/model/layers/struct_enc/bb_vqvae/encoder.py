# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
# Copyright 2026 Chan Zuckerberg Biohub, Inc.
#
# K-Fold modifications:
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
#
# Modified from https://github.com/biohub/esm, MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import torch
import torch.nn as nn

from .affine_utils import Affine3D, build_affine3d_from_coordinates
from .transformer import GeometricEncoderStack, RelativePositionEmbedding


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)
    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[tuple(ranges)]


def node_gather(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return batched_gather(s.unsqueeze(-3), edges, -2, no_batch_dims=len(s.shape) - 1)


def knn_graph(
    coords: torch.Tensor,
    mask: torch.Tensor,
    no_knn: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """From ESM3 codebase"""
    L = coords.shape[-2]
    num_by_dist = min(no_knn, L)
    device = coords.device

    coords = coords.nan_to_num()
    dist_mask = ~(mask[..., None, :] & mask[..., :, None])
    dists = (coords.unsqueeze(-2) - coords.unsqueeze(-3)).norm(dim=-1)
    arange = torch.arange(L, device=device)
    seq_dists = (arange.unsqueeze(-1) - arange.unsqueeze(-2)).abs()
    max_dist = 1e6
    torch._assert_async((dists[~dist_mask] < max_dist).all())
    struct_then_seq_dist = (
        seq_dists.to(dists.dtype)
        .mul(1e2)
        .add(max_dist)
        .where(dist_mask, dists)
        .masked_fill(dist_mask, torch.inf)
    )
    dists, edges = struct_then_seq_dist.sort(dim=-1, descending=False)
    chosen_edges = edges[..., :num_by_dist]
    chosen_mask = dists[..., :num_by_dist].isfinite()
    return chosen_edges, chosen_mask


class BackboneEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int,
        n_layers: int,
        d_out: int,
    ):
        super().__init__()
        self.transformer = GeometricEncoderStack(d_model, n_heads, v_heads, n_layers)
        self.pre_vq_proj = nn.Linear(d_model, d_out)
        self.relative_positional_embedding = RelativePositionEmbedding(32, d_model)
        self.knn = 16

    def forward(
        self,
        coords: torch.Tensor,
        res_idx: torch.Tensor | None = None,
    ):
        # Extract N, CA, C coordinates for geometric reasoning
        affine, affine_mask = build_affine3d_from_coordinates(coords)
        ca_coords = coords[:, :, 1, :]
        z = self.encode_local_structure(ca_coords, affine, affine_mask, res_idx)
        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)
        z = self.pre_vq_proj(z)
        return z

    def encode_local_structure(
        self,
        ca_coords: torch.Tensor,
        affine: Affine3D,
        mask: torch.Tensor,
        res_idx: torch.Tensor | None = None,
    ):
        """Encodes local structure using a geometric transformer with KNN attention.

        Parameters
        ----------
        ca_coords: torch.Tensor
            Tensor of shape (B, L, 3) containing CA coordinates for each residue.
        affine: Affine3D
            Affine3D object containing rotation and translation information for each
            residue.
        mask: torch.Tensor
            Mask tensor of shape (B, L) indicating valid residues.
        res_idx: torch.Tensor | None
            Optional tensor of shape (B, L) containing residue indices for relative
            positional embedding.
            If None, will use the KNN edge indices as a proxy for residue indices.
        """
        with (
            torch.no_grad(),
            torch.autocast(device_type=ca_coords.device.type, enabled=False),
        ):
            knn_edges, knn_edge_mask = self.find_knn_edges(
                ca_coords, mask=mask, no_knn=self.knn
            )
            B, L, E = knn_edges.shape
            knn_edge_mask = knn_edge_mask.view(-1, E)
            affine_tensor = affine.tensor
            T_D = affine_tensor.size(-1)
            knn_affine_tensor = node_gather(affine_tensor, knn_edges)
            knn_affine_tensor = knn_affine_tensor.view(-1, E, T_D).contiguous()
            affine = Affine3D.from_tensor(knn_affine_tensor)
            knn_mask = node_gather(mask.unsqueeze(-1), knn_edges).view(-1, E)
            knn_mask = torch.logical_and(knn_mask, knn_edge_mask)
            if res_idx is None:
                pos_i = knn_edges.view(-1, E)
            else:
                pos_i = node_gather(res_idx.unsqueeze(-1), knn_edges).view(-1, E)

        z = self.relative_positional_embedding(pos_i[:, 0], pos_i)

        z, _ = self.transformer(
            x=z,
            attention_mask=knn_mask,
            affine=affine,
            affine_mask=knn_mask,
        )
        z = z.view(B, L, E, -1)
        z = z[:, :, 0, :]
        return z

    @staticmethod
    def find_knn_edges(coords: torch.Tensor, mask: torch.Tensor, no_knn: int):
        with torch.autocast(device_type=coords.device.type, enabled=False):
            edges, edge_mask = knn_graph(coords, mask, no_knn)
        return edges, edge_mask
