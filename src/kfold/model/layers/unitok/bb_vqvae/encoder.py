import torch
import torch.nn as nn

from ..utils.esm_utils.affine3d import Affine3D, build_affine3d_from_coordinates
from .layers.transformer import (
    VanillaGeometricEncoderStack,
    VanillaRelativePositionEmbedding,
)


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


class VanillaStructureTokenEncoder(nn.Module):
    def __init__(self, d_model, n_heads, v_heads, n_layers, d_out, n_codes):
        super().__init__()
        self.transformer = VanillaGeometricEncoderStack(
            d_model, n_heads, v_heads, n_layers
        )
        self.pre_vq_proj = nn.Linear(d_model, d_out)
        self.relative_positional_embedding = VanillaRelativePositionEmbedding(
            32, d_model, init_std=0.02
        )
        self.knn = 16
        self.d_out = d_out

    def encode_local_structure(
        self,
        ca_coords: torch.Tensor,
        affine: Affine3D,
        mask: torch.Tensor,
        residue_index: torch.Tensor | None = None,
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
        residue_index: torch.Tensor | None
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
            if residue_index is None:
                res_idxs = knn_edges.view(-1, E)
            else:
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges).view(-1, E)

        z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs)

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

    def encode(self, coords: torch.Tensor, residue_index: torch.Tensor | None = None):
        # Extract N, CA, C coordinates for geometric reasoning
        affine, affine_mask = build_affine3d_from_coordinates(coords)
        ca_coords = coords[:, :, 1, :]
        z = self.encode_local_structure(ca_coords, affine, affine_mask, residue_index)
        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)
        z = self.pre_vq_proj(z)
        return z
