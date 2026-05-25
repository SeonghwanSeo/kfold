import torch
import torch.nn as nn
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput


def get_pair_mask(mask: torch.Tensor) -> torch.Tensor:
    """Get pair mask from token mask."""
    return mask[..., :, None] & mask[..., None, :]


def get_distogram(
    x: torch.Tensor,
    boundaries: torch.Tensor,
) -> torch.Tensor:
    """Compute the distogram from the input coordinates."""
    # Compute pairwise distances
    d = (x[..., None, :, :] - x[..., :, None, :]).norm(dim=-1)  # [..., L, L]
    num_bins = boundaries.shape[0] + 1
    distogram = F.one_hot(
        (d[..., None] > boundaries).sum(dim=-1), num_bins
    )  # [..., L, L, num_bins]
    return distogram


def express_coords_in_frames(
    x: torch.Tensor,
    frame: torch.Tensor,
) -> torch.Tensor:
    """Project coordinates `x` into the local frames.
    See Section 4.3.2 Algorithm 29 of the AlphaFold paper.
    """
    # Line 1
    a, b, c = frame.unbind(dim=-2)  # [..., L, 3]

    # Line 2
    w1 = a - b
    w1 /= w1.norm(dim=-1, keepdim=True) + 1e-8  # [... L, 3]

    # Line 3
    w2 = c - b
    w2 /= w2.norm(dim=-1, keepdim=True) + 1e-8  # [... L, 3]

    # Build orthogonal frame basis (e1, e2, e3)
    # Line 4
    e1 = w1 + w2
    e1 /= e1.norm(dim=-1, keepdim=True) + 1e-8  # [... L, 3]

    # Line 5
    e2 = w2 - w1
    e2 /= e2.norm(dim=-1, keepdim=True) + 1e-8  # [... L, 3]

    # Line 6
    e3 = torch.linalg.cross(e1, e2, dim=-1)  # [... L, 3]

    # Project onto frame basis
    # Line 7
    d = x.unsqueeze(-3) - b.unsqueeze(-2)  # [..., L, L, 3]

    # Line 8
    x_transformed = torch.stack(
        [
            torch.einsum("...id,...ijd->...ij", e1, d),
            torch.einsum("...id,...ijd->...ij", e2, d),
            torch.einsum("...id,...ijd->...ij", e3, d),
        ],
        dim=-1,
    )  # [..., L, L, 3]
    return x_transformed  # [..., L, L, 3]


class ApoEmbedding(nn.Module):
    def __init__(
        self,
        num_bins: int = 39,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        max_r: int = 64,
    ) -> None:
        super().__init__()
        # Featurization
        self.max_r = max_r
        self.num_bins = num_bins
        boundaries = torch.linspace(min_dist, max_dist, num_bins - 1)
        self.register_buffer("boundaries", boundaries, persistent=False)
        self.num_channels = self.num_bins + 1 + 3 + 1

    def forward(self, f_input: FoldingInput):
        """Embed the input features into the pairwise representation."""
        with torch.no_grad(), torch.autocast(f_input.device.type, enabled=False):
            return self.get_features(f_input)

    def get_features(self, f_input: FoldingInput) -> torch.Tensor:
        # Get distogram
        repr_coords = f_input.token.apo_repr_coords  # [B, L, 3]
        dgram = get_distogram(repr_coords, self.boundaries)  # [B, L, L, num_bins]
        dgram = dgram.float()  # Convert to float
        dgram_mask = get_pair_mask(f_input.token.apo_repr_mask)  # [B, L, L]
        dgram.masked_fill_(~dgram_mask[..., None], 0.0)

        # Get frame-based representation
        center_coords = f_input.token.apo_center_coords  # [B, L, 3]
        frame = f_input.token.apo_frame_coords  # [B, L, 3, 3]
        coords_in_frame = express_coords_in_frames(center_coords, frame)  # [B, L, L, 3]
        unit_vector = coords_in_frame / (
            coords_in_frame.norm(dim=-1, keepdim=True) + 1e-8
        )  # [B, L, L, 3]
        unit_vector_mask = get_pair_mask(f_input.token.apo_frame_mask)  # [B, L, L]
        unit_vector.masked_fill_(~unit_vector_mask[..., None], 0.0)

        feat = torch.cat(
            [
                dgram,
                dgram_mask[..., None].float(),
                unit_vector,
                unit_vector_mask[..., None].float(),
            ],
            dim=-1,
        )  # [B, L, L, num_bins + 1 + 3 + 1]

        # Get pair mask
        pair_mask = get_pair_mask(f_input.token.apo_repr_mask)  # [B, L, L]
        # Apo is only defined for intra-chain pairs.
        asym_id = f_input.token.asym_id  # [B, L]
        pair_mask &= asym_id[..., :, None] == asym_id[..., None, :]  # [B, L, L]
        # Mask out pairs that are too far in sequence
        res_idx = f_input.token.residue_index  # [B, L]
        pair_mask &= (res_idx[..., :, None] - res_idx[..., None, :]).abs() <= self.max_r

        feat.masked_fill_(~pair_mask[..., None], 0.0)
        return feat  # [B, L, L, num_bins + 1 + 3 + 1]
