import math

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput


class RelativePositionEncoding(torch.nn.Module):
    """Relative position encoder.
    NOTE: Differ to AlphaFold3 official algorithm, its official algorithm does
    not pass linear projection layer here.
    """

    def __init__(self, r_max: int = 32, s_max: int = 2):
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
        self.dimension: int = 4 * (r_max + 1) + 2 * (s_max + 1) + 1

    def forward(
        self, f_input: FoldingInput, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """See Section 3.1.2 Algorithm 3: Relative position encoding in the AF3 paper.
        NOTE: Differ to AlphaFold3 official algorithm, its official algorithm does
        not pass linear projection layer here.
        """
        return self.get_relative_position_encoding(f_input, dtype)

    def get_relative_position_encoding(
        self, f_input: FoldingInput, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        # All shape: [B, Lt]
        asym_id = f_input.token.asym_id
        entity_id = f_input.token.entity_id
        sym_id = f_input.token.sym_id
        residue_index = f_input.token.residue_index
        token_index = f_input.token.token_index

        # Line 1
        b_same_chain = torch.eq(asym_id[:, :, None], asym_id[:, None, :])
        # Line 2
        b_same_residue = torch.eq(residue_index[:, :, None], residue_index[:, None, :])
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
        a_rel_pos = F.one_hot(d_residue, 2 * self.r_max + 2).to(dtype)

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
        a_rel_token = F.one_hot(d_token, 2 * self.r_max + 2).to(dtype)

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
        a_rel_chain = F.one_hot(d_chain, 2 * self.s_max + 2).to(dtype)

        # Line 10 (concat)
        rel_position_encoding = torch.cat(
            [
                a_rel_pos,
                a_rel_token,
                b_same_entity.to(dtype).unsqueeze(-1),
                a_rel_chain,
            ],
            dim=-1,
        )
        return rel_position_encoding  # [B, L, L, D]


class FourierEmbedding(torch.nn.Module):
    """Fourier embedding layer.
    Section 3.7 Algorithm 22 Fourier Embedding
    """

    def __init__(self, channel: int):
        """Initialize the Fourier Embeddings.

        Parameters
        ----------
        channel : int
            The fourier embedding dimension.
        seed : int, optional
            The random seed, by default 42

        """
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(42)

        # Line 1: Randomly generate weight/bias once before training
        w = torch.randn(size=(1, channel), generator=generator)
        b = torch.randn(size=(1, channel), generator=generator)
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("b", b, persistent=False)

    def forward(self, t_hat: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        See Section 3.7 Algorithm 22 of AlphaFold3 paper.

        Parameters
        ----------
        t_hat : torch.Tensor
            The input noise level. Shape (B, N,)

        Returns
        -------
        torch.Tensor
            The Fourier embeddings. Shape (B, N, channel)
        """
        # Line 2
        return torch.cos((2 * math.pi) * t_hat[..., None] * self.w + self.b)


class ConstraintEncoding(torch.nn.Module):
    """Constraint encoding"""

    def __init__(
        self,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        bin_size: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist
        self.bin_size: float = bin_size
        self.no_upper_limit: float = max_dist + bin_size
        boundaries = torch.arange(min_dist, max_dist + 2 * bin_size, bin_size)
        self.register_buffer("boundaries", boundaries, persistent=False)
        self.num_bins: int = len(boundaries) - 1

    def forward(
        self, f_input: FoldingInput, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        f_input : FoldingInput
            Input features

        Returns
        -------
        constraint_encoding : torch.Tensor
            The encoded constraints in pairwise representation, shape [B, L, L, bins]
        """
        cond_index = f_input.constraint.token_index  # [B, num_conds, 2]
        mask = f_input.constraint.pad_mask  # [B, num_conds]
        lower_bound = f_input.constraint.lower_bound  # [B, num_conds]
        upper_bound = f_input.constraint.upper_bound  # [B, num_conds]

        # Handle -1 flags:
        # -1 upper bound means no upper limit (use infinity)
        lower_bound = lower_bound.clamp(self.min_dist, self.no_upper_limit)
        no_upper_bound = upper_bound == -1
        upper_bound = upper_bound.clamp(self.min_dist, self.no_upper_limit)
        upper_bound[no_upper_bound] = self.no_upper_limit

        B, L = f_input.batch_size, f_input.num_tokens
        dev = f_input.device

        # Retrieve bin boundaries: [num_bins]
        boundaries: torch.Tensor = self.boundaries  # [num_bins + 1]
        bin_lower = boundaries[:-1]
        bin_upper = boundaries[1:]

        # Calculate overlap of constraint with each bin: [B, num_conds, num_bins]
        lower_bound, upper_bound = lower_bound[..., None], upper_bound[..., None]
        overlap = (lower_bound < bin_upper) & (upper_bound > bin_lower)
        overlap = (overlap & mask[..., None]).to(dtype)

        # Create pairwise representation directly in target dtype
        adj = torch.zeros((B, L, L, self.num_bins), device=dev, dtype=dtype)
        b_idc = torch.arange(B, device=dev).view(B, 1)
        src, dst = cond_index[:, :, 0], cond_index[:, :, 1]

        # Advanced indexing to populate the adjacency matrix
        adj[b_idc, src, dst] = overlap
        adj[b_idc, dst, src] = overlap  # undirected

        # Ensure padding tokens (index 0) have no constraints
        adj[:, 0, 0] = 0.0

        # Normalize across bins
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1.0)

        return adj


class ApoEmbedding(torch.nn.Module):
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
        def get_pair_mask(m: torch.Tensor) -> torch.Tensor:
            return m[..., :, None] & m[..., None, :]

        # Get distogram
        repr_coords = f_input.token.apo_repr_coords  # [B, L, 3]
        dgram = self.get_distogram(repr_coords)  # [B, L, L, num_bins]
        dgram = dgram.float()  # Convert to float
        dgram_mask = get_pair_mask(f_input.token.apo_repr_mask)  # [B, L, L]
        dgram.masked_fill_(~dgram_mask[..., None], 0.0)

        # Get frame-based representation
        center_coords = f_input.token.apo_center_coords  # [B, L, 3]
        frame = f_input.token.apo_frame_coords  # [B, L, 3, 3]
        coords_in_frame = self.express_coords_in_frames(
            center_coords, frame
        )  # [B, L, L, 3]
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

    def get_distogram(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the distogram from the input coordinates."""
        # Compute pairwise distances
        d = (x[..., None, :, :] - x[..., :, None, :]).norm(dim=-1)  # [..., L, L]
        num_bins = self.boundaries.shape[0] + 1
        distogram = F.one_hot(
            (d[..., None] > self.boundaries).sum(dim=-1), num_bins
        )  # [..., L, L, num_bins]
        return distogram

    @staticmethod
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
