import torch

from kfold.data.types.model_input import FoldingInput


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
