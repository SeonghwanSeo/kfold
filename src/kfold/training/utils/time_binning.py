from __future__ import annotations

import torch


def compute_bin_index(u: torch.Tensor, nbins: int) -> torch.Tensor:
    """Compute bin index for u∈[0,1] into {0..nbins-1}.

    u==1.0 is mapped to the last bin.
    """
    if nbins <= 0:
        raise ValueError(f"nbins must be positive, got {nbins}")
    return torch.clamp((u * nbins).to(dtype=torch.long), 0, nbins - 1)


def binned_sum_and_count(
    values: torch.Tensor, bin_index: torch.Tensor, nbins: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-bin sums and counts using scatter_add.

    Parameters
    ----------
    values : torch.Tensor
        Arbitrary-shaped tensor of values.
    bin_index : torch.Tensor
        Same shape as values, containing bin indices in [0, nbins-1].
    nbins : int
        Number of bins.
    """
    if values.shape != bin_index.shape:
        raise ValueError(
            "values and bin_index must have same shape, got "
            f"{values.shape} vs {bin_index.shape}"
        )
    if nbins <= 0:
        raise ValueError(f"nbins must be positive, got {nbins}")

    flat_vals = values.reshape(-1)
    flat_bin = bin_index.reshape(-1)
    ones = torch.ones_like(flat_vals)

    bin_sum = torch.zeros(nbins, device=flat_vals.device).scatter_add_(
        0, flat_bin, flat_vals
    )
    bin_count = torch.zeros(nbins, device=flat_vals.device).scatter_add_(
        0, flat_bin, ones
    )
    return bin_sum, bin_count
