from __future__ import annotations

import torch


def entity_bin_index_from_asym_id(
    asym_id: torch.Tensor, pad_mask: torch.Tensor
) -> torch.Tensor:
    """Compute entity-bin indices per batch item from token asym_id.

    Definition:
    - entity count = unique(asym_id among pad_mask==True tokens)
    - bins: 1..9 map to indices 0..8, >=10 maps to 9

    Parameters
    ----------
    asym_id : torch.Tensor
        Token asym ids. Shape [B, Lt].
    pad_mask : torch.Tensor
        Valid-token mask. Shape [B, Lt], bool.
    """
    if asym_id.ndim != 2 or pad_mask.ndim != 2:
        raise ValueError("asym_id and pad_mask must be 2D tensors [B, Lt]")
    if asym_id.shape != pad_mask.shape:
        raise ValueError("asym_id and pad_mask must have the same shape")

    B = asym_id.shape[0]
    out = torch.empty(B, device=asym_id.device, dtype=torch.long)
    for b in range(B):
        ids = asym_id[b][pad_mask[b]]
        n = int(torch.unique(ids).numel()) if ids.numel() > 0 else 0
        if n <= 1:
            out[b] = 0
        elif n >= 10:
            out[b] = 9
        else:
            out[b] = n - 1
    return out
