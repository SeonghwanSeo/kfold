"""Implemented by https://github.com/jwohlwend/boltz"""

import torch
import torch.nn as nn


def get_dropout_mask(
    z: torch.Tensor,
    dropout: float,
    training: bool,
    columnwise: bool = False,
) -> torch.Tensor:
    """Get the dropout mask.

    Parameters
    ----------
    dropout : float
        The dropout rate
    z : torch.Tensor
        The tensor to apply dropout to
    training : bool
        Whether the model is in training mode
    columnwise : bool, optional
        Whether to apply dropout columnwise

    Returns
    -------
    torch.Tensor
        The dropout mask

    """
    if (not training) or (dropout == 0.0):
        return torch.ones_like(z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1])
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    d = torch.rand_like(v) >= dropout
    d = d * 1.0 / (1.0 - dropout)
    return d


class DropoutRowwise(nn.Module):
    """Row-wise dropout layer."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout: float = dropout

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        if training is None:
            training = self.training

        if not training or self.dropout == 0.0:
            return x

        # Apply dropout
        mask = get_dropout_mask(x, self.dropout, training, columnwise=False)
        return x * mask


class DropoutColumnwise(nn.Module):
    """Row-wise dropout layer."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout: float = dropout

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        if training is None:
            training = self.training

        if not training or self.dropout == 0.0:
            return x

        # Apply dropout
        mask = get_dropout_mask(x, self.dropout, training, columnwise=True)
        return x * mask
