import torch


def add(x: torch.Tensor, y: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Add two tensors together, optionally in-place.

    Parameters
    ----------
    x : torch.Tensor
        The first tensor to add.
    y : torch.Tensor
        The second tensor to add.
    inplace : bool, optional
        Whether to perform the addition in-place on `x`, by default False.

    Returns
    -------
    torch.Tensor
        The result of adding `x` and `y`.
    """
    if inplace:
        return x.add_(y)
    else:
        return x + y
