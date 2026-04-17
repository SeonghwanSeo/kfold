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


def expand_dim(
    tensor: torch.Tensor,
    n_repeat: int,
    dim: int,
    add_new_dim: bool = True,
) -> torch.Tensor:
    """Expanding a tensor with new shape"""
    if add_new_dim:
        tensor = tensor.unsqueeze(dim)
    shape = tensor.shape
    to_expand = [-1] * len(shape)
    to_expand[dim] = n_repeat
    return tensor.expand(*to_expand)


def repeat_dim(
    tensor: torch.Tensor,
    n_repeat: int,
    dim: int,
    add_new_dim: bool = True,
) -> torch.Tensor:
    """Repeat a tensor with new shape"""
    if add_new_dim:
        tensor = tensor.unsqueeze(dim)
    shape = tensor.shape
    to_repeat = [1] * len(shape)
    to_repeat[dim] = n_repeat
    return tensor.repeat(*to_repeat)


def pad_dim(
    tensor: torch.Tensor,
    dim: int,
    max_len: int,
    pad_value: float | int | bool = 0.0,
) -> torch.Tensor:
    """Pad a tensor with new shape"""
    current_len = tensor.shape[dim]
    if current_len > max_len:
        raise ValueError(
            f"Cannot pad tensor of shape {tensor.shape} to max_len {max_len} "
            f"along dim {dim}"
        )
    if current_len == max_len:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = max_len - current_len
    pad_tensor = torch.full(shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=dim)
