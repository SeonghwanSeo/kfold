from collections.abc import Sequence

import torch


def permute_final_dims(tensor: torch.Tensor, inds: Sequence[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int) -> torch.Tensor:
    return t.reshape(t.shape[:-no_dims] + (-1,))


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


def gather_dim(
    tensor: torch.Tensor,
    dim: int,
    index: torch.Tensor,
) -> torch.Tensor:
    """Helper function to gather values from `tensor` along `dim` using `index`."""
    dim = dim % tensor.ndim  # Normalize negative dims to positive
    assert tensor.ndim == index.ndim
    expand_shape = [-1 if i == dim else d for i, d in enumerate(tensor.shape)]

    mask = index < 0  # Mask for out-of-bounds indices
    index_clipped = index.clamp(min=0)
    out = tensor.gather(dim, index=index_clipped.expand(expand_shape))
    return out.masked_fill_(mask, 0)


def get_context_dtype(device_type: str | None = None) -> torch.dtype:
    """Get the current context dtype for autocast."""
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    else:
        return torch.float32
