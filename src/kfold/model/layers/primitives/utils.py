import torch


def add(x: torch.Tensor, y: float | torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.add_(y)
    else:
        return x + y


def sub(x: torch.Tensor, y: float | torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.sub_(y)
    else:
        return x - y


def div(x: torch.Tensor, y: float | torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.div_(y)
    else:
        return x / y


def mul(x: torch.Tensor, y: float | torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.mul_(y)
    else:
        return x * y
