import torch


@torch.no_grad()
def gradient_norm(module: torch.nn.Module) -> float:
    # Only compute over parameters that are being trained
    parameters = filter(lambda p: p.requires_grad, module.parameters())
    parameters = filter(lambda p: p.grad is not None, parameters)
    norm = torch.tensor([p.grad.norm(p=2) ** 2 for p in parameters]).sum().sqrt()
    return norm.item()


@torch.no_grad()
def parameter_norm(module: torch.nn.Module) -> float:
    # Only compute over parameters that are being trained
    parameters = filter(lambda p: p.requires_grad, module.parameters())
    norm = torch.tensor([p.norm(p=2) ** 2 for p in parameters]).sum().sqrt()
    return norm.item()
