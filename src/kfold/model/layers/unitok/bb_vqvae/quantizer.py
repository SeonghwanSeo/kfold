from functools import cached_property

import torch
import torch.nn as nn


class Quantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        codebook_embed_size: int,
        use_linear_project: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_embed_size = codebook_embed_size
        self.codebook = nn.Embedding(codebook_size, codebook_embed_size)
        self.use_linear_project = use_linear_project
        if self.use_linear_project:
            self.linear_proj = nn.Linear(codebook_embed_size, codebook_embed_size)

    @cached_property
    def weight(self):
        if self.use_linear_project:
            w = self.linear_proj(self.codebook.weight).float()
        else:
            w = self.codebook.weight.float()
        wT = w.T.contiguous()  # [hidden_dim, codebook_size]
        return w, wT

    def indices2embedding(self, indices: torch.IntTensor) -> torch.Tensor:
        z_q = self.codebook.weight[indices]
        return z_q

    def forward(self, z: torch.Tensor):
        """
        Return: quantized_z, detached codes
        """
        raise NotImplementedError

    def embedding2indices(self, z: torch.Tensor) -> torch.Tensor:
        B, L, D = z.shape
        with torch.autocast(device_type=z.device.type, enabled=False):
            w, wT = self.weight  # [codebook_size, hidden_dim]
            flat_z = z.float().reshape(B * L, D)
            d = (
                torch.sum(flat_z**2, dim=1, keepdim=True)
                + torch.sum(w**2, dim=1)
                - 2 * torch.matmul(flat_z, wT)
            )  # [B * L, codebook_size]
        quantized_indices = torch.argmin(d, dim=1)
        quantized_indices = quantized_indices.view(B, L)
        return quantized_indices
