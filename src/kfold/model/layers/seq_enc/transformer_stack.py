import math

import torch

from .blocks import TransformerBlock


class TransformerStack(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        scale_residue: bool = True,
        expansion_ratio: float = 8 / 3,
        use_moe: bool = False,
    ):
        super().__init__()
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers

        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    expansion_ratio=expansion_ratio,
                    use_moe=use_moe,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor,
        pos_id: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:  # hidden states, attentions
        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, seq_id, pos_id)
            hidden_states.append(x)
        x = self.norm(x)
        return x, hidden_states
