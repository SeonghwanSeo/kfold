# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from https://github.com/SeonghwanSeo/atlasfold

import math

import torch

from .blocks import TransformerBlock
from .nn import LayerNorm
from .rotary import RotaryEmbedding


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
        self.rotary = RotaryEmbedding(d_model // n_heads, max_seqlen=20000)

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
        self.norm = LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor,
        pos_id: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:  # hidden states, attentions
        hidden_states: list[torch.Tensor] = []
        rotary = self.rotary(pos_id)
        for block in self.blocks:
            x = block(x, seq_id, rotary)
            hidden_states.append(x)
        x = self.norm(x)
        return x, hidden_states
