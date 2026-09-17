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

import torch

from .linear import LinearNoBias


class SwiGLU(torch.nn.Module):
    """SiLU Gated Linear Unit (SwiGLU) activation function."""

    def __init__(self, channel_in: int, channel_out: int):
        super().__init__()
        self.linear = LinearNoBias(channel_in, channel_out * 2, init="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.linear(x).chunk(2, dim=-1)
        return torch.nn.functional.silu(a) * b
