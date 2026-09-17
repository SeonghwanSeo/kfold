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

from torch import Tensor, nn

from kfold.model.primitives import LayerNorm, LinearNoBias, SwiGLU


class Transition(nn.Module):
    """Perform a two-layer MLP.
    See Section 3.3 Algorithm 11 Transition layer
    """

    def __init__(
        self,
        channel: int,
        expansion_factor: int,
    ) -> None:
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        channel: int
            The dimension of the input
        expansion_factor: int
            The expansion factor for the hidden dimension

        """
        super().__init__()

        model_dim = channel * expansion_factor
        self.model_dim = model_dim
        self.layernorm = LayerNorm(channel)
        self.swiglu = SwiGLU(channel, model_dim)
        self.linear_out = LinearNoBias(model_dim, channel, init="final")

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., D)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (..., D)

        """
        # Line 1
        x = self.layernorm(x)

        # Line 2-4
        x = self.swiglu(x)
        x = self.linear_out(x)
        return x
