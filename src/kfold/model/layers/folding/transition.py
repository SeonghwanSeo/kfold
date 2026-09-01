import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kfold.model.primitives import LayerNorm, LinearNoBias, SwiGLU

# This shared function sees several valid channel/expansion signatures in K-Fold.
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)


def _transition_forward(
    x: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    swiglu_weight: Tensor,
    output_weight: Tensor,
    eps: float,
) -> Tensor:
    """Transition math shared by eager and compiled inference paths."""
    normalized = F.layer_norm(
        x.float(),
        (x.shape[-1],),
        norm_weight,
        norm_bias,
        eps,
    ).to(x.dtype)
    a, b = F.linear(normalized, swiglu_weight).chunk(2, dim=-1)
    return F.linear(F.silu(a) * b, output_weight)


_compiled_transition_forward = torch.compile(
    _transition_forward,
    mode="max-autotune-no-cudagraphs",
    fullgraph=True,
    dynamic=True,
)


def use_compiled_transition(module: nn.Module, x: Tensor) -> bool:
    """Use the measured max-autotune path only for standalone CUDA inference."""
    return x.is_cuda and not module.training and not torch.compiler.is_compiling()


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
        if use_compiled_transition(self, x):
            return _compiled_transition_forward(
                x,
                self.layernorm.weight,
                self.layernorm.bias,
                self.swiglu.linear.weight,
                self.linear_out.weight,
                self.layernorm.eps,
            )

        # Line 1
        x = self.layernorm(x)

        # Line 2-4
        x = self.swiglu(x)
        x = self.linear_out(x)
        return x
