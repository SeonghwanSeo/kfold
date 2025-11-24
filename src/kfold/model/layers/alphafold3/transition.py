from torch import Tensor, nn

from kfold.model.layers.primitives import LayerNorm, LinearNoBias, SwiGLU


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
