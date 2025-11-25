from functools import partial

import torch.nn as nn

from . import initialize


class Linear(nn.Linear):
    """A linear layer with various initialization methods.
    Starting from https://github.com/aqlaboratory/openfold-3
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "default",
    ):
        super().__init__(in_features, out_features, bias)
        if init == "default":
            initialize.lecun_normal_init_(self.weight)
        elif init == "relu":
            initialize.he_normal_init_(self.weight)
        elif init == "gating":
            # weight: zero
            initialize.gating_init_(self.weight)
        elif init == "gating_ada_zero":
            # weight: zero, bias: -2
            assert self.bias is not None, (
                "Bias must be True for gating_ada_zero initialization."
            )
            initialize.gating_init_(self.weight)
            nn.init.constant_(self.bias, -2.0)
        elif init == "final":
            # weight: zero
            initialize.final_init_(self.weight)
        else:
            raise ValueError(f"Unknown initialization method: {init}")


LinearNoBias = partial(Linear, bias=False)
