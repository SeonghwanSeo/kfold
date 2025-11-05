# Started from code from https://github.com/jwohlwend/boltz, MIT License

from collections.abc import Callable

import torch
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from torch import nn, sigmoid
from torch.nn import (
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    Sequential,
)

from ..layers.attention import AttentionPairBias
from .utils import LinearNoBias, SwiGLU, default


class AdaLN(Module):
    """Adaptive Layer Normalization"""

    def __init__(self, dim: int, dim_single_cond: int):
        """Initialize the adaptive layer normalization.

        Parameters
        ----------
        dim : int
            The input dimension.
        dim_single_cond : int
            The single condition dimension.

        """
        super().__init__()
        self.a_norm = LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = LayerNorm(dim_single_cond, bias=False)
        self.s_scale = Linear(dim_single_cond, dim)
        self.s_bias = LinearNoBias(dim_single_cond, dim)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a = self.a_norm(a)
        s = self.s_norm(s)
        a = sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a


class ConditionedTransitionBlock(Module):
    """Conditioned Transition Block"""

    def __init__(self, dim_single, dim_single_cond, expansion_factor=2):
        """Initialize the conditioned transition block.

        Parameters
        ----------
        dim_single : int
            The single dimension.
        dim_single_cond : int
            The single condition dimension.
        expansion_factor : int, optional
            The expansion factor, by default 2

        """
        super().__init__()

        self.adaln = AdaLN(dim_single, dim_single_cond)

        dim_inner = int(dim_single * expansion_factor)
        self.swish_gate = Sequential(
            LinearNoBias(dim_single, dim_inner * 2),
            SwiGLU(),
        )
        self.a_to_b = LinearNoBias(dim_single, dim_inner)
        self.b_to_a = LinearNoBias(dim_inner, dim_single)

        output_projection_linear = Linear(dim_single_cond, dim_single)
        nn.init.zeros_(output_projection_linear.weight)
        nn.init.constant_(output_projection_linear.bias, -2.0)

        self.output_projection = nn.Sequential(output_projection_linear, nn.Sigmoid())

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a = self.adaln(a, s)
        b = self.swish_gate(a) * self.a_to_b(a)
        a = self.output_projection(s) * self.b_to_a(b)

        return a


class DiffusionTransformer(Module):
    """Diffusion Transformer"""

    def __init__(
        self,
        depth: int,
        heads: int,
        dim: int = 384,
        dim_single_cond: int | None = None,
        dim_pairwise: int = 128,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
    ):
        """Initialize the diffusion transformer.

        Parameters
        ----------
        depth : int
            The depth.
        heads : int
            The number of heads.
        dim : int, optional
            The dimension, by default 384
        dim_single_cond : int, optional
            The single condition dimension, by default None
        dim_pairwise : int, optional
            The pairwise dimension, by default 128
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False
        offload_to_cpu : bool, optional
            Whether to offload to CPU, by default False

        """
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        dim_single_cond = default(dim_single_cond, dim)

        self.layers = ModuleList()
        for _ in range(depth):
            if activation_checkpointing:
                self.layers.append(
                    checkpoint_wrapper(
                        DiffusionTransformerLayer(
                            heads,
                            dim,
                            dim_single_cond,
                            dim_pairwise,
                        ),
                        offload_to_cpu=offload_to_cpu,
                    )
                )
            else:
                self.layers.append(
                    DiffusionTransformerLayer(
                        heads,
                        dim,
                        dim_single_cond,
                        dim_pairwise,
                    )
                )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None,
        to_keys: Callable | None = None,
        multiplicity: int = 1,
        model_cache=None,
    ):
        for i, layer in enumerate(self.layers):
            layer_cache = None
            if model_cache is not None:
                prefix_cache = "layer_" + str(i)
                if prefix_cache not in model_cache:
                    model_cache[prefix_cache] = {}
                layer_cache = model_cache[prefix_cache]
            a = layer(
                a,
                s,
                z,
                mask=mask,
                to_keys=to_keys,
                multiplicity=multiplicity,
                layer_cache=layer_cache,
            )
        return a


class DiffusionTransformerLayer(Module):
    """Diffusion Transformer Layer"""

    def __init__(
        self,
        heads: int,
        dim: int = 384,
        dim_single_cond: int | None = None,
        dim_pairwise: int = 128,
    ):
        """Initialize the diffusion transformer layer.

        Parameters
        ----------
        heads : int
            The number of heads.
        dim : int, optional
            The dimension, by default 384
        dim_single_cond : int, optional
            The single condition dimension, by default None
        dim_pairwise : int, optional
            The pairwise dimension, by default 128

        """
        super().__init__()

        dim_single_cond = default(dim_single_cond, dim)

        self.adaln = AdaLN(dim, dim_single_cond)

        self.pair_bias_attn = AttentionPairBias(
            c_s=dim, c_z=dim_pairwise, num_heads=heads, initial_norm=False
        )

        self.output_projection_linear = Linear(dim_single_cond, dim)
        nn.init.zeros_(self.output_projection_linear.weight)
        nn.init.constant_(self.output_projection_linear.bias, -2.0)

        self.output_projection = nn.Sequential(
            self.output_projection_linear, nn.Sigmoid()
        )
        self.transition = ConditionedTransitionBlock(
            dim_single=dim, dim_single_cond=dim_single_cond
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        to_keys: Callable | None = None,
        multiplicity: int = 1,
        layer_cache=None,
    ) -> torch.Tensor:
        b = self.adaln(a, s)
        b = self.pair_bias_attn(
            s=b,
            z=z,
            mask=mask,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=layer_cache,
        )
        b = self.output_projection(s) * b

        # NOTE: Added residual connection!
        a = a + b
        a = a + self.transition(a, s)
        return a


class AtomTransformer(Module):
    """Atom Transformer"""

    def __init__(
        self,
        attn_window_queries: int,
        attn_window_keys: int,
        **diffusion_transformer_kwargs,
    ):
        """Initialize the atom transformer.

        Parameters
        ----------
        attn_window_queries : int
            The attention window queries, by default None
        attn_window_keys : int
            The attention window keys, by default None
        diffusion_transformer_kwargs : dict
            The diffusion transformer keyword arguments

        """
        super().__init__()
        self.attn_window_queries: int = attn_window_queries
        self.attn_window_keys: int = attn_window_keys
        self.diffusion_transformer = DiffusionTransformer(**diffusion_transformer_kwargs)

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        mask: torch.Tensor,
        to_keys,
        multiplicity: int = 1,
        model_cache=None,
    ) -> torch.Tensor:
        W: int = self.attn_window_queries
        H: int = self.attn_window_keys

        B, N, D = q.shape
        NW = N // W

        # reshape tokens
        q = q.view((B * NW, W, -1))
        c = c.view((B * NW, W, -1))
        p = p.view((B * NW, W, H, -1))
        mask = mask.view(B * NW, W)

        to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)

        # main transformer
        q = self.diffusion_transformer(
            a=q,
            s=c,
            z=p,
            mask=mask.float(),
            multiplicity=1,  # bias term already expanded with multiplicity
            to_keys=to_keys_new,
            model_cache=model_cache,
        )

        if W is not None:
            q = q.view((B, NW * W, D))

        return q
