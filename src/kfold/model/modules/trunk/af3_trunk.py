import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3 import initialize as init
from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.model.layers.alphafold3.primitives import LinearNoBias
from kfold.utils.registry import TRUNK

from .base import BaseTrunk


@TRUNK.register()
class AF3PairformerTrunk(BaseTrunk):
    """AlphaFold3 pairformer module.

    See Section 3 Algorithm 1 Main Inference Loop: Line[6-14]
    """

    @dataclass
    class Config(BaseTrunk.Config):
        """Configuration for the Pairformer module.

        Parameters
        ----------
        channel_s : int
            The token single embedding size.
        channel_z : int
            The token pairwise embedding size.
        num_blocks : int
            The number of blocks.
        num_heads : int, optional
            The number of heads, by default 16
        dropout : float, optional
            The dropout rate, by default 0.25
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        use_template: bool, optional
            Whether to use template, by default False
        use_msa: bool, optional
            Whether to use MSA, by default False
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False
        offload_to_cpu : bool, optional
            Whether to offload to CPU, by default False
        use_kernels : bool, optional
            Whether to use custom kernels, by default False
        tri_attn_chunk_threshold : int, optional
            The threshold for chunking in triangle attention, by default 384
        """

        channel_s: int = 384
        channel_z: int = 128
        num_blocks: int = 48
        num_heads: int = 16
        dropout: float = 0.25
        pairwise_head_width: int = 32
        pairwise_num_heads: int = 4
        use_msa: bool = False
        use_template: bool = False
        activation_checkpointing: bool = False
        offload_to_cpu: bool = False
        use_kernels: bool = False
        tri_attn_chunk_threshold: int = 384

    def __init__(self, cfg: Config):
        """Initialize the Pairformer module."""
        super().__init__(cfg)
        self.use_msa: bool = cfg.use_msa
        self.use_template: bool = cfg.use_template
        self.use_kernels: bool = cfg.use_kernels
        self.chunk_threshold: int = cfg.tri_attn_chunk_threshold

        if self.use_template:
            raise NotImplementedError(
                "Template Embedder is not implemented yet (Boltz1 does not support too)"
            )
        if self.use_msa:
            # TODO: Implement MSA Module
            raise NotImplementedError("MSA Module is not implemented yet")

        self.pairformer_module: PairformerStack = PairformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            pairwise_head_width=cfg.pairwise_head_width,
            pairwise_num_heads=cfg.pairwise_num_heads,
            activation_checkpointing=cfg.activation_checkpointing,
            offload_to_cpu=cfg.offload_to_cpu,
            use_kernels=self.use_kernels,
        )

        # For recycling
        self.layernorm_s = nn.LayerNorm(cfg.channel_s)
        self.layernorm_z = nn.LayerNorm(cfg.channel_z)
        self.linear_no_bias_s = LinearNoBias(cfg.channel_s, cfg.channel_s)
        self.linear_no_bias_z = LinearNoBias(cfg.channel_z, cfg.channel_z)
        init.gating_init_(self.linear_no_bias_s.weight)
        init.gating_init_(self.linear_no_bias_z.weight)

    def do_compile(self):
        """Compile the trunk module."""
        # NOTE: you should compile the submodules inside the trunk
        # since the computation graph is changed depending on the
        # number of recycling steps. Thus, compile the sub module
        # instead of the whole trunk module.
        self.pairformer_module = torch.compile(
            self.pairformer_module, dynamic=False, fullgraph=False
        )  # type: ignore

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_cycles: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.
        See Section 3 Algorithm 1 Main Inference Loop: Line[6-14]

        Parameters
        ----------
        s_inputs : torch.Tensor
            Tensor of shape (B, L, C_s) containing input single features
        s_inits: torch.Tensor
            Tensor of shape (B, L, C_s) containing initial single representation
        z_inits: torch.Tensor
            Tensor of shape (B, L, L, C_s) containing initial pair representation
        f_input : FoldingInput
            The input features.
        num_cycles : int
            The number of recycling steps.

        Returns
        -------
        s_trunk: torch.Tensor
            The updated tensor of shape (B, L, c_s).
        z_trunk: torch.Tensor
            The updated tensor of shape (B, L, L, c_z).
        """
        if not self.training:
            if z_init.shape[1] > self.chunk_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        # Line 6, z_hat, s_hat = 0, 0
        z_hat, s_hat = z_init, s_init  # just to make sure the types are correct

        for i in range(1, num_cycles + 1):
            no_grad = self.training and (i < num_cycles)

            with no_grad and torch.no_grad() or contextlib.nullcontext():
                # Fixes an issue with unused parameters in autocast
                if self.training and (i == num_cycles) and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                # Line 8
                z = z_init + self.linear_no_bias_z(self.layernorm_z(z_hat))

                # Line 9: TemplateEmbedder
                if self.use_template:
                    raise NotImplementedError("Template Embedder is not implemented yet")

                # Line 10: MSAModule
                if self.use_msa:
                    raise NotImplementedError("MSA Module is not implemented yet")

                # Line 11
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s_hat))

                # Line 12
                # Revert to uncompiled version for validation
                pairformer_module: PairformerStack
                if self.is_compiled and not self.training:
                    pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                else:
                    pairformer_module = self.pairformer_module

                s, z = pairformer_module(
                    s,
                    z,
                    mask=f_input.token.pad_mask.float(),
                    chunk_size_tri_attn=chunk_size_tri_attn,
                )

                # Line 13
                s_hat, z_hat = s, z

        s_trunk, z_trunk = s_hat, z_hat
        return s_trunk, z_trunk
