import torch
import torch.nn as nn

from kfold.model.layers.alphafold3 import initialize as init
from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.model.layers.alphafold3.primitives import LinearNoBias
from kfold.utils.registry import TRUNK

from .base import BaseTrunk


@TRUNK.register()
class AF3PairformerModule(BaseTrunk):
    """AlphaFold3 pairformer module.

    See Section 3 Algorithm 1 Main Inference Loop: Line[6-14]
    """

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

    def __init__(self, cfg: Config, use_kernels: bool = False):
        """Initialize the Pairformer module."""
        super().__init__(cfg)
        self.use_msa = cfg.use_msa
        self.use_template = cfg.use_template
        self.use_kernels = cfg.use_kernels or use_kernels

        assert self.no_template, "Boltz1 does not support template."

        self.pairformer_module = PairformerStack(
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

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        num_recycles: int,
        chunk_size_tri_attn: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.
        See Section 3 Algorithm 1 Main Inference Loop: Line[6-14]

        Parameters
        ----------
        s_inputs : torch.Tensor
            Tensor of shape (L, C_s) containing input single features
        s_inits: torch.Tensor
            Tensor of shape (L, C_s) containing initial single representation
        z_inits: torch.Tensor
            Tensor of shape (L, L, C_s) containing initial pair representation
        mask : torch.Tensor
            The token mask of shape (B, L)
        pair_mask : torch.Tensor
            The pairwise mask of shape (B, L, L)
        num_recycles : int
            The number of recycling steps.

        Returns
        -------
        s_trunk: torch.Tensor
            The updated tensor of shape (B, L, c_s).
        z_trunk: torch.Tensor
            The updated tensor of shape (B, L, L, c_z).
        """
        if not self.training:
            if z_init.shape[1] > 384:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        # Line 6, z_hat, s_hat = 0, 0
        z_hat, s_hat = z_init, s_init  # just to make sure the types are correct

        for i in range(num_recycles + 1):
            enable_grad = self.training and (i == num_recycles)

            with torch.set_grad_enabled(enable_grad):
                # Fixes an issue with unused parameters in autocast
                if self.training and (i == num_recycles) and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                # Line 8
                z = z_init + self.linear_no_bias_z(self.layernorm_z(z_hat))

                # Line 9: TemplateEmbedder
                if self.use_templte:
                    raise NotImplementedError("Template Embedder is not implemented yet")

                # Line 10: MSAModule
                if self.use_msa:
                    raise NotImplementedError("MSA Module is not implemented yet")

                # Line 11
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s_hat))

                # Line 12
                # Revert to uncompiled version for validation
                pairformer_module: PairformerStack
                if self.is_pairformer_compiled and not self.training:
                    pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                else:
                    pairformer_module = self.pairformer_module
                s, z = pairformer_module(
                    s,
                    z,
                    mask=mask,
                    pair_mask=pair_mask,
                    chunk_size_tri_attn=chunk_size_tri_attn,
                )

                # Line 13
                s_hat, z_hat = s, z

        s_trunk, z_trunk = s_hat, z_hat
        return s_trunk, z_trunk
