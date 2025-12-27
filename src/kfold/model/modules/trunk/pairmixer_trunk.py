"""AF3-like trunk utilizing the Pairmixer module."""

import torch

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.model.layers.pairmixer.pairmixer import PairmixerStack
from kfold.model.layers.primitives import LayerNorm, LinearNoBias
from kfold.utils.registry import TRUNK

from .base import BaseTrunk


@TRUNK.register()
class PairmixerTrunk(BaseTrunk):
    """AF3-like trunk with the Pairmixer module."""

    class Config(BaseTrunk.Config):
        """Configuration for the Pairmixer module.

        Parameters
        ----------
        channel_s : int
            The token single embedding size.
        channel_z : int
            The token pairwise embedding size.
        num_blocks : int
            The number of blocks.
        dropout : float, optional
            The dropout rate, by default 0.25
        use_template: bool, optional
            Whether to use template, by default False
        use_msa: bool, optional
            Whether to use MSA, by default False
        blocks_per_ckpt : int, optional
            The number of blocks per checkpoint, by default None
        """

        channel_s: int = 384
        channel_z: int = 128
        num_blocks: int = 48
        dropout: float = 0.25
        use_msa: bool = False
        use_template: bool = False
        blocks_per_ckpt: int | None = None

    def __init__(self, cfg: Config, kernel_config):
        """Initialize the Pairmixer module."""
        super().__init__(cfg, kernel_config)
        self.use_msa: bool = cfg.use_msa
        self.use_template: bool = cfg.use_template

        if self.use_template:
            raise NotImplementedError(
                "Template Embedder is not implemented yet (Boltz1 does not support too)"
            )
        if self.use_msa:
            # TODO: Implement MSA Module
            # NOTE: will absence of MSA degrade performance of pairmixer?
            raise NotImplementedError("MSA Module is not implemented yet")

        self.pairmixer_module: PairmixerStack = PairmixerStack(
            channel_z=cfg.channel_z,
            num_blocks=cfg.num_blocks,
            dropout=cfg.dropout,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )

        # For recycling
        self.layernorm_s = LayerNorm(cfg.channel_s)
        self.layernorm_z = LayerNorm(cfg.channel_z)
        self.linear_s = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")

    def do_compile(self):
        """Compile the trunk module."""
        # NOTE: you should compile the submodules inside the trunk
        # since the computation graph is changed depending on the
        # number of recycling steps. Thus, compile the sub module
        # instead of the whole trunk module.
        self.pairmixer_module = torch.compile(
            self.pairmixer_module, dynamic=False, fullgraph=False
        )  # type: ignore

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_recycles: int,
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
        num_recycles : int
            The number of recycling steps.

        Returns
        -------
        s_trunk: torch.Tensor
            The updated tensor of shape (B, L, c_s).
        z_trunk: torch.Tensor
            The updated tensor of shape (B, L, L, c_z).
        """

        # Line 6, z_hat, s_hat = 0, 0
        s_hat = torch.zeros_like(s_init)
        z_hat = torch.zeros_like(z_init)

        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i == num_recycles

            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                # Line 8
                z = z_init + self.linear_z(self.layernorm_z(z_hat))

                # Line 9: TemplateEmbedder
                if self.use_template:
                    raise NotImplementedError("Template Embedder is not implemented yet")

                # Line 10: MSAModule
                if self.use_msa:
                    raise NotImplementedError("MSA Module is not implemented yet")

                # Line 11
                s = s_init + self.linear_s(self.layernorm_s(s_hat))

                # Line 12
                # Revert to uncompiled version for validation
                pairmixer_module: PairmixerStack
                if self.is_compiled and not self.training:
                    pairmixer_module = self.pairmixer_module._orig_mod  # noqa: SLF001
                else:
                    pairmixer_module = self.pairmixer_module

                s, z = pairmixer_module(
                    s,
                    z,
                    mask=f_input.token.pad_mask,
                    use_cuequiv_kernels=self.kernel_config.cuequivariance,
                )

                # Line 13
                s_hat, z_hat = s, z

        s_trunk, z_trunk = s_hat, z_hat
        return s_trunk, z_trunk


@TRUNK.register()
class PairmixerformerTrunk(BaseTrunk):
    """Intermediate trunk module with Pairmixer early on,
    Pairformer later."""

    class Config(BaseTrunk.Config):
        channel_s: int = 384
        channel_z: int = 128
        dropout: float = 0.25
        use_msa: bool = False
        use_template: bool = False
        blocks_per_ckpt: int | None = None
        num_heads_attn: int = 16
        num_heads_tri_attn: int = 4
        num_blocks: int = 48
        pairmixer_blocks: int = 42
        chunk_threshold: int = 384

    def __init__(self, cfg: Config):
        """Initialize the Pairmixerformer module."""
        super().__init__(cfg)

        if cfg.num_blocks < cfg.pairmixer_blocks:
            raise ValueError(
                f"pairmixer_blocks ({cfg.pairmixer_blocks}) must be less than or equal"
                f" to num_blocks ({cfg.num_blocks})"
            )

        self.use_msa: bool = cfg.use_msa
        self.use_template: bool = cfg.use_template
        self.chunk_threshold: int = cfg.chunk_threshold

        if self.use_template:
            raise NotImplementedError("Template Embedder is not implemented yet")
        if self.use_msa:
            raise NotImplementedError("MSA Module is not implemented yet")

        self.pairmixer_module: PairmixerStack = PairmixerStack(
            channel_z=cfg.channel_z,
            num_blocks=cfg.pairmixer_blocks,
            dropout=cfg.dropout,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )

        self.pairformer_module: PairformerStack = PairformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_blocks=cfg.num_blocks - cfg.pairmixer_blocks,
            num_heads_attn=cfg.num_heads_attn,
            num_heads_tri_attn=cfg.num_heads_tri_attn,
            dropout=cfg.dropout,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )

        self.layernorm_s = LayerNorm(cfg.channel_s)
        self.layernorm_z = LayerNorm(cfg.channel_z)
        self.linear_s = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")

    def do_compile(self):
        self.pairmixer_module = torch.compile(
            self.pairmixer_module, dynamic=False, fullgraph=False
        )  # type: ignore
        self.pairformer_module = torch.compile(
            self.pairformer_module, dynamic=False, fullgraph=False
        )  # type: ignore

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_recycles: int,
        use_cuequiv_kernels: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass."""
        if not self.training:
            if z_init.shape[1] > self.chunk_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        s_hat = torch.zeros_like(s_init)
        z_hat = torch.zeros_like(z_init)

        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i == num_recycles

            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                # Line 8
                z = z_init + self.linear_z(self.layernorm_z(z_hat))

                # Line 9: TemplateEmbedder
                if self.use_template:
                    raise NotImplementedError("Template Embedder is not implemented yet")

                # Line 10: MSAModule
                if self.use_msa:
                    raise NotImplementedError("MSA Module is not implemented yet")

                # Line 11
                s = s_init + self.linear_s(self.layernorm_s(s_hat))

                # Line 12
                # Revert to uncompiled version for validation
                if self.is_compiled and not self.training:
                    pairmixer_module = self.pairmixer_module._orig_mod  # noqa: SLF001
                    pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                else:
                    pairmixer_module = self.pairmixer_module
                    pairformer_module = self.pairformer_module

                s, z = pairmixer_module(
                    s,
                    z,
                    mask=f_input.token.pad_mask,
                    use_cuequiv_kernels=self.kernel_config.cuequivariance,
                )

                s, z = pairformer_module(
                    s,
                    z,
                    mask=f_input.token.pad_mask,
                    chunk_size_tri_attn=chunk_size_tri_attn,
                    use_cuequiv_kernels=self.kernel_config.cuequivariance,
                )

                # Line 13
                s_hat, z_hat = s, z

        s_trunk, z_trunk = s_hat, z_hat
        return s_trunk, z_trunk
