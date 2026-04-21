import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.model.layers.primitives import LayerNorm, LinearNoBias
from kfold.utils.registry import TRUNK
from kfold.utils.torch import add

from .base import BaseTrunk


@TRUNK.register()
class AF3PairformerTrunk(BaseTrunk):
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
        num_heads_attn : int, optional
            The number of attention heads, by default 16
        num_heads_tri_attn : int, optional
            The number of triangle attention heads, by default 4
        num_blocks : int
            The number of blocks.
        dropout : float, optional
            The dropout rate, by default 0.25
        use_template: bool, optional
            Whether to use template, by default False
        use_msa: bool, optional
            Whether to use MSA, by default False
        """

        channel_s: int = 384
        channel_z: int = 128
        num_heads_attn: int = 16
        num_heads_tri_attn: int = 4
        num_blocks: int = 48
        dropout: float = 0.25
        use_msa: bool = False
        use_template: bool = False
        blocks_per_ckpt: int | None = None

    def __init__(self, cfg: Config, kernel_config):
        """Initialize the Pairformer module."""
        super().__init__(cfg, kernel_config)
        self.use_msa: bool = cfg.use_msa
        self.use_template: bool = cfg.use_template

        if self.use_template:
            raise NotImplementedError(
                "Template Embedder is not implemented yet (Boltz1 does not support too)"
            )
        if self.use_msa:
            # TODO: Implement MSA Module
            raise NotImplementedError("MSA Module is not implemented yet")

        # Pairformer stack
        self.pairformer_stack: PairformerStack = PairformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_heads_attn=cfg.num_heads_attn,
            num_heads_tri_attn=cfg.num_heads_tri_attn,
            num_blocks=cfg.num_blocks,
            dropout=cfg.dropout,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )

        # For recycling
        self.layernorm_s = LayerNorm(cfg.channel_s)
        self.layernorm_z = LayerNorm(cfg.channel_z)
        self.linear_s = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")

    def _compile(self, **kwargs):
        """Compile the trunk module."""
        # NOTE: you should compile the submodules inside the trunk
        # since the computation graph is changed depending on the
        # number of recycling steps.
        self.pairformer_stack = torch.compile(self.pairformer_stack, **kwargs)

    def get_pairformer_stack(self, no_compile: bool = False) -> PairformerStack:
        if self.is_compiled and no_compile:
            return self.pairformer_stack._orig_mod  # type: ignore
        else:
            return self.pairformer_stack

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_recycles: int,
    ) -> dict[str, torch.Tensor]:
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
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        mask = f_input.token.pad_mask

        # Revert to uncompiled version for validation
        pairformer_stack: PairformerStack = self.get_pairformer_stack(not self.training)

        # Line 7-14
        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i == num_recycles
            _inplace = not enable_grad

            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                # Line 8
                z = add(self.linear_z(self.layernorm_z(z)), z_init, _inplace)

                # Line 9: TemplateEmbedder
                if self.use_template:
                    raise NotImplementedError("Template Embedder is not implemented yet")

                # Line 10: MSAModule
                if self.use_msa:
                    raise NotImplementedError("MSA Module is not implemented yet")

                # Line 11
                s = add(self.linear_s(self.layernorm_s(s)), s_init, _inplace)

                # Line 12
                s, z = pairformer_stack(
                    s, z, mask, use_cuequiv_kernels=self.kernel_config.cuequivariance
                )

                # Line 13
                s_hat, z_hat = s, z

        return {"s_trunk": s_hat, "z_trunk": z_hat}
