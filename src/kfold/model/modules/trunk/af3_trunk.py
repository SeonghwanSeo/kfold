import torch

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.model.layers.primitives import LayerNorm, LinearNoBias
from kfold.utils.registry import TRUNK

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
        use_cuequiv_kernels : bool, optional
            Whether to use cuequivariance kernels, by default False
        tri_attn_chunk_threshold : int, optional
            The threshold for chunking in triangle attention, by default 384
        """

        channel_s: int = 384
        channel_z: int = 128
        num_heads_attn: int = 16
        num_heads_tri_attn: int = 4
        num_blocks: int = 48
        dropout: float = 0.25
        use_msa: bool = False
        use_template: bool = False
        use_cuequiv_kernels: bool = False
        blocks_per_ckpt: int | None = None
        tri_attn_chunk_threshold: int = 384

    def __init__(self, cfg: Config):
        """Initialize the Pairformer module."""
        super().__init__(cfg)
        self.use_msa: bool = cfg.use_msa
        self.use_template: bool = cfg.use_template
        self.use_cuequiv_kernels: bool = cfg.use_cuequiv_kernels
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
        s_hat = torch.zeros_like(s_init)
        z_hat = torch.zeros_like(z_init)

        for i in range(1, num_cycles + 1):
            enable_grad = self.training and i == num_cycles

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
                pairformer_module: PairformerStack
                if self.is_compiled and not self.training:
                    pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                else:
                    pairformer_module = self.pairformer_module

                s, z = pairformer_module(
                    s,
                    z,
                    mask=f_input.token.pad_mask,
                    chunk_size_tri_attn=chunk_size_tri_attn,
                    use_cuequiv_attn=self.use_cuequiv_kernels,
                    use_cuequiv_mul=self.use_cuequiv_kernels,
                )

                # Line 13
                s_hat, z_hat = s, z

        s_trunk, z_trunk = s_hat, z_hat
        return s_trunk, z_trunk
