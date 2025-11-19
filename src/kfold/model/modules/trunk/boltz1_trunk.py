import torch
import torch.nn as nn

from kfold.data.model_input import FoldingInput
from kfold.model.layers.boltz1 import initialize as init
from kfold.model.layers.boltz1.trunk import PairformerModule
from kfold.utils.registry import TRUNK

from .base import BaseTrunk


@TRUNK.register()
class Boltz1PairformerTrunk(BaseTrunk):
    """Boltz1 pairformer module."""

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
        tri_attn_chunk_threshold: int = 384

    def __init__(self, cfg: Config):
        """Initialize the Pairformer module."""
        super().__init__(cfg)
        self.use_msa: bool = cfg.use_msa
        self.use_template: bool = cfg.use_template
        self.chunk_threshold: int = cfg.tri_attn_chunk_threshold

        if self.use_template:
            raise NotImplementedError(
                "Template Embedder is not implemented yet (Boltz1 does not support too)"
            )
        if self.use_msa:
            # TODO: Implement MSA Module
            raise NotImplementedError("MSA Module is not implemented yet")

        self.pairformer_module = PairformerModule(
            token_s=cfg.channel_s,
            token_z=cfg.channel_z,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            pairwise_head_width=cfg.pairwise_head_width,
            pairwise_num_heads=cfg.pairwise_num_heads,
        )

        # For recycling
        # Normalization layers
        self.s_norm = nn.LayerNorm(cfg.channel_s)
        self.z_norm = nn.LayerNorm(cfg.channel_z)

        # Recycling projections
        self.s_recycle = nn.Linear(cfg.channel_s, cfg.channel_s, bias=False)
        self.z_recycle = nn.Linear(cfg.channel_z, cfg.channel_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

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

        mask = f_input.token.pad_mask.float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        for i in range(1, num_cycles + 1):
            enable_grad = self.training and i == num_cycles

            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                s = s_init + self.s_recycle(self.s_norm(s))
                z = z_init + self.z_recycle(self.z_norm(z))

                # Revert to uncompiled version for validation
                s, z = self.pairformer_module(s, z, mask, pair_mask)

        return s, z
