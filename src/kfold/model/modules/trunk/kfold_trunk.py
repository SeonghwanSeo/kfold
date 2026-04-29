"""KFold trunk module."""

import dataclasses

import torch
import torch.nn as nn

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.model.layers.kfold.plm_module import PLMEmbedder, PLMModule
from kfold.model.layers.primitives import LayerNorm, LinearNoBias
from kfold.utils.registry import TRUNK

from .base import BaseTrunk


@dataclasses.dataclass(kw_only=True)
class PLMModuleConfig:
    num_heads_attn: int = 16
    num_heads_tri_attn: int = 4
    num_blocks: int = 4
    dropout_plm: float = 0.15
    dropout_z: float = 0.25
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class PairformerConfig:
    num_heads_attn: int = 16
    num_heads_tri_attn: int = 4
    num_blocks: int = 48
    dropout: float = 0.25
    blocks_per_ckpt: int | None = None


@TRUNK.register()
class KFoldTrunk(BaseTrunk):
    class Config(BaseTrunk.Config):
        """Configuration for the KFoldTrunk module.

        Parameters
        ----------
        channel_s : int
            The token single embedding size.
        channel_z : int
            The token pairwise embedding size.
        channel_plm : int
            The hidden dimension for the PLMModule.
        channel_seq_emb : int
            The sequence embedding size for the PLM module.
        channel_struct_emb : int
            The structure embedding size for the PLM module.
        channel_seq_attn : int
            The number of attention maps for the PLM module.
        num_heads_attn : int, optional
            The number of attention heads, by default 16
        num_heads_tri_attn : int, optional
            The number of triangle attention heads, by default 4
        dropout : float, optional
            The dropout rate, by default 0.25
        """

        channel_s: int = 384
        channel_z: int = 128

        # PLM dimensions.
        channel_plm: int = 768
        channel_seq_emb: int = 1152
        channel_struct_emb: int = 1536
        channel_seq_attn: int = 648

        # plm module
        plm_module: PLMModuleConfig = dataclasses.field(default_factory=PLMModuleConfig)

        # pairformer
        pairformer: PairformerConfig = dataclasses.field(default_factory=PairformerConfig)

    def __init__(self, cfg: Config, kernel_config=None):
        """Initialize the KFoldTrunk module."""
        super().__init__(cfg, kernel_config)

        # === PLM feature processing layers === #
        self.layernorm_seq_emb = LayerNorm(cfg.channel_seq_emb, create_offset=False)
        self.layernorm_struct_emb = LayerNorm(cfg.channel_struct_emb, create_offset=False)
        channel_plm_input = cfg.channel_seq_emb + cfg.channel_struct_emb

        # Projections from PLM features to trunk features.
        # NOTE (Seonghwan): LayerNorm is applied for scalability to sequence length,
        # as the scale of attention maps is reduced by sequence length.
        # TODO: If we consider two separate plms for intra- and inter-chain attentions,
        # we may want to have separate projections for s_plm and z_plm.
        self.proj_seq_attn_to_z_init = nn.Sequential(
            LayerNorm(cfg.channel_seq_attn, create_offset=False),
            LinearNoBias(cfg.channel_seq_attn, cfg.channel_z, init="relu"),
            nn.ReLU(),
            LinearNoBias(cfg.channel_z, cfg.channel_z, init="final"),
        )

        # === PLM Module === #
        self.plm_embedder: PLMEmbedder = PLMEmbedder(
            channel_s_input=cfg.channel_s,
            channel_plm_input=channel_plm_input,
            channel_plm=cfg.channel_plm,
        )
        self.plm_module: PLMModule = PLMModule(
            channel_z=cfg.channel_z,
            channel_plm=cfg.channel_plm,
            num_heads_attn=cfg.plm_module.num_heads_attn,
            num_heads_tri_attn=cfg.plm_module.num_heads_tri_attn,
            num_blocks=cfg.plm_module.num_blocks,
            dropout_plm=cfg.plm_module.dropout_plm,
            dropout_z=cfg.plm_module.dropout_z,
            use_qk_norm=cfg.plm_module.use_qk_norm,
            blocks_per_ckpt=cfg.plm_module.blocks_per_ckpt,
        )

        # === Pairformer Module === #
        self.pairformer_stack: PairformerStack = PairformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_heads_attn=cfg.pairformer.num_heads_attn,
            num_heads_tri_attn=cfg.pairformer.num_heads_tri_attn,
            num_blocks=cfg.pairformer.num_blocks,
            dropout=cfg.pairformer.dropout,
            use_qk_norm=cfg.pairformer.use_qk_norm,
            blocks_per_ckpt=cfg.pairformer.blocks_per_ckpt,
        )

        # === Recycling === #
        self.layernorm_s = LayerNorm(cfg.channel_s)
        self.layernorm_z = LayerNorm(cfg.channel_z)
        self.linear_s = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")

        # === Skip connection === #
        self.proj_plm_to_s_trunk = LinearNoBias(
            channel_plm_input, cfg.channel_s, init="final"
        )

    def _compile(self, **kwargs):
        """Compile the trunk module."""
        self.plm_module = torch.compile(self.plm_module, **kwargs)
        self.pairformer_stack = torch.compile(self.pairformer_stack, **kwargs)

    def get_plm_module(self, no_compile: bool = False) -> PLMModule:
        """Get the PLMModule."""
        if self.is_compiled and no_compile:
            return self.plm_module._orig_mod  # type: ignore
        return self.plm_module

    def get_pairformer_stack(self, no_compile: bool = False) -> PairformerStack:
        """Get the PairformerStack."""
        if self.is_compiled and no_compile:
            return self.pairformer_stack._orig_mod  # type: ignore
        return self.pairformer_stack

    def forward(  # type: ignore
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_recycles: int,
        seq_emb: torch.Tensor,
        seq_attn: torch.Tensor,
        struct_emb: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s_inputs : torch.Tensor
            Tensor of shape (B, L, C_s) containing input single features
        s_init: torch.Tensor
            Tensor of shape (B, L, C_s) containing initial single representation
        z_init: torch.Tensor
            Tensor of shape (B, L, L, C_s) containing initial pair representation
        f_input : FoldingInput
            The input features.
        num_recycles : int
            The number of recycling steps.
        seq_emb : torch.Tensor
            Tensor of shape (B, L, channel_seq_emb) containing sequence embeddings.
        seq_attn : torch.Tensor
            Tensor of shape (B, L, L, channel_seq_attn) containing attention maps
            from the sequence encoder.
        struct_emb : torch.Tensor
            Tensor of shape (B, L, channel_struct_emb) containing structure embeddings.

        Returns
        -------
        s_trunk: torch.Tensor
            The updated tensor of shape (B, L, c_s).
        z_trunk: torch.Tensor
            The updated tensor of shape (B, L, L, c_z).
        """
        # === Get PLM features === #
        seq_emb = self.layernorm_seq_emb(seq_emb)
        struct_emb = self.layernorm_struct_emb(struct_emb)
        plm_inputs = torch.cat([seq_emb, struct_emb], dim=-1)
        mask = f_input.token.pad_mask
        asym_id = f_input.token.asym_id

        # === Main trunk iteration with recycling === #
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        s_plm = self.plm_embedder(s_inputs, plm_inputs)

        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i == num_recycles
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                # Recycling
                s = s_init + self.linear_s(self.layernorm_s(s))
                z = z_init + self.linear_z(self.layernorm_z(z))

                # Run trunk
                s, z = self._run_trunk(s, z, s_plm, asym_id, mask)

        # Skip connection to s_trunk
        s = s + self.proj_plm_to_s_trunk(plm_inputs)

        return s, z

    def _run_trunk(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        s_plm: torch.Tensor,
        asym_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run trunk body"""
        # Revert to uncompiled version for validation
        pairformer_stack = self.get_pairformer_stack(not self.training)
        plm_module = self.get_plm_module(not self.training)

        use_cuequiv_kernels = self.kernel_config.get("cuequivariance", False)
        z = plm_module(z, s_plm, asym_id, mask, use_cuequiv_kernels=use_cuequiv_kernels)
        s, z = pairformer_stack(s, z, mask, use_cuequiv_kernels=use_cuequiv_kernels)
        return s, z
