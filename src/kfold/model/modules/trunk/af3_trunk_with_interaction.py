"""AlphaFold3 Pairformer trunk with interaction support."""

from __future__ import annotations

import torch

import kfold.constants as C
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.model.layers.primitives import LayerNorm, LinearNoBias
from kfold.utils.registry import TRUNK

from .base import BaseTrunk


@TRUNK.register()
class AF3PairformerTrunkWithInteraction(BaseTrunk):
    """AlphaFold3 Pairformer trunk with interaction support."""

    class Config(BaseTrunk.Config):
        channel_s: int = 384
        channel_z: int = 128
        num_heads_attn: int = 16
        num_heads_tri_attn: int = 4
        num_blocks: int = 48
        dropout: float = 0.25
        use_msa: bool = False
        use_template: bool = False
        blocks_per_ckpt: int | None = None
        tri_attn_chunk_threshold: int = 384
        # Interaction-specific configs
        use_interaction: bool = True
        interaction_mode: str = "fused"
        channel_z_interaction: int = 32
        interaction_num_heads: int = 2

    def __init__(self, cfg: Config, kernel_config):
        super().__init__(cfg, kernel_config)
        self.use_msa: bool = cfg.use_msa
        self.use_template: bool = cfg.use_template
        self.chunk_threshold: int = cfg.tri_attn_chunk_threshold
        self.use_interaction = cfg.use_interaction
        self.interaction_mode = cfg.interaction_mode

        if self.use_template:
            raise NotImplementedError(
                "Template Embedder is not implemented yet (Boltz1 does not support too)"
            )
        if self.use_msa:
            raise NotImplementedError("MSA Module is not implemented yet")

        if self.use_interaction and self.interaction_mode == "separate":
            interaction_dim = cfg.channel_z_interaction + C.NUM_PAIR_INTERACTION_TYPES
            self.linear_z_concat = LinearNoBias(
                cfg.channel_z + interaction_dim, cfg.channel_z
            )

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

    def do_compile(self, mode: str = "default"):
        self.pairformer_module = torch.compile(
            self.pairformer_module,
            mode=mode,
            dynamic=False,
            fullgraph=False,
        )  # type: ignore

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_recycles: int,
        z_interaction_init: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            if z_init.shape[1] > self.chunk_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        if self.use_interaction and self.interaction_mode == "separate":
            assert z_interaction_init is not None, (
                "z_interaction_init is required for interaction_mode='separate'"
            )
            z_init = self._merge_interaction(z_init, z_interaction_init)

        return self._forward_fused(
            s_inputs,
            s_init,
            z_init,
            f_input,
            num_recycles,
            chunk_size_tri_attn,
        )

    def _merge_interaction(
        self, z_init: torch.Tensor, z_interaction_init: torch.Tensor
    ) -> torch.Tensor:
        expected_dim = self.cfg.channel_z_interaction + C.NUM_PAIR_INTERACTION_TYPES
        if z_interaction_init.ndim != 4 or z_interaction_init.shape[-1] != expected_dim:
            raise ValueError(
                "z_interaction_init has unexpected shape. Expected "
                f"[B, L, L, {expected_dim}], got {tuple(z_interaction_init.shape)}."
            )
        z_concat = torch.cat([z_init, z_interaction_init], dim=-1)
        return self.linear_z_concat(z_concat)

    def _forward_fused(
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_recycles: int,
        chunk_size_tri_attn: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Revert to uncompiled version for validation
        pairformer_module: PairformerStack
        if self.is_compiled and not self.training:
            pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
        else:
            pairformer_module = self.pairformer_module

        # Line 6, z_hat, s_hat = 0, 0
        s_hat = torch.zeros_like(s_init)
        z_hat = torch.zeros_like(z_init)

        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i == num_recycles

            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                if self.is_compiled and enable_grad and i > 0:
                    s_hat = s_hat.clone()
                    z_hat = z_hat.clone()

                z = z_init + self.linear_z(self.layernorm_z(z_hat))
                s = s_init + self.linear_s(self.layernorm_s(s_hat))

                s, z = pairformer_module(
                    s,
                    z,
                    mask=f_input.token.pad_mask,
                    chunk_size_tri_attn=chunk_size_tri_attn,
                    use_cuequiv_kernels=self.kernel_config.cuequivariance,
                )

                s_hat, z_hat = s, z

        return s_hat, z_hat
