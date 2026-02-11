"""KFold trunk module.

Compared to the AlphaFold3 trunk (which comprises the MSAModule, TemplateModule,
and Pairformer), KFold replaces these components with custom modules designed to
incorporate apo structure information and evolutionary pre-trained sequence
features.

1. Feeding apo structure information
------------------------------------
There are four sources for apo structures:
1. Experimental apo structures
2. Experimental holo structures
3. Predicted apo structures (e.g., AlphaFold2, ESMFold)
4. Permuted structures from KFold's apo-permutation module.

Sources 1-3 provide multi-state information about the protein.
Source 4 provides local flexibility information.

2. Bidirectional information flow (Single <-> Pairwise)
-------------------------------------------------------
The InterformerStack is a modified version of the AlphaFold3 PairformerStack.
In InterformerStack, the information flow between single (s) and pairwise (z)
representations is fully bidirectional (s <-> z). This enables a more integrated
representation that captures the interplay between evolutionary features and
interaction features.

Sub-modules
-----------
The KFoldTrunk module consists of the following:
    - MultiStateModule (Modified Template Embedder, Not implemented yet):
        Directly uses multi-state apo coordinates.
        Input shape: (B, N_atom, N_apo, 3)
    - InterformerStack (Modified Pairformer Stack):
        Performs bidirectional updates between single (s) and pairwise (z)
        representations.
        Input shape: s: (B, L, c_s), z: (B, L, L, c_z)
"""

import dataclasses

import torch
import torch.nn as nn

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.kfold.interformer import InterformerStack
from kfold.model.layers.primitives import LayerNorm, LinearNoBias
from kfold.utils.registry import TRUNK

from .base import BaseTrunk


@dataclasses.dataclass(kw_only=True)
class InterformerConfig:
    num_heads_attn: int = 16
    num_heads_tri_attn: int = 4
    num_blocks: int = 48
    dropout: float = 0.25
    use_separate_projections: bool = True
    skip_tri_attn: bool = False
    # Proteina-style QK normalization (LayerNorm on Q and K before head split)
    use_qk_norm: bool = False


@TRUNK.register()
class KFoldTrunkV0(BaseTrunk):
    class Config(BaseTrunk.Config):
        """Configuration for the KFoldTrunk module.

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
        dropout : float, optional
            The dropout rate, by default 0.25
        tri_attn_chunk_threshold : int, optional
            The threshold for chunking in triangle attention, by default 384
        """

        channel_s: int = 384
        channel_z: int = 128

        # pairformer
        interformer: InterformerConfig = dataclasses.field(
            default_factory=InterformerConfig
        )

        # other options
        blocks_per_ckpt: int | None = None
        tri_attn_chunk_threshold: int = 384

        # Proteina-style register tokens.
        # These tokens are prepended to the sequence representation inside the trunk,
        # and removed before returning (so downstream structure modules remain
        # unchanged).
        num_register_tokens: int = 0
        register_token_init_std: float = 0.05
        # How register tokens participate in intra-chain masking:
        # - "all": register tokens are treated as intra with all chains (default).
        # - "separate": register tokens form their own chain.
        register_token_intra_mode: str = "all"

    def __init__(self, cfg: Config, kernel_config=None):
        """Initialize the MultiStateApoTrunk module."""
        super().__init__(cfg, kernel_config)
        self.pairformer_module: InterformerStack = InterformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_heads_attn=cfg.interformer.num_heads_attn,
            num_heads_tri_attn=cfg.interformer.num_heads_tri_attn,
            num_blocks=cfg.interformer.num_blocks,
            dropout=cfg.interformer.dropout,
            skip_tri_attn=cfg.interformer.skip_tri_attn,
            use_separate_projections=cfg.interformer.use_separate_projections,
            use_qk_norm=cfg.interformer.use_qk_norm,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )

        # For recycling
        self.layernorm_s = LayerNorm(cfg.channel_s)
        self.layernorm_z = LayerNorm(cfg.channel_z)
        self.linear_s = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")

        # Other options
        self.chunk_threshold: int = cfg.tri_attn_chunk_threshold

        # Proteina-style register tokens (learnable sequence-level registers).
        self.num_register_tokens: int = int(cfg.num_register_tokens)
        if self.num_register_tokens < 0:
            raise ValueError("num_register_tokens must be >= 0")
        self.register_token_intra_mode = str(cfg.register_token_intra_mode)
        if self.register_token_intra_mode not in {"all", "separate"}:
            raise ValueError(
                "register_token_intra_mode must be one of: 'all', 'separate'"
            )
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.empty(self.num_register_tokens, cfg.channel_s)
            )
            nn.init.normal_(
                self.register_tokens,
                mean=0.0,
                std=float(cfg.register_token_init_std),
            )
        else:
            self.register_tokens = None

    def do_compile(self, mode: str = "default"):
        """Compile the trunk module."""
        # NOTE: you should compile the submodules inside the trunk
        # since the computation graph is changed depending on the
        # number of recycling steps. Thus, compile the sub module
        # instead of the whole trunk module.
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
        **kwargs,
    ) -> dict[str, torch.Tensor]:
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

        # === Proteina-style register tokens (optional) ===
        # We extend (s, z, mask, intra_mask) internally, and slice them out before
        # return.
        mask_real = f_input.token.pad_mask
        intra_mask_real = (
            f_input.token.asym_id[..., :, None] == f_input.token.asym_id[..., None, :]
        )  # [..., L, L]

        s_init, z_init, mask, intra_mask = self._extend_registers(
            s_init, z_init, mask_real, intra_mask_real
        )

        # Revert to uncompiled version for validation
        pairformer_module: InterformerStack
        if self.is_compiled and not self.training:
            pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
        else:
            pairformer_module = self.pairformer_module

        # z_hat, s_hat = 0, 0
        s_hat = torch.zeros_like(s_init)
        z_hat = torch.zeros_like(z_init)

        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i == num_recycles

            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                s = s_init + self.linear_s(self.layernorm_s(s_hat))
                z = z_init + self.linear_z(self.layernorm_z(z_hat))

                s, z = pairformer_module(
                    s,
                    z,
                    mask=mask,
                    intra_mask=intra_mask,
                    chunk_size_tri_attn=chunk_size_tri_attn,
                    use_cuequiv_kernels=self.kernel_config.cuequivariance,
                )

                s_hat, z_hat = s, z

        # Remove register tokens before returning.
        s_hat, z_hat = self._undo_registers(s_hat, z_hat)
        return {"s_hat": s_hat, "z_hat": z_hat}

    def _extend_registers(
        self,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        mask: torch.Tensor,
        intra_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepend register tokens to s/z/mask/intra_mask (Proteina-style)."""
        R = self.num_register_tokens
        if R <= 0:
            return s_init, z_init, mask, intra_mask

        assert self.register_tokens is not None
        B, L, _ = s_init.shape
        reg = self.register_tokens.to(dtype=s_init.dtype, device=s_init.device)
        reg = reg.unsqueeze(0).expand(B, -1, -1)  # [B, R, C_s]
        s_init = torch.cat([reg, s_init], dim=1)  # [B, R+L, C_s]

        z_pad = torch.zeros(
            (B, L + R, L + R, z_init.shape[-1]),
            device=z_init.device,
            dtype=z_init.dtype,
        )
        z_pad[:, R:, R:] = z_init
        z_init = z_pad

        reg_mask = torch.ones((B, R), device=mask.device, dtype=mask.dtype)
        mask = torch.cat([reg_mask, mask], dim=-1)  # [B, R+L]

        intra_pad = torch.zeros(
            (B, L + R, L + R),
            device=intra_mask.device,
            dtype=intra_mask.dtype,
        )
        intra_pad[:, R:, R:] = intra_mask
        if self.register_token_intra_mode == "all":
            intra_pad[:, :R, :] = True
            intra_pad[:, :, :R] = True
        else:
            intra_pad[:, :R, :R] = True
        intra_mask = intra_pad
        return s_init, z_init, mask, intra_mask

    def _undo_registers(
        self,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove register tokens from s/z outputs."""
        R = self.num_register_tokens
        if R <= 0:
            return s_trunk, z_trunk
        return s_trunk[:, R:], z_trunk[:, R:, R:]
