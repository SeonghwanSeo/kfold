"""KFold trunk module."""

import dataclasses

import torch
import torch.nn as nn

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.model.layers.kfold.plm_module import PLMModule
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
    use_separate_projections: bool = True
    use_qk_norm: bool = False
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class PairformerConfig:
    num_heads_attn: int = 16
    num_heads_tri_attn: int = 4
    num_blocks: int = 48
    dropout: float = 0.25
    # Proteina-style QK normalization (LayerNorm on Q and K before head split)
    use_qk_norm: bool = False
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
        channel_s_plm : int
            The token single embedding size for the PLM module.
        channel_z_plm : int
            The token attention map size for the PLM module.
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

        # PLM dimensions.
        channel_s_plm: int = 1152
        channel_z_plm: int = 648

        # plm module
        plm_module: PLMModuleConfig = dataclasses.field(default_factory=PLMModuleConfig)

        # pairformer
        pairformer: PairformerConfig = dataclasses.field(default_factory=PairformerConfig)

        # Proteina-style register tokens.
        num_register_tokens: int = 0
        register_token_init_std: float = 0.05

        # other options
        tri_attn_chunk_threshold: int = 384

    def __init__(self, cfg: Config, kernel_config=None):
        """Initialize the KFoldTrunk module."""
        super().__init__(cfg, kernel_config)

        # PLM module
        self.plm_module: PLMModule = PLMModule(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            channel_s_plm=cfg.channel_s_plm,
            num_heads_attn=cfg.plm_module.num_heads_attn,
            num_heads_tri_attn=cfg.plm_module.num_heads_tri_attn,
            num_blocks=cfg.plm_module.num_blocks,
            dropout_plm=cfg.plm_module.dropout_plm,
            dropout_z=cfg.plm_module.dropout_z,
            use_separate_projections=cfg.plm_module.use_separate_projections,
            use_qk_norm=cfg.plm_module.use_qk_norm,
            blocks_per_ckpt=cfg.plm_module.blocks_per_ckpt,
        )

        # Pairformer module
        self.pairformer_module: PairformerStack = PairformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_heads_attn=cfg.pairformer.num_heads_attn,
            num_heads_tri_attn=cfg.pairformer.num_heads_tri_attn,
            num_blocks=cfg.pairformer.num_blocks,
            dropout=cfg.pairformer.dropout,
            use_qk_norm=cfg.pairformer.use_qk_norm,
            blocks_per_ckpt=cfg.pairformer.blocks_per_ckpt,
        )

        # For recycling
        self.layernorm_s = LayerNorm(cfg.channel_s)
        self.layernorm_z = LayerNorm(cfg.channel_z)
        self.linear_s = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")

        # Projections from PLM features to trunk features.
        # TODO: if we consider two separate plms for intra- and inter-chain attentions,
        # we may want to have separate projections for s_plm and z_plm.
        self.proj_plm_to_z_init = nn.Sequential(
            LayerNorm(cfg.channel_z_plm, create_offset=False),
            LinearNoBias(cfg.channel_z_plm, cfg.channel_z, init="relu"),
            nn.ReLU(),
            LinearNoBias(cfg.channel_z, cfg.channel_z, init="default"),
        )

        # For the skip connection from PLM features to s_trunk output.
        self.proj_plm_to_s_trunk = LinearNoBias(
            cfg.channel_s_plm, cfg.channel_s, init="final"
        )

        # Proteina-style register tokens (learnable sequence-level registers).
        self.num_register_tokens: int = cfg.num_register_tokens
        assert self.num_register_tokens >= 0, "num_register_tokens must be non-negative"
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.empty(self.num_register_tokens, cfg.channel_s)
            )
            nn.init.normal_(
                self.register_tokens, mean=0.0, std=float(cfg.register_token_init_std)
            )
        else:
            self.register_tokens = None

        # Other options
        self.chunk_threshold: int = cfg.tri_attn_chunk_threshold

    def do_compile(self, mode: str = "default"):
        """Compile the trunk module."""
        # NOTE: you should compile the submodules inside the trunk
        # since the computation graph is changed depending on the
        # number of recycling steps. Thus, compile the sub module
        # instead of the whole trunk module.
        self.plm_module = torch.compile(
            self.plm_module,
            mode=mode,
            dynamic=False,
            fullgraph=False,
        )  # type: ignore
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

        # Get PLM features
        assert "s_plm" in kwargs, "PLM features s_plm must be provided in kwargs"
        assert "z_plm" in kwargs, "PLM features z_plm must be provided in kwargs"
        s_plm: torch.Tensor = kwargs["s_plm"]
        z_plm: torch.Tensor = kwargs["z_plm"]

        # Add z_plm to z_init before trunk.
        z_init = z_init + self.proj_plm_to_z_init(z_plm)

        # === Proteina-style register tokens (optional) ===
        mask = f_input.token.pad_mask
        asym_id = f_input.token.asym_id
        s_inputs, s_init, s_plm, z_init, asym_id, mask = self._extend_registers(
            s_inputs, s_init, s_plm, z_init, asym_id, mask
        )

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

                s_hat, z_hat = self._run_trunk(
                    s=s,
                    z=z,
                    s_inputs=s_inputs,
                    s_plm=s_plm,
                    asym_id=asym_id,
                    mask=mask,
                    chunk_size_tri_attn=chunk_size_tri_attn,
                )

        # Skip connection to s_trunk
        s_hat = s_hat + self.proj_plm_to_s_trunk(s_plm)

        # Remove register tokens before returning.
        s_hat, z_hat = self._undo_registers(s_hat, z_hat)
        return {"s_trunk": s_hat, "z_trunk": z_hat}

    def _run_trunk(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        s_plm: torch.Tensor,
        asym_id: torch.Tensor,
        mask: torch.Tensor,
        chunk_size_tri_attn: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Revert to uncompiled version for validation
        pairformer_module: PairformerStack
        plm_module: PLMModule
        if self.is_compiled and not self.training:
            pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
            plm_module = self.plm_module._orig_mod  # noqa: SLF001
        else:
            pairformer_module = self.pairformer_module  # noqa: SLF001
            plm_module = self.plm_module  # noqa: SLF001

        z = plm_module(
            z,
            s_inputs,
            s_plm,
            asym_id,
            mask,
            chunk_size_tri_attn=chunk_size_tri_attn,
            use_cuequiv_kernels=self.kernel_config.cuequivariance,
        )
        s, z = pairformer_module(
            s,
            z,
            mask=mask,
            chunk_size_tri_attn=chunk_size_tri_attn,
            use_cuequiv_kernels=self.kernel_config.cuequivariance,
        )
        return s, z

    def _extend_registers(
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        s_plm: torch.Tensor,
        z_init: torch.Tensor,
        asym_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Prepend register tokens (Proteina-style)."""
        R = self.num_register_tokens
        if R <= 0:
            return s_inputs, s_init, s_plm, z_init, asym_id, mask

        device = s_init.device

        assert self.register_tokens is not None
        B, L, _ = s_init.shape

        s_inputs_pad = torch.zeros(
            (B, L + R, s_inputs.shape[-1]), device=device, dtype=s_inputs.dtype
        )
        s_inputs_pad[:, R:] = s_inputs

        s_plm_pad = torch.zeros(
            (B, L + R, s_plm.shape[-1]), device=device, dtype=s_plm.dtype
        )
        s_plm_pad[:, R:] = s_plm

        reg = self.register_tokens.to(device=device, dtype=s_init.dtype)
        reg = reg.unsqueeze(0).expand(B, -1, -1)  # [B, R, C_s]
        s_init_pad = torch.cat([reg, s_init], dim=1)  # [B, R+L, C_s]

        z_pad = torch.zeros(
            (B, L + R, L + R, z_init.shape[-1]), device=device, dtype=z_init.dtype
        )
        z_pad[:, R:, R:] = z_init

        asym_id_pad = torch.zeros((B, L + R), device=device, dtype=asym_id.dtype)
        asym_id_pad[:, R:] = asym_id

        mask_pad = torch.ones((B, L + R), device=device, dtype=mask.dtype)
        mask_pad[:, R:] = mask

        return s_inputs_pad, s_init_pad, s_plm_pad, z_pad, asym_id_pad, mask_pad

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
