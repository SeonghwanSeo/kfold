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
        channel_plm : int
            The hidden dimension for the PLMModule.
        channel_seq_emb : int
            The sequence embedding size for the PLM module.
        channel_struct_emb : int
            The structure embedding size for the PLM module.
        channel_seq_attn : int
            The number of attention maps for the PLM module.
        use_attn : bool
            Whether to use attention in the PLM module.
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
        use_attn: bool = True

        # plm module
        plm_module: PLMModuleConfig = dataclasses.field(default_factory=PLMModuleConfig)

        # pairformer
        pairformer: PairformerConfig = dataclasses.field(default_factory=PairformerConfig)

        # Proteina-style register tokens.
        num_register_tokens: int = 0
        register_token_init_std: float = 0.05

    def __init__(self, cfg: Config, kernel_config=None):
        """Initialize the KFoldTrunk module."""
        super().__init__(cfg, kernel_config)

        # === PLM feature processing layers === #
        self.layernorm_seq_emb = LayerNorm(cfg.channel_seq_emb, create_offset=False)
        self.layernorm_struct_emb = LayerNorm(cfg.channel_struct_emb, create_offset=False)
        plm_input_dim = cfg.channel_seq_emb + cfg.channel_struct_emb

        # Projections from PLM features to trunk features.
        # TODO: if we consider two separate plms for intra- and inter-chain attentions,
        # we may want to have separate projections for s_plm and z_plm.
        self.use_attn = cfg.use_attn
        if self.use_attn:
            # (Seonghwan) LayerNorm is applied for scalability to sequence length,
            # as the scale of attention maps is reduced by sequence length.
            self.proj_seq_attn_to_z_init = nn.Sequential(
                LayerNorm(cfg.channel_seq_attn, create_offset=False),
                LinearNoBias(cfg.channel_seq_attn, cfg.channel_z, init="relu"),
                nn.ReLU(),
                LinearNoBias(cfg.channel_z, cfg.channel_z, init="final"),
            )

        # For the skip connection from PLM features to s_trunk output.
        self.proj_plm_to_s_trunk = LinearNoBias(
            plm_input_dim, cfg.channel_s, init="final"
        )

        # === PLM Module === #
        self.plm_embedder_refine: PLMEmbedder = PLMEmbedder(
            channel_s_input=cfg.channel_s,
            channel_plm_input=plm_input_dim,
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

        # === Recycling === #
        self.layernorm_s = LayerNorm(cfg.channel_s)
        self.layernorm_z = LayerNorm(cfg.channel_z)
        self.linear_s = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")

        # Proteina-style register tokens (learnable sequence-level registers).
        self.num_register_tokens: int = cfg.num_register_tokens
        if self.num_register_tokens < 0:
            raise ValueError("num_register_tokens must be >= 0")
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.empty(self.num_register_tokens, cfg.channel_s)
            )
            nn.init.normal_(
                self.register_tokens, mean=0.0, std=float(cfg.register_token_init_std)
            )
        else:
            self.register_tokens = None

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
        # Get PLM features
        for k in ["seq_emb", "seq_attn", "struct_emb"]:
            if k not in kwargs:
                raise ValueError(f"Missing required PLM feature: {k}")
        seq_emb: torch.Tensor = kwargs["seq_emb"]
        seq_attn: torch.Tensor = kwargs["seq_attn"]
        struct_emb: torch.Tensor = kwargs["struct_emb"]

        seq_emb = self.layernorm_seq_emb(seq_emb)
        struct_emb = self.layernorm_struct_emb(struct_emb)
        plm_input = torch.cat([seq_emb, struct_emb], dim=-1)

        if self.use_attn:
            # Feed attention maps to initialize z_init.
            z_init = z_init + self.proj_seq_attn_to_z_init(seq_attn)
        del seq_attn  # free memory

        # === Proteina-style register tokens (optional) ===
        mask = f_input.token.pad_mask
        asym_id = f_input.token.asym_id
        s_inputs, s_init, plm_input, z_init, asym_id, mask = self._extend_registers(
            s_inputs, s_init, plm_input, z_init, asym_id, mask
        )

        # z_hat, s_hat = 0, 0
        s_hat = torch.zeros_like(s_init)
        z_hat = torch.zeros_like(z_init)
        s_plm = self.plm_embedder_refine(s_inputs, plm_input)

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
                    s_plm=s_plm,
                    asym_id=asym_id,
                    mask=mask,
                )

        # Skip connection to s_trunk
        s_hat = s_hat + self.proj_plm_to_s_trunk(plm_input)

        # Remove register tokens before returning.
        s_hat, z_hat = self._undo_registers(s_hat, z_hat)
        return {"s_trunk": s_hat, "z_trunk": z_hat}

    def _run_trunk(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        s_plm: torch.Tensor,
        asym_id: torch.Tensor,
        mask: torch.Tensor,
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
            s_plm,
            asym_id,
            mask,
            use_cuequiv_kernels=self.kernel_config.cuequivariance,
        )
        s, z = pairformer_module(
            s,
            z,
            mask=mask,
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
