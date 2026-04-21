"""KFold trunk module with priming stage before recycling"""

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.model.layers.kfold.plm_module import PLMEmbedder, PLMModule
from kfold.model.layers.primitives import LayerNorm, LinearNoBias
from kfold.utils.registry import TRUNK
from kfold.utils.torch import add

from .base import BaseTrunk
from .kfold_trunk import PairformerConfig, PLMModuleConfig


@TRUNK.register()
class KFoldTrunkPrime(BaseTrunk):
    class Config(BaseTrunk.Config):
        """Configuration for the KFoldTrunkPrime module.

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

        # plm module
        plm_module: PLMModuleConfig = dataclasses.field(default_factory=PLMModuleConfig)

        # pairformer
        pairformer: PairformerConfig = dataclasses.field(default_factory=PairformerConfig)

        # Proteina-style register tokens.
        num_register_tokens: int = 0
        register_token_init_std: float = 0.05

    def __init__(self, cfg: Config, kernel_config=None):
        """Initialize the KFoldTrunkPrime module."""
        super().__init__(cfg, kernel_config)

        # === PLM feature processing layers === #
        self.layernorm_seq_emb = LayerNorm(cfg.channel_seq_emb, create_offset=False)
        self.layernorm_struct_emb = LayerNorm(cfg.channel_struct_emb, create_offset=False)
        plm_input_dim = cfg.channel_seq_emb + cfg.channel_struct_emb

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

        # === Priming pass before recycling === #
        self.plm_embedder_prime: PLMEmbedder = PLMEmbedder(
            channel_s_input=cfg.channel_s,
            channel_plm_input=plm_input_dim,
            channel_plm=cfg.channel_plm,
        )
        self.plm_module_prime: PLMModule = PLMModule(
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
        # Pairformer module
        self.pairformer_stack_prime: PairformerStack = PairformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_heads_attn=cfg.pairformer.num_heads_attn,
            num_heads_tri_attn=cfg.pairformer.num_heads_tri_attn,
            num_blocks=cfg.pairformer.num_blocks,
            dropout=cfg.pairformer.dropout,
            use_qk_norm=cfg.pairformer.use_qk_norm,
            blocks_per_ckpt=cfg.pairformer.blocks_per_ckpt,
        )

        # === Refine Module with recycling === #
        # PLM module
        self.plm_embedder_refine: PLMEmbedder = PLMEmbedder(
            channel_s_input=cfg.channel_s,
            channel_plm_input=plm_input_dim,
            channel_plm=cfg.channel_plm,
        )
        self.plm_module_refine: PLMModule = PLMModule(
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
        # Pairformer module
        self.pairformer_stack_refine: PairformerStack = PairformerStack(
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
        self.layernorm_s_prime = LayerNorm(cfg.channel_s)
        self.layernorm_z_prime = LayerNorm(cfg.channel_z)
        self.linear_s_prime = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z_prime = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")
        self.layernorm_s = LayerNorm(cfg.channel_s)
        self.layernorm_z = LayerNorm(cfg.channel_z)
        self.linear_s = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")

        # For the skip connection from PLM features to s_trunk output.
        self.proj_plm_to_s_trunk = LinearNoBias(
            plm_input_dim, cfg.channel_s, init="final"
        )

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

    def _compile(self, **kwargs):
        """Compile the trunk module."""
        # NOTE: you should compile the submodules inside the trunk
        # since the computation graph is changed depending on the
        # number of recycling steps. Thus, compile the sub module
        # instead of the whole trunk module.
        self.plm_module_prime = torch.compile(self.plm_module_prime, **kwargs)
        self.pairformer_stack_prime = torch.compile(self.pairformer_stack_prime, **kwargs)
        self.plm_module_refine = torch.compile(self.plm_module_refine, **kwargs)
        self.pairformer_stack_refine = torch.compile(
            self.pairformer_stack_refine, **kwargs
        )

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
        inplace = not self.training

        # === Get PLM features === #
        seq_emb = self.layernorm_seq_emb(seq_emb)
        struct_emb = self.layernorm_struct_emb(struct_emb)
        plm_input = torch.cat([seq_emb, struct_emb], dim=-1)

        # Add PLM attention maps.
        z_init = add(z_init, self.proj_seq_attn_to_z_init(seq_attn), inplace)
        del seq_attn  # free memory

        mask = f_input.token.pad_mask
        asym_id = f_input.token.asym_id

        # === Proteina-style register tokens (optional) ===
        s_inputs, s_init, plm_input, z_init, asym_id, mask = self._extend_registers(
            s_inputs, s_init, plm_input, z_init, asym_id, mask
        )

        # === Priming pass before recycling === #
        s_plm_prime = self.plm_embedder_prime(s_inputs, plm_input)
        s_prime, z_prime = self._run_trunk(
            plm_module=self.plm_module_prime,
            pairformer_stack=self.pairformer_stack_prime,
            s=s_init,
            z=z_init,
            s_plm=s_plm_prime,
            asym_id=asym_id,
            mask=mask,
        )

        # === Refining loop with recycling === #
        s, z = s_prime.clone(), z_prime.clone()
        s_plm = self.plm_embedder_refine(s_inputs, plm_input)
        s_bias = self.linear_s_prime(self.layernorm_s_prime(s_prime))
        z_bias = self.linear_z_prime(self.layernorm_z_prime(z_prime))

        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i == num_recycles
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                # Recycle linear pass
                s = add(s, s_bias, enable_grad)
                z = add(z, z_bias, enable_grad)
                s = add(self.linear_s(self.layernorm_s(s)), s_init, enable_grad)
                z = add(self.linear_z(self.layernorm_z(z)), z_init, enable_grad)

                # Trunk
                s, z = self._run_trunk(
                    plm_module=self.plm_module_refine,
                    pairformer_stack=self.pairformer_stack_refine,
                    s=s,
                    z=z,
                    s_plm=s_plm,
                    asym_id=asym_id,
                    mask=mask,
                )

        # Skip connection to s_trunk
        s = add(s, self.proj_plm_to_s_trunk(plm_input), inplace)

        # === Revert register tokens === #
        s, z, z_prime = self._undo_registers(s, z, z_prime)

        return {"s_trunk": s, "z_trunk": z, "z_aug": z_prime}

    def _run_trunk(
        self,
        plm_module: PLMModule,
        pairformer_stack: PairformerStack,
        s: torch.Tensor,
        z: torch.Tensor,
        s_plm: torch.Tensor,
        asym_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_compiled and not self.training:
            pairformer_stack = pairformer_stack._orig_mod  # noqa: SLF001
            plm_module = plm_module._orig_mod  # noqa: SLF001

        z = plm_module(
            z,
            s_plm,
            asym_id,
            mask,
            use_cuequiv_kernels=self.kernel_config.cuequivariance,
        )
        s, z = pairformer_stack(
            s,
            z,
            mask,
            use_cuequiv_kernels=self.kernel_config.cuequivariance,
        )
        return s, z

    # === Proteina-style register tokens === #
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

        # [B, L, ...] -> [B, R+L, ...]
        s_inputs_pad = F.pad(s_inputs, (0, 0, R, 0))
        s_plm_pad = F.pad(s_plm, (0, 0, R, 0))
        asym_id_pad = F.pad(asym_id, (R, 0))  # asym_id is 1-based index.
        mask_pad = F.pad(mask, (R, 0), value=True)

        # [B, L, L, ...] -> [B, R+L, R+L, ...]
        z_init_pad = F.pad(z_init, (0, 0, R, 0, R, 0))

        # Add register tokens to the beginning of s_init.
        assert self.register_tokens is not None
        reg = self.register_tokens.to(s_init.dtype)
        reg = reg.unsqueeze(0).expand(s_init.shape[0], -1, -1)  # [B, R, C_s]
        s_init_pad = torch.cat([reg, s_init], dim=1)  # [B, R+L, C_s]

        return s_inputs_pad, s_init_pad, s_plm_pad, z_init_pad, asym_id_pad, mask_pad

    def _undo_registers(
        self, s_trunk: torch.Tensor, z_trunk: torch.Tensor, z_prime: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove register tokens from s/z outputs."""
        R = self.num_register_tokens
        if R <= 0:
            return s_trunk, z_trunk, z_prime
        return s_trunk[:, R:], z_trunk[:, R:, R:], z_prime[:, R:, R:]
