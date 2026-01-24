import torch
import torch.nn as nn

import kfold.constants as C
from kfold.data.types.model_input import FoldingInput
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
        tri_attn_chunk_threshold : int, optional
            The threshold for chunking in triangle attention, by default 384

        use_interaction : bool, optional
            Whether to enable interaction features, by default False
        interaction_mode : str, optional
            How to handle interactions ("fused" or "separate"), by default "fused"
        channel_z_interaction : int, optional
            Interaction channel size (excluding type indicators), by default 32
        interaction_num_heads : int, optional
            Interaction attention heads (when applicable), by default 2

        use_qk_norm : bool, optional
            Whether to apply Proteina-style QK normalization (LayerNorm on Q/K
            before head split) inside pairformer attention blocks.
        num_register_tokens : int, optional
            Number of Proteina-style register tokens to prepend internally to the
            sequence representation. These tokens are removed before returning
            (so downstream structure modules remain unchanged).
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
        tri_attn_chunk_threshold: int = 384
        # Interaction-specific configs
        use_interaction: bool = False
        interaction_mode: str = "fused"
        channel_z_interaction: int = 32
        interaction_num_heads: int = 2

        # Proteina-style options
        use_qk_norm: bool = False
        num_register_tokens: int = 0
        register_token_init_std: float = 0.05

    def __init__(self, cfg: Config, kernel_config):
        """Initialize the Pairformer module."""
        super().__init__(cfg, kernel_config)
        self.use_msa: bool = cfg.use_msa
        self.use_template: bool = cfg.use_template
        self.chunk_threshold: int = cfg.tri_attn_chunk_threshold
        self.use_interaction: bool = cfg.use_interaction
        self.interaction_mode: str = cfg.interaction_mode

        if self.use_template:
            raise NotImplementedError(
                "Template Embedder is not implemented yet (Boltz1 does not support too)"
            )
        if self.use_msa:
            # TODO: Implement MSA Module
            raise NotImplementedError("MSA Module is not implemented yet")

        if self.use_interaction and self.interaction_mode == "separate":
            interaction_dim = cfg.channel_z_interaction + C.NUM_PAIR_INTERACTION_TYPES
            self.linear_z_concat = LinearNoBias(
                cfg.channel_z + interaction_dim,
                cfg.channel_z,
            )

        self.pairformer_module: PairformerStack = PairformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_heads_attn=cfg.num_heads_attn,
            num_heads_tri_attn=cfg.num_heads_tri_attn,
            num_blocks=cfg.num_blocks,
            dropout=cfg.dropout,
            use_qk_norm=cfg.use_qk_norm,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )

        # For recycling
        self.layernorm_s = LayerNorm(cfg.channel_s)
        self.layernorm_z = LayerNorm(cfg.channel_z)
        self.linear_s = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")

        # Proteina-style register tokens.
        self.num_register_tokens: int = int(cfg.num_register_tokens)
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
        self.pairformer_module = torch.compile(
            self.pairformer_module,
            mode=mode,
            dynamic=False,
            fullgraph=False,
        )  # type: ignore

    def _extend_registers(
        self,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Prepend register tokens to s/z/mask (Proteina-style)."""
        R = self.num_register_tokens
        if R <= 0:
            return s_init, z_init, mask, 0

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
        return s_init, z_init, mask, R

    def _undo_registers(
        self,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        num_register_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove register tokens from s/z outputs."""
        if num_register_tokens <= 0:
            return s_trunk, z_trunk
        return (
            s_trunk[:, num_register_tokens:],
            z_trunk[:, num_register_tokens:, num_register_tokens:],
        )

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
        z_interaction_init : torch.Tensor | None
            Optional interaction pair features. Required when
            ``interaction_mode == "separate"``.

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

        if self.use_interaction and self.interaction_mode == "separate":
            if z_interaction_init is None:
                raise ValueError(
                    "z_interaction_init is required for interaction_mode='separate'"
                )
            # Merge the auxiliary interaction tensor into the pair channels.
            z_init = self._merge_interaction(z_init, z_interaction_init)

        # === Proteina-style register tokens (optional) ===
        # We extend (s, z, mask) internally, and slice them out before return.
        mask_real = f_input.token.pad_mask
        s_init, z_init, mask, R = self._extend_registers(s_init, z_init, mask_real)

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
                    # Clone the tensors for compilation.
                    s_hat = s_hat.clone()
                    z_hat = z_hat.clone()

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
                s, z = pairformer_module(
                    s,
                    z,
                    mask=mask,
                    chunk_size_tri_attn=chunk_size_tri_attn,
                    use_cuequiv_kernels=self.kernel_config.cuequivariance,
                )

                # Line 13
                s_hat, z_hat = s, z

        # Remove register tokens before returning.
        return self._undo_registers(s_hat, z_hat, R)

    def _merge_interaction(
        self,
        z_init: torch.Tensor,
        z_interaction_init: torch.Tensor,
    ) -> torch.Tensor:
        """Merge interaction channels into the base pair representation."""
        expected_dim = self.cfg.channel_z_interaction + C.NUM_PAIR_INTERACTION_TYPES
        if z_interaction_init.ndim != 4 or z_interaction_init.shape[-1] != expected_dim:
            raise ValueError(
                "z_interaction_init has unexpected shape. Expected "
                f"[B, L, L, {expected_dim}], got {tuple(z_interaction_init.shape)}."
            )
        z_concat = torch.cat([z_init, z_interaction_init], dim=-1)
        return self.linear_z_concat(z_concat)
