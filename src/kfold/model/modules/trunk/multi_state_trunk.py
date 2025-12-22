"""KFold trunk module.

Compared to AlphaFold3 trunk with MSAModule and TemplateModule,
KFold replace these modules with custom modules to feed apo information.

There are four source to get apo structures:
1. Experimental apo structures
2. Experimental holo structures
3. Predicted apo structures (e.g., AlphaFold2, ESMFold)
4. Permutated structures from KFold's apo-permutation module.

Source 1-3 provide the multi-state information of the protein.
Source 4 provides local flexibility information of the protein.

The KFoldTrunk module consists of the following sub-modules:
    - EnsembleModule (Modified MSA Module):
        Handle the pre-trained embeddings from multiple apo structures.
        input shape: (B, L, N_apo, c_struct)
    - MultiStateModule (Modified Template Embedder, Not implemented yet):
        Directly use multi-state apo coordinates.
        input shape: (B, Natom, Napo, 3)
"""

import dataclasses

import torch

from kfold.data.model_input import FoldingInput
from kfold.model.layers.alphafold3.pairformer import PairformerStack
from kfold.model.layers.kfold.ensemble_module import EnsembleModule
from kfold.model.layers.primitives import LayerNorm, LinearNoBias
from kfold.utils.registry import TRUNK

from .base import BaseTrunk


@dataclasses.dataclass(kw_only=True)
class PairformerConfig:
    num_heads_attn: int = 16
    num_heads_tri_attn: int = 4
    num_blocks: int = 48
    dropout: float = 0.25


@dataclasses.dataclass(kw_only=True)
class EnsembleModuleConfig:
    channel_struct_input: int = 0  # to be set according to input feature
    channel_struct: int = 128
    channel_hidden_opm: int = 32
    num_heads_pwa: int = 8
    num_heads_tri_attn: int = 4
    num_blocks: int = 4
    struct_dropout: float = 0.15
    z_dropout: float = 0.25


# TODO: Define the configuration for MultiStateModule when implemented
@dataclasses.dataclass(kw_only=True)
class MultiStateModuleConfig: ...


@TRUNK.register()
class MultiStateApoTrunk(BaseTrunk):
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
        use_ensemble: bool, optional
            Whether to use EnsembleModule, by default True
        use_multi_state: bool, optional
            Whether to use MultiStateModule, by default False
        use_cuequiv_kernels : bool, optional
            Whether to use cuequivariance kernels, by default False
        tri_attn_chunk_threshold : int, optional
            The threshold for chunking in triangle attention, by default 384
        """

        channel_s: int = 384
        channel_z: int = 128

        # pairformer
        pairformer: PairformerConfig = dataclasses.field(default_factory=PairformerConfig)

        # ensemble module options
        use_ensemble: bool = False
        ensemble_module: EnsembleModuleConfig = dataclasses.field(
            default_factory=EnsembleModuleConfig
        )

        use_multi_state: bool = False
        multi_state_module: MultiStateModuleConfig = dataclasses.field(
            default_factory=MultiStateModuleConfig
        )

        # other options
        use_cuequiv_kernels: bool = False
        blocks_per_ckpt: int | None = None
        tri_attn_chunk_threshold: int = 384

    def __init__(self, cfg: Config):
        """Initialize the Pairformer module."""
        super().__init__(cfg)
        self.use_ensemble: bool = cfg.use_ensemble
        self.use_multi_state: bool = cfg.use_multi_state

        if self.use_ensemble:
            self.ensemble_module: EnsembleModule = EnsembleModule(
                channel_s=cfg.channel_s,
                channel_z=cfg.channel_z,
                channel_struct_input=cfg.ensemble_module.channel_struct_input,
                channel_struct=cfg.ensemble_module.channel_struct,
                channel_hidden_opm=cfg.ensemble_module.channel_hidden_opm,
                num_heads_pwa=cfg.ensemble_module.num_heads_pwa,
                num_heads_tri_attn=cfg.ensemble_module.num_heads_tri_attn,
                num_blocks=cfg.ensemble_module.num_blocks,
                struct_dropout=cfg.ensemble_module.struct_dropout,
                z_dropout=cfg.ensemble_module.z_dropout,
                blocks_per_ckpt=cfg.blocks_per_ckpt,
            )

        if self.use_multi_state:
            raise NotImplementedError("Multi-State Module is not implemented yet")

        self.pairformer_module: PairformerStack = PairformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_heads_attn=cfg.pairformer.num_heads_attn,
            num_heads_tri_attn=cfg.pairformer.num_heads_tri_attn,
            num_blocks=cfg.pairformer.num_blocks,
            dropout=cfg.pairformer.dropout,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )

        # For recycling
        self.layernorm_s = LayerNorm(cfg.channel_s)
        self.layernorm_z = LayerNorm(cfg.channel_z)
        self.linear_s = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")

        # Other options
        self.use_cuequiv_kernels: bool = cfg.use_cuequiv_kernels
        self.chunk_threshold: int = cfg.tri_attn_chunk_threshold

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_recycles: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

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

        for i in range(1, num_recycles + 1):
            enable_grad = self.training and i == num_recycles

            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                z = z_init + self.linear_z(self.layernorm_z(z_hat))

                if self.use_multi_state:
                    raise NotImplementedError("Template Embedder is not implemented yet")

                if self.use_ensemble:
                    z = self.ensemble_module(
                        f_input,
                        z,
                        s_inputs,
                        use_cuequiv_mul=self.use_cuequiv_kernels,
                        use_cuequiv_attn=self.use_cuequiv_kernels,
                    )

                s = s_init + self.linear_s(self.layernorm_s(s_hat))

                # Revert to uncompiled version for validation
                s, z = self.pairformer_module(
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
