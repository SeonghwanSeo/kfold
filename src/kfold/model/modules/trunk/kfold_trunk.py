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
    - EnsembleModule (Modified MSA Module):
        Handles pre-trained embeddings from multiple apo structures.
        Input shape: (B, L, N_apo, c_struct)
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

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.kfold.ensemble_module import EnsembleModule
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
    split_intra_inter_channels: bool = True
    skip_tri_attn: bool = False


@dataclasses.dataclass(kw_only=True)
class EnsembleModuleConfig:
    channel_struct_input: int = 0  # to be set according to input feature
    channel_struct: int = 128
    channel_hidden_opm: int = 32
    num_heads_pwa: int = 8
    num_heads_tri_attn: int = 4
    num_blocks: int = 4
    dropout_struct: float = 0.15
    dropout_z: float = 0.25


# TODO: Define the configuration for MultiStateModule when implemented
@dataclasses.dataclass(kw_only=True)
class MultiStateModuleConfig: ...


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
        num_heads_attn : int, optional
            The number of attention heads, by default 16
        num_heads_tri_attn : int, optional
            The number of triangle attention heads, by default 4
        dropout : float, optional
            The dropout rate, by default 0.25
        use_ensemble: bool, optional
            Whether to use EnsembleModule, by default True
        use_multi_state: bool, optional
            Whether to use MultiStateModule, by default False
        tri_attn_chunk_threshold : int, optional
            The threshold for chunking in triangle attention, by default 384
        """

        channel_s: int = 384
        channel_z: int = 128

        # pairformer
        interformer: InterformerConfig = dataclasses.field(
            default_factory=InterformerConfig
        )

        # ensemble module (optional)
        use_ensemble: bool = False
        ensemble_module: EnsembleModuleConfig = dataclasses.field(
            default_factory=EnsembleModuleConfig
        )

        # multi-state module (optional)
        use_multi_state: bool = False
        multi_state_module: MultiStateModuleConfig = dataclasses.field(
            default_factory=MultiStateModuleConfig
        )

        # other options
        blocks_per_ckpt: int | None = None
        tri_attn_chunk_threshold: int = 384

    def __init__(self, cfg: Config, kernel_config=None):
        """Initialize the MultiStateApoTrunk module."""
        super().__init__(cfg, kernel_config)
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
                dropout_struct=cfg.ensemble_module.dropout_struct,
                dropout_z=cfg.ensemble_module.dropout_z,
                blocks_per_ckpt=cfg.blocks_per_ckpt,
            )

        if self.use_multi_state:
            raise NotImplementedError("Multi-State Module is not implemented yet")

        self.pairformer_module: InterformerStack = InterformerStack(
            channel_s=cfg.channel_s,
            channel_z=cfg.channel_z,
            num_heads_attn=cfg.interformer.num_heads_attn,
            num_heads_tri_attn=cfg.interformer.num_heads_tri_attn,
            num_blocks=cfg.interformer.num_blocks,
            dropout=cfg.interformer.dropout,
            skip_tri_attn=cfg.interformer.skip_tri_attn,
            split_intra_inter_channels=cfg.interformer.split_intra_inter_channels,
            blocks_per_ckpt=cfg.blocks_per_ckpt,
        )

        # For recycling
        self.layernorm_s = LayerNorm(cfg.channel_s)
        self.layernorm_z = LayerNorm(cfg.channel_z)
        self.linear_s = LinearNoBias(cfg.channel_s, cfg.channel_s, init="final")
        self.linear_z = LinearNoBias(cfg.channel_z, cfg.channel_z, init="final")

        # Other options
        self.chunk_threshold: int = cfg.tri_attn_chunk_threshold

    def do_compile(self, mode: str = "default"):
        """Compile the trunk module."""
        # NOTE: you should compile the submodules inside the trunk
        # since the computation graph is changed depending on the
        # number of recycling steps. Thus, compile the sub module
        # instead of the whole trunk module.
        if self.use_ensemble:
            self.ensemble_module = torch.compile(
                self.ensemble_module,
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

        # Revert to uncompiled version for validation
        pairformer_module: InterformerStack
        if self.is_compiled and not self.training:
            pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
        else:
            pairformer_module = self.pairformer_module

        # z_hat, s_hat = 0, 0
        s_hat = torch.zeros_like(s_init)
        z_hat = torch.zeros_like(z_init)

        intra_mask = (
            f_input.token.asym_id[..., :, None] == f_input.token.asym_id[..., None, :]
        )  # [..., L, L]

        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i == num_recycles

            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                if self.is_compiled and enable_grad and i > 0:
                    # Clone the tensors for compilation.
                    s_hat = s_hat.clone()
                    z_hat = z_hat.clone()

                s = s_init + self.linear_s(self.layernorm_s(s_hat))
                z = z_init + self.linear_z(self.layernorm_z(z_hat))

                if self.use_multi_state:
                    raise NotImplementedError("MultiStateEmbedder is not implemented yet")

                if self.use_ensemble:
                    z = self.ensemble_module(
                        f_input,
                        z,
                        s_inputs,
                        chunk_size_tri_attn=chunk_size_tri_attn,
                        use_cuequiv_kernels=self.kernel_config.cuequivariance,
                    )

                s, z = pairformer_module(
                    s,
                    z,
                    mask=f_input.token.pad_mask,
                    intra_mask=intra_mask,
                    chunk_size_tri_attn=chunk_size_tri_attn,
                    use_cuequiv_kernels=self.kernel_config.cuequivariance,
                )

                s_hat, z_hat = s, z

        s_trunk, z_trunk = s_hat, z_hat
        return s_trunk, z_trunk
