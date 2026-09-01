import dataclasses
import logging
import math
import pathlib
import time
from collections.abc import Mapping
from typing import Self

import torch
import torch.nn.functional as F

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules import (
    apo_module,
    confidence_head,
    distogram_head,
    input_embedder,
    patch_geometry,
    sequence_encoder,
    structure_encoder,
    tri_stack,
)
from kfold.model.modules.structure import sample_diffusion, score_model
from kfold.model.primitives import LayerNorm, Linear, LinearNoBias
from kfold.utils.config import resolve_config
from kfold.utils.kernels import KernelPolicy
from kfold.utils.registry import Registry

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class ParcaeConfig:
    state_init: str = "trunc_normal"
    decay_init: float = math.sqrt(1.0 / 5.0)


@dataclasses.dataclass(kw_only=True)
class TrunkConfig:
    num_lm_blocks: int = 4
    num_main_blocks: int = 48
    num_refine_blocks: int = 2
    dropout: float = 0.25
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class KFoldConfig:
    # Model dimensions
    channel_s: int = 384
    channel_z: int = 256
    dropout: float = 0.25

    # Sub-module configurations
    input_embedder: input_embedder.InputEmbedder.Config
    apo_module: apo_module.ApoModule.Config
    protein_sequence_encoder: sequence_encoder.SequenceEncoder.Config
    protein_structure_encoder: structure_encoder.StructureEncoder.Config
    rna_sequence_encoder: sequence_encoder.SequenceEncoder.Config
    trunk: TrunkConfig
    parcae: ParcaeConfig
    score_model: score_model.DiffusionModule.Config
    diffusion_head: sample_diffusion.BaseStructureModule.Config
    distogram_head: distogram_head.DistogramHead.Config
    confidence_head: confidence_head.ConfidenceHead.Config
    patch_pair_geometry: patch_geometry.PatchPairGeometryHead.Config = dataclasses.field(
        default_factory=patch_geometry.PatchPairGeometryHead.Config
    )

    # Kernel configurations
    kernel_cuequivariance: bool = True
    kernel_backend: str | None = None

    # For training
    diffusion_conditioning_drop_rate: float = 0.0
    confidence_conditioning_drop_rate: float = 0.0


class LMEncoder(torch.nn.Module):
    def __init__(self, channel_lm: int, n_layers: int, channel_s: int):
        super().__init__()
        self.w_lm_layer = torch.nn.Parameter(torch.zeros(n_layers + 1))
        self.proj_lm = torch.nn.Sequential(
            LayerNorm(channel_lm, create_offset=False),
            LinearNoBias(channel_lm, channel_s),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Merge encoder block features and project to the shared LM single dim."""
        if hidden_states.ndim == 4:
            w = self.w_lm_layer.softmax(-1)  # [Nlayer+1]
            hidden_states = torch.einsum("n, b l n d -> b l d", w, hidden_states)
        elif hidden_states.ndim != 3:
            raise ValueError(
                "LMEncoder expects hidden states with shape [B, L, D] or "
                f"[B, L, Nlayer+1, D], got {hidden_states.shape}."
            )
        return self.proj_lm(hidden_states)


class LMToPair(torch.nn.Module):
    def __init__(self, channel_s: int, channel_z: int):
        super().__init__()
        self.proj = torch.nn.Sequential(
            LayerNorm(channel_s),
            Linear(channel_s, channel_z * 2),
        )
        self.mlp = torch.nn.Sequential(
            Linear(2 * channel_z, channel_z),
            torch.nn.GELU(),
            Linear(channel_z, channel_z),
        )
        self.layernorm_pair = LayerNorm(channel_z)

    def forward(self, s_lm: torch.Tensor) -> torch.Tensor:
        # Outer product to get pairwise features
        xi, xj = torch.chunk(self.proj(s_lm), 2, dim=-1)  # [B, L, D], [B, L, D]
        xi, xj = xi.unsqueeze(-2), xj.unsqueeze(-3)  # [B, L, 1, D], [B, 1, L, D]
        z = self.mlp(torch.cat([xi * xj, xi - xj], dim=-1))  # [B, L, L, D]
        z = self.layernorm_pair(z)
        return z


class KFold(torch.nn.Module):
    def __init__(self, config: KFoldConfig):
        super().__init__()
        self.config: KFoldConfig = config
        self.channel_s: int = config.channel_s
        self.channel_z: int = config.channel_z
        self.dropout: float = config.dropout
        self.trunk_config = resolve_config(TrunkConfig, config.trunk)
        self.parcae_config = resolve_config(ParcaeConfig, config.parcae)

        self.kernel_config = {
            "cuequivariance": config.kernel_cuequivariance,
        }
        self.kernel_policy = KernelPolicy.from_config(config)

        # Initialize pre-trained sequence and structure encoders.
        self.prot_seq_encoder = sequence_encoder.SequenceEncoder(
            config.protein_sequence_encoder
        )
        self.rna_seq_encoder = sequence_encoder.SequenceEncoder(
            config.rna_sequence_encoder
        )
        self.prot_struct_encoder = structure_encoder.StructureEncoder(
            config.protein_structure_encoder
        )

        # Initialize input featurizer.
        self.input_embedder = input_embedder.InputEmbedder(
            config.input_embedder, kernel_policy=self.kernel_policy
        )

        # Initialize LM single and pairwise feature projection modules.
        self.prot_seq_to_s_lm = LMEncoder(
            self.prot_seq_encoder.d_model, self.prot_seq_encoder.n_layers, self.channel_s
        )
        self.rna_seq_to_s_lm = LMEncoder(
            self.rna_seq_encoder.d_model, self.rna_seq_encoder.n_layers, self.channel_s
        )
        self.prot_struct_to_s_lm = torch.nn.Sequential(
            LayerNorm(self.prot_struct_encoder.d_model, create_offset=False),
            LinearNoBias(self.prot_struct_encoder.d_model, self.channel_s),
        )
        self.lm_to_pair = LMToPair(self.channel_s, self.channel_z)

        # Initialize trunk
        self.apo_module = apo_module.ApoModule(
            config.apo_module, kernel_policy=self.kernel_policy
        )
        self.layernorm_z = LayerNorm(self.channel_z)

        # Parcae theory: learn a continuous negative-diagonal state transition
        # and an Euler-discretized input injection for the pair recurrence.
        self.parcae_log_a = torch.nn.Parameter(torch.zeros(self.channel_z))
        # Parcae config: decay_init is the initial discrete contraction a when
        # log_a starts at zero, so delta_init = -log(decay_init).
        parcae_decay_init = self.parcae_config.decay_init
        parcae_delta_init = -math.log(parcae_decay_init)
        self.parcae_log_delta = torch.nn.Parameter(
            torch.full(
                (self.channel_z,),
                math.log(math.expm1(parcae_delta_init)),
                dtype=torch.float32,
            )
        )
        self.parcae_b_cont = torch.nn.Parameter(torch.eye(self.channel_z))

        self.lm_stack = tri_stack.TrianglularStack(
            self.channel_z,
            self.trunk_config.num_lm_blocks,
            self.trunk_config.dropout,
            kernel_policy=self.kernel_policy,
        )
        self.main_stack = tri_stack.TrianglularStack(
            self.channel_z,
            self.trunk_config.num_main_blocks,
            self.trunk_config.dropout,
            blocks_per_ckpt=self.trunk_config.blocks_per_ckpt,
            kernel_policy=self.kernel_policy,
        )
        # Recyling
        self.linear_refine = LinearNoBias(self.channel_z, self.channel_z, init="identity")
        self.refine_stack = tri_stack.TrianglularStack(
            self.channel_z,
            self.trunk_config.num_refine_blocks,
            self.trunk_config.dropout,
            kernel_policy=self.kernel_policy,
        )

        # Initialize prediction heads
        self.score_model = score_model.DiffusionModule(
            config.score_model, kernel_policy=self.kernel_policy
        )
        # NOTE: diffusion_head is not a torch.nn.Module
        # TODO: After we fix the diffusion algorith, remove Registry.instantiate
        self.diffusion_head = Registry.instantiate(
            config.diffusion_head, score_model=self.score_model
        )
        self.distogram_head = distogram_head.DistogramHead(config.distogram_head)
        self.patch_pair_geometry_head = patch_geometry.PatchPairGeometryHead(
            config.patch_pair_geometry,
            channel_z=self.channel_z,
        )
        self.confidence_head = confidence_head.ConfidenceHead(
            config.confidence_head, kernel_policy=self.kernel_policy
        )

    def _parcae_discretized_dynamics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the Parcae ZOH/Euler-discretized pair-state dynamics."""
        delta = F.softplus(self.parcae_log_delta)
        a = torch.exp(-delta * torch.exp(self.parcae_log_a))
        b = delta[:, None] * self.parcae_b_cont
        return a, b

    def _init_parcae_pair_state(self, ref: torch.Tensor) -> torch.Tensor:
        """Initialize z_0 as in ESMFold2's domain-adapted Parcae recurrence."""
        # Parcae config: "zero" preserves KFold's previous pair-state init.
        if self.parcae_config.state_init == "zero":
            return torch.zeros_like(ref)

        # ESMFold2 cofolding adaptation: randomized truncated-normal pair state.
        std = math.sqrt(2.0 / (5.0 * ref.shape[-1]))
        state = torch.empty_like(ref, dtype=torch.float32)
        torch.nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return state.to(dtype=ref.dtype)

    # ============================================================
    # Inference Methods
    # ============================================================
    @torch.inference_mode()
    def inference(
        self,
        f_input: FoldingInput,
        num_recycles: int = 10,
        num_steps: int = 100,
        num_samples: int = 5,
        return_embeddings: bool = False,
        return_traj: bool = False,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
        """Run KFold structure prediction from a fully prepared input.

        Parameters
        ----------
        f_input : FoldingInput
            Input data for folding model.
        num_recycles : int
            Number of recycling cycles in trunk.
        num_steps : int
            Number of diffusion steps for training.
        num_samples : int
            Number of diffusion samples for training.

        Returns
        -------
        model_out : dict[str, dict[str, torch.Tensor]]
            Output dictionary containing sampled structures and intermediate features:
            - trunk: intermediate trunk outputs. (optional)
            - distogram: predicted distogram logits.
            - diffusion: sampled structures from diffusion head.
            - confidence: predicted confidence metrics from confidence head.

        time_logs : dict[str, float]
            Dictionary containing time taken for each module during sampling.
        """
        # If input is not batched, add batch dimension for processing
        # and remove it from output at the end.
        if f_input.is_batched:
            return_batched_output = True
        else:
            f_input = f_input.add_batch_dim()
            return_batched_output = False

        if f_input.batch_size != 1:
            # TODO: Support batched inference.
            raise NotImplementedError(
                "Batched input with batch_size > 1 is not supported for inference yet."
            )

        # Sample structures
        model_out, time_logs = self.sample(
            f_input,
            num_recycles,
            num_steps,
            num_samples,
            return_embeddings=return_embeddings,
            return_traj=return_traj,
        )

        # remove batch dimension
        if not return_batched_output:
            model_out = {
                k: {kk: vv.squeeze(0) for kk, vv in v.items()}
                for k, v in model_out.items()
            }

        return model_out, time_logs

    @torch.inference_mode()
    def sample(
        self,
        f_input: FoldingInput,
        num_recycles: int = 10,
        num_steps: int = 100,
        num_samples: int = 5,
        return_embeddings: bool = False,
        return_traj: bool = False,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
        """Forward pass of KFold model for model training.

        Parameters
        ----------
        f_input : FoldingInput
            Input data for folding model.
        num_recycles : int
            Number of recycling cycles in trunk.
        num_steps : int
            Number of diffusion steps for training.
        num_samples : int
            Number of diffusion samples for training.
        return_embeddings : bool, optional
            Whether to return intermediate sequence and structure embeddings.
        return_traj : bool, optional
            Whether to return sampling trajectories.

        Returns
        -------
        model_out : dict[str, dict[str, torch.Tensor]]
            Output dictionary containing sampled structures and intermediate features:
            - trunk: intermediate trunk outputs. (optional)
            - distogram: predicted distogram logits.
            - diffusion: sampled structures from diffusion head.
            - confidence: predicted confidence metrics from confidence head.

        time_logs : dict[str, float]
            Dictionary containing time taken for each module during sampling.
        """
        dict_out: dict[str, dict[str, torch.Tensor]] = {}
        time_logs: dict[str, float] = {}

        if f_input.batch_size != 1:
            # TODO: Support batched inference.
            raise NotImplementedError(
                "Batched input with batch_size > 1 is not supported for inference yet."
            )

        # Trunk with recycling
        st = time.time()
        s_inputs, s_lm, z = self.run_trunk(f_input, num_recycles)
        z = z.float()
        et = time.time()
        time_logs["trunk"] = et - st

        if return_embeddings:
            dict_out["trunk"] = {
                "s_inputs": s_inputs,
                "s_lm": s_lm,
                "z": z,
            }

        # Distogram head
        st = time.time()
        dict_out["distogram"] = self.distogram_head.forward_inference(f_input, z)
        et = time.time()
        time_logs["distogram_head"] = et - st

        # Diffusion head
        # pred_atom_coords: [B, Nsample, La, 3]
        st = time.time()
        with torch.autocast(f_input.device.type, enabled=False):
            dict_out["diffusion"] = self.diffusion_head.sample_structure(
                f_input,
                s_inputs,
                z,
                num_steps,
                num_samples,
                chunk_size=5,
                return_traj=return_traj,
            )
        et = time.time()
        time_logs["diffusion_head"] = et - st

        st = time.time()
        coords = dict_out["diffusion"]["coordinates"]
        dict_out["confidence"] = self.confidence_head(f_input, s_inputs, s_lm, z, coords)
        et = time.time()
        time_logs["confidence_head"] = et - st

        return dict_out, time_logs

    def _encode_lm_single(self, f_input: FoldingInput) -> torch.Tensor:
        """Merge the enabled pretrained encoders into the shared LM single."""
        return (
            self.prot_seq_to_s_lm(self.prot_seq_encoder(f_input))
            + self.rna_seq_to_s_lm(self.rna_seq_encoder(f_input))
            + self.prot_struct_to_s_lm(self.prot_struct_encoder(f_input))
        )

    def run_trunk(
        self,
        f_input: FoldingInput,
        num_recycles: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        f_input : FoldingInput
            The input features.
        num_recycles : int
            The number of recycling steps.

        Returns
        -------
        s_inputs: torch.Tensor
            The input single representation of shape (B, L, c_s).
        s_lm: torch.Tensor
            The LM single representation of shape (B, L, c_s_lm).
        z: torch.Tensor
            The updated tensor of shape (B, L, L, c_z).
        """
        dtype = torch.get_autocast_dtype(f_input.device.type)

        # Parcae theory: stable channel-wise state decay (a) and
        # Euler-discretized normalized input injection (b).
        a, b = self._parcae_discretized_dynamics()  # [C_z], [C_z, C_z]
        a, b = a.to(dtype), b.to(dtype)

        # Input embedding
        s_inputs, z_inputs = self.input_embedder(f_input)

        # Embedding of the apo state into the pair representation.
        z_inputs = z_inputs + self.apo_module(f_input)

        # Initialize an independent pair-state z_0 instead of recycling from zeros.
        z = self._init_parcae_pair_state(z_inputs)
        token_mask = f_input.token.pad_mask
        pair_mask = token_mask[..., None] & token_mask[..., None, :]

        # Extract LM representation
        s_lm = self._encode_lm_single(f_input)
        z_lm = self.lm_to_pair(s_lm)

        # Main trunk iteration with Parcae recurrence
        for _ in range(0, num_recycles + 1):
            # Intentional dropout during inference.
            _z_lm = F.dropout(z_lm, p=self.dropout, training=True)
            u_t = z_inputs + self.lm_stack(_z_lm, pair_mask)
            # Parcae recurrence: z_in = a * z_t + B_bar LN(u_t)
            z = a * z + F.linear(self.layernorm_z(u_t), b)
            z = self.main_stack(z, pair_mask)

        # Refinement iteration
        z = self.refine_stack(self.linear_refine(z), pair_mask)

        return s_inputs, s_lm, z

    # ============================================================
    # Utility Methods
    # ============================================================
    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | pathlib.Path,
        ckpt_path: str | pathlib.Path,
        override_args: list[str] | None = None,
        use_ema: bool = True,
        strict: bool = True,
    ) -> Self:
        """Load model from checkpoint."""
        from omegaconf import OmegaConf

        from kfold.config import load_config

        # Load model config
        config = load_config(config_path)
        if "model" in config:
            # Get model config if wrapped in a higher-level config
            config = config.model

        if override_args is not None:
            # Override specific arguments in the config
            overrides = OmegaConf.from_dotlist(override_args)
            config = OmegaConf.merge(config, overrides)

        # Initialize model
        model = cls(config)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" not in ckpt:
            # Assume the checkpoint is a state_dict itself
            state_dict = ckpt
        elif use_ema:
            # Load EMA weights
            if "ema" not in ckpt:
                raise KeyError(
                    "EMA weights not found in checkpoint. "
                    "Please set use_ema=False to load regular weights."
                )
            else:
                state_dict = ckpt["ema"]["shadow_params"]
        else:
            # Load regular weights
            state_dict = ckpt["state_dict"]

        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=strict)
        del ckpt, state_dict

        return model

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load state dict without pretrained sequence encoder"""
        # If strict is False, it is fine to have missing keys (e.g., pretrained model)
        incompatible_keys = super().load_state_dict(
            state_dict, strict=False, assign=assign
        )
        if strict:
            missing_keys = incompatible_keys.missing_keys
            unexpected_keys = incompatible_keys.unexpected_keys
            # If the sequence encoder is pretrained and not included in the state dict,
            # missing keys starting with "sequence_encoder." or "structure_encoder." are
            # allowed.
            missing_keys = {
                k
                for k in missing_keys
                if not k.startswith(
                    (
                        "prot_seq_encoder.",
                        "rna_seq_encoder.",
                        "prot_struct_encoder.",
                    )
                )
            }
            if missing_keys:
                raise KeyError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")
        return incompatible_keys
