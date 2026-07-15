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
from kfold.utils.registry import MAIN_MODULE, Registry

logger = logging.getLogger(__name__)


def _inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x))


@dataclasses.dataclass(kw_only=True)
class ParcaeConfig:
    state_init: str = "trunc_normal"
    decay_init: float = math.sqrt(1.0 / 5.0)
    coda_n_layers: int | None = None


@dataclasses.dataclass(kw_only=True)
class TrunkConfig:
    num_lm_blocks: int = 4
    num_main_blocks: int = 48
    num_refine_blocks: int = 2
    dropout: float = 0.25
    blocks_per_ckpt: int | None = None
    parcae: ParcaeConfig = dataclasses.field(default_factory=ParcaeConfig)


_MISSING = object()


def _get_config_value(config: object, key: str, default: object = _MISSING) -> object:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_parcae_config(config: object | None) -> ParcaeConfig:
    if config is None:
        parcae_config = ParcaeConfig()
    elif isinstance(config, ParcaeConfig):
        parcae_config = config
    else:
        values = {}
        for field in dataclasses.fields(ParcaeConfig):
            value = _get_config_value(config, field.name, _MISSING)
            if value is not _MISSING:
                values[field.name] = value
        parcae_config = ParcaeConfig(**values)

    if parcae_config.state_init not in {"trunc_normal", "zero"}:
        raise ValueError(
            "ParcaeConfig.state_init must be either 'trunc_normal' or 'zero', "
            f"got {parcae_config.state_init!r}."
        )
    if not 0.0 < parcae_config.decay_init < 1.0:
        raise ValueError(
            "ParcaeConfig.decay_init must be in (0, 1) so it maps to a "
            f"positive step size, got {parcae_config.decay_init}."
        )
    if parcae_config.coda_n_layers is not None and parcae_config.coda_n_layers < 0:
        raise ValueError(
            "ParcaeConfig.coda_n_layers must be non-negative or None, "
            f"got {parcae_config.coda_n_layers}."
        )

    return parcae_config


@dataclasses.dataclass(kw_only=True)
class KFoldConfig:
    # Model dimensions
    channel_s: int = 384
    channel_z: int = 256
    dropout: float = 0.25

    # Sub-module configurations
    input_embedder: input_embedder.InputEmbedder.Config
    protein_sequence_encoder: sequence_encoder.SequenceEncoder.Config
    protein_structure_encoder: structure_encoder.StructureEncoder.Config
    rna_sequence_encoder: sequence_encoder.SequenceEncoder.Config
    trunk: TrunkConfig
    score_model: score_model.DiffusionModule.Config
    diffusion_head: sample_diffusion.BaseStructureModule.Config
    distogram_head: distogram_head.DistogramHead.Config
    confidence_head: confidence_head.ConfidenceHead.Config
    patch_pair_geometry: patch_geometry.PatchPairGeometryHead.Config = dataclasses.field(
        default_factory=patch_geometry.PatchPairGeometryHead.Config
    )

    # Kernel configurations
    kernel_cuequivariance: bool = True

    # For training
    diffusion_conditioning_drop_rate: float = 0.0
    confidence_conditioning_drop_rate: float = 0.0


class LMEncoder(torch.nn.Module):
    def __init__(self, channel_lm: int, n_layers: int, channel_s: int):
        super().__init__()
        self.channel_lm: int = channel_lm
        self.n_layers: int = n_layers
        self.channel_s: int = channel_s

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
        self.channel_s: int = channel_s
        self.channel_z: int = channel_z

        self.proj = torch.nn.Sequential(
            LayerNorm(channel_s), Linear(channel_s, channel_z * 2)
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


@MAIN_MODULE.register()
class KFold(torch.nn.Module):
    def __init__(self, config: KFoldConfig):
        super().__init__()
        self.config: KFoldConfig = config
        self.channel_s: int = config.channel_s
        self.channel_z: int = config.channel_z
        self.dropout: float = config.dropout
        self.parcae_config: ParcaeConfig = _normalize_parcae_config(
            _get_config_value(config.trunk, "parcae", None)
        )

        kernel_config = {
            "cuequivariance": config.kernel_cuequivariance,
        }
        self.kernel_config = kernel_config

        # Initialize input featurizer.
        self.input_embedder = input_embedder.InputEmbedder(config.input_embedder)

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
                _inverse_softplus(parcae_delta_init),
                dtype=torch.float32,
            )
        )
        self.parcae_b_cont = torch.nn.Parameter(torch.eye(self.channel_z))

        self.lm_stack = tri_stack.TrianglularStack(
            self.channel_z,
            config.trunk.num_lm_blocks,
            config.trunk.dropout,
        )
        self.main_stack = tri_stack.TrianglularStack(
            self.channel_z,
            config.trunk.num_main_blocks,
            config.trunk.dropout,
            blocks_per_ckpt=config.trunk.blocks_per_ckpt,
        )
        # Recyling
        self.linear_refine = LinearNoBias(self.channel_z, self.channel_z, init="identity")
        # ESMFold2 coda adaptation: coda_n_layers controls the refinement stack;
        # None preserves the previous KFold num_refine_blocks config path.
        coda_n_layers = (
            config.trunk.num_refine_blocks
            if self.parcae_config.coda_n_layers is None
            else self.parcae_config.coda_n_layers
        )
        self.refine_stack = tri_stack.TrianglularStack(
            self.channel_z,
            coda_n_layers,
            config.trunk.dropout,
        )

        # Initialize prediction heads
        self.score_model = score_model.DiffusionModule(
            config.score_model, kernel_config=kernel_config
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
            config.confidence_head, kernel_config=kernel_config
        )
        self.is_compiled = False

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
        if self.parcae_config.state_init != "trunc_normal":
            raise ValueError(
                "ParcaeConfig.state_init must be either 'trunc_normal' or 'zero', "
                f"got {self.parcae_config.state_init!r}."
            )
        std = math.sqrt(2.0 / (5.0 * ref.shape[-1]))
        state = torch.empty_like(ref, dtype=torch.float32)
        torch.nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return state.to(dtype=ref.dtype)

    def do_compile(self, mode: str = "default", dynamic: bool = False):
        """Compile the trunk and score model."""
        opts = {"mode": mode, "dynamic": dynamic}
        self.is_compiled = True
        self.prot_seq_encoder = torch.compile(self.prot_seq_encoder, **opts)
        self.rna_seq_encoder = torch.compile(self.rna_seq_encoder, **opts)
        self.prot_struct_encoder = torch.compile(self.prot_struct_encoder, **opts)

        self.lm_stack = torch.compile(self.lm_stack, **opts)
        self.main_stack = torch.compile(self.main_stack, **opts)
        self.refine_stack = torch.compile(self.refine_stack, **opts)
        self.score_model.do_compile(**opts)
        self.confidence_head.do_compile(**opts)

    def _get_model_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Return the underlying module when a compiled wrapper is not used."""
        if self.is_compiled and not self.training:
            return getattr(module, "_orig_mod", module)
        return module

    # ============================================================
    # Inference Methods
    # ============================================================
    @torch.inference_mode()
    def inference(
        self,
        f_input: FoldingInput,
        num_recycles: int = 10,
        num_steps: int = 200,
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
        num_steps: int = 200,
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

        # Embed inputs
        st = time.time()
        s_inputs, z_init = self.input_embedder(f_input)
        et = time.time()
        time_logs["input_embedder"] = et - st

        # Trunk with recycling
        st = time.time()
        z, s_lm = self.run_trunk(z_init, f_input, num_recycles)
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
        dict_out["confidence"] = self.confidence_head.forward_inference(
            f_input, s_inputs, s_lm, z, coords
        )
        et = time.time()
        time_logs["confidence_head"] = et - st

        # If the input was not batched, remove the batch dimension
        if not return_batched_output:
            for key in dict_out:
                dict_out[key] = {k: v.squeeze(0) for k, v in dict_out[key].items()}
        return dict_out, time_logs

    def _encode_lm_single(self, f_input: FoldingInput) -> torch.Tensor:
        """Merge the enabled pretrained encoders into the shared LM single."""
        prot_seq_encoder = self._get_model_module(self.prot_seq_encoder)
        rna_seq_encoder = self._get_model_module(self.rna_seq_encoder)
        prot_struct_encoder = self._get_model_module(self.prot_struct_encoder)

        return (
            self.prot_seq_to_s_lm(prot_seq_encoder(f_input))
            + self.rna_seq_to_s_lm(rna_seq_encoder(f_input))
            + self.prot_struct_to_s_lm(prot_struct_encoder(f_input))
        )

    def run_trunk(
        self,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_recycles: int,
        grad_recurrence_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        z_init: torch.Tensor
            Tensor of shape (B, L, L, C_z) containing initial pair representation
        f_input : FoldingInput
            The input features.
        num_recycles : int
            The number of recycling steps.
        grad_recurrence_steps : int, optional
            Number of final recurrent trunk steps to track with autograd during
            training, by default 1.

        Returns
        -------
        z: torch.Tensor
            The updated tensor of shape (B, L, L, c_z).
        s_lm: torch.Tensor
            The merged LM single representation of shape (B, L, c_s_lm).
        """
        use_cuequiv_kernels = self.kernel_config.get("cuequivariance", False)

        lm_stack = self._get_model_module(self.lm_stack)
        main_stack = self._get_model_module(self.main_stack)
        refine_stack = self._get_model_module(self.refine_stack)

        # Merge pretrained encoder block features into one shared LM single.
        s_lm = self._encode_lm_single(f_input)
        z_lm = self.lm_to_pair(s_lm)

        # ESMFold2 cofolding adaptation: initialize an independent pair-state
        # z_0 instead of recycling from zeros.
        z = self._init_parcae_pair_state(z_init)
        token_mask = f_input.token.pad_mask
        pair_mask = token_mask[..., None] & token_mask[..., None, :]

        # Parcae theory: stable channel-wise state decay (a) and
        # Euler-discretized normalized input injection (b).
        a, b = self._parcae_discretized_dynamics()
        a = a.view(*((1,) * (z_init.ndim - 1)), -1).to(
            device=z_init.device, dtype=z_init.dtype
        )
        b = b.to(device=z_init.device, dtype=z_init.dtype)

        # === Main trunk iteration with ESMFold2-style Parcae recurrence === #
        # Training-time stochastic recycle-count sampling is handled by the
        # trainer so this loop preserves num_recycles + 1 public semantics.
        grad_recurrence_steps = max(1, int(grad_recurrence_steps))
        grad_start = max(0, num_recycles + 1 - grad_recurrence_steps)

        # Intentionally not implemented: per-sequence depth sampling.
        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i >= grad_start
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()
                _z_lm = F.dropout(z_lm, p=self.dropout)
                # ESMFold2 cofolding adaptation: u_t combines input pair
                # features with the refined LM pair contribution each loop.
                u_t = z_init + lm_stack(_z_lm, pair_mask, use_cuequiv_kernels)
                # Parcae recurrence: z_in = a * z_t + B_bar LN(u_t), followed
                # by the pair folding trunk as the nonlinear recurrent update.
                z = a * z + F.linear(self.layernorm_z(u_t), b)
                z = main_stack(z, pair_mask, use_cuequiv_kernels)

        # Refinement iteration
        z = refine_stack(self.linear_refine(z), pair_mask, use_cuequiv_kernels)

        return z, s_lm

    # ============================================================
    # Training Methods
    # ============================================================
    def forward_train(
        self,
        f_input: FoldingInput,
        num_recycles: int = 3,
        diffusion_batch_size: int = 48,
        num_mini_rollout_steps: int = 20,
        num_mini_rollout_samples: int = 1,
        train_diffusion_head: bool = True,
        train_confidence_module: bool = True,
        grad_recurrence_steps: int = 1,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass of KFold for model training.
        See Figure 2c in the main article of AlphaFold3.

        Parameters
        ----------
        f_input : FoldingInput
            Input data for folding model. Preferred to be batched.

        # For trunk with recycling:
        num_recycles : int
            Number of recycling cycles in trunk.
        grad_recurrence_steps : int, optional
            Number of final recurrent trunk steps to track with autograd during
            training.

        # For structure module training:
        diffusion_batch_size : int
            Batch size for diffusion training step.

        # For confidence module training with diffusion mini-rollout:
        num_mini_rollout_steps : int
            Number of diffusion steps to sample structures:
            Used for validation and confidence module training.
        num_mini_rollout_samples : int
            Number of diffusion samples to sample structures for
            confidence module training.

        train_diffusion_head : bool, optional
            Whether to train diffusion head, by default True
        train_confidence_module : bool, optional
            Whether to train confidence module, by default True

        Returns
        -------
        model_out : dict[str, torch.Tensor]

            # For structure module training (distogram, diffusion)
            - distogram:
                - logits: [B, Ltoken, Ltoken, Dd]
                    Distogram logits
            - diffusion:
                - loss_weights: [B, N_noise]
                    Weights for diffusion noise scale
                - prior_atom_coords: [B, N_noise, Latom, 3]
                    Prior atom coordinates
                - noised_atom_coords: [B, N_noise, Latom, 3]
                    Noised atom coordinates
                - denoised_atom_coords: [B, N_noise, Latom, 3]
                    Denoised atom coordinates
                - true_atom_coords: [B, N_noise, Latom, 3]
                    Ground truth atom coordinates

            # For confidence module training
            - sample:
                - coordinates: [B, N_samples, Ltoken, 3]
                    Sampled atom coordinates
            - confidence:
                - pae_logits: [B, Ltoken, Ltoken, Dp]
                    Predicted aligned error logits
                - pde_logits: [B, Ltoken, Dp]
                    Predicted distance error logits
                - plddt_logits: [B, Latom, Dp]
                    Predicted lDDT logits
                - experimental_resolved_logits: [B, Latom, 2]
                    Predicted experimental resolved logits
        """
        # Ensure batched input
        assert f_input.is_batched, "Input must be batched for training.."
        batch_size: int = f_input.batch_size
        device: torch.device = f_input.device

        # Output dictionary
        dict_out: dict[str, dict[str, torch.Tensor]] = {}

        # Input embedding
        s_inputs, z_init = self.input_embedder(f_input)

        # Trunk with recycling
        z_init = z_init.float()  # cast to float32 for numerical stability
        z, s_lm = self.run_trunk(
            z_init,
            f_input,
            num_recycles,
            grad_recurrence_steps=grad_recurrence_steps,
        )
        z = z.float()

        # Distogram head
        dict_out["distogram"] = {
            "logits": self.distogram_head(z),
        }
        patch_geometry_out = self.patch_pair_geometry_head(f_input, z)
        if patch_geometry_out:
            dict_out["patch_geometry"] = patch_geometry_out

        if train_diffusion_head:
            # Diffusion head
            _z = z
            drop_rate = self.config.diffusion_conditioning_drop_rate
            if drop_rate > 0.0:
                drop_conditioning = torch.rand(batch_size, device=device) < drop_rate
                mask = ~drop_conditioning
                _z = z * mask[:, None, None, None]

            # Forward pass through diffusion head for training.
            with torch.autocast(device.type, enabled=False):
                dict_out["diffusion"] = self.diffusion_head.training_step(
                    f_input, s_inputs, _z, diffusion_batch_size
                )

        if train_confidence_module:
            # Stop gradients to input features and trunk outputs.
            # Sample structures with diffusion mini-rollout.
            with torch.no_grad(), torch.autocast(device.type, enabled=False):
                coordinates = self.diffusion_head.sample_structure(
                    f_input=f_input,
                    s_inputs=s_inputs,
                    z=z,
                    num_steps=num_mini_rollout_steps,
                    num_samples=num_mini_rollout_samples,
                )["coordinates"]  # [B, N_samples, Latom, 3]
            dict_out["sample"] = {
                "coordinates": coordinates,
            }
            _s_inputs = s_inputs.detach()
            _s_lm = s_lm.detach()
            _z = z.detach()

            # Randomly drop conditioning information for confidence head.
            drop_rate = self.config.confidence_conditioning_drop_rate
            if drop_rate > 0.0:
                drop_conditioning = torch.rand(batch_size, device=device) < drop_rate
                mask = ~drop_conditioning
                _z = _z * mask[:, None, None, None]

            # Forward pass through confidence head
            pae_logits, pde_logits, plddt_logits, resolved_logits = self.confidence_head(
                f_input, _s_inputs, _s_lm, _z, coordinates
            )
            dict_out["confidence"] = {
                "pae_logits": pae_logits,
                "pde_logits": pde_logits,
                "plddt_logits": plddt_logits,
                "resolved_logits": resolved_logits,
            }

        return dict_out

    def get_pretrained_module_names(self) -> list[str]:
        """Get the names of pretrained modules."""
        return [
            "prot_seq_encoder",
            "rna_seq_encoder",
            "prot_struct_encoder",
        ]

    def get_trunk_module_names(self) -> list[str]:
        """Get the names of trunk modules."""
        return [
            "input_embedder",
            "prot_seq_to_s_lm",
            "rna_seq_to_s_lm",
            "prot_struct_to_s_lm",
            "lm_to_pair",
            "layernorm_z",
            "lm_stack",
            "main_stack",
            "linear_refine",
            "refine_stack",
            "patch_pair_geometry_head",
        ]

    def get_trunk_parameter_names(self) -> list[str]:
        """Get standalone trunk parameter names (Parcae)."""
        return [
            "parcae_log_a",
            "parcae_log_delta",
            "parcae_b_cont",
        ]

    def get_distogram_head_module_names(self) -> list[str]:
        """Get the names of distogram head modules."""
        return ["distogram_head"]

    def get_diffusion_head_module_names(self) -> list[str]:
        """Get the names of diffusion head modules."""
        return ["score_model"]

    def get_confidence_head_module_names(self) -> list[str]:
        """Get the names of confidence head modules."""
        return ["confidence_head"]

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
        # Add '._orig_mod.' to state dict keys if required for compiled models
        state_dict = self._add_orig_mod_to_state_dict(state_dict)

        # If strict is False, it is fine to have missing keys (e.g., pretrained model)
        incompatible_keys = super().load_state_dict(state_dict, strict=False)
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

    def _add_orig_mod_to_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Add '._orig_mod.' to state dict keys if required"""
        model_keys = set(self.state_dict().keys())
        state_keys = set(state_dict.keys())

        # Keys expected by the compiled model but missing in the checkpoint
        remaining_keys = model_keys - state_keys
        if len(remaining_keys) == 0:
            return dict(state_dict)  # No modification needed

        new_state_dict = dict(state_dict)
        for rk in remaining_keys:
            k = rk.replace("._orig_mod.", ".")
            if k in state_dict:
                new_state_dict[rk] = new_state_dict.pop(k)
        return new_state_dict
