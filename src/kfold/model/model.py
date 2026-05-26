import dataclasses
import logging
import pathlib
import time
from collections.abc import Mapping
from functools import partial
from typing import Self

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules import (
    confidence_head,
    distogram_head,
    input_embedder,
    pairformer,
    plm_module,
    sequence_encoder,
    structure_encoder,
)
from kfold.model.modules.structure import sample_diffusion, score_model
from kfold.model.primitives import LayerNorm, LinearNoBias
from kfold.model.primitives.utils import add
from kfold.utils.registry import MAIN_MODULE, Registry

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class PLMModuleConfig:
    num_heads_attn: int = 16
    num_heads_tri_attn: int = 4
    num_blocks: int = 4
    dropout_plm: float = 0.15
    dropout_z: float = 0.25
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class PairformerConfig:
    num_heads_attn: int = 16
    num_heads_tri_attn: int = 4
    num_blocks: int = 48
    dropout: float = 0.25
    blocks_per_ckpt: int | None = None


@dataclasses.dataclass(kw_only=True)
class KFoldConfig:
    # Model dimensions
    channel_s: int = 384
    channel_z: int = 128
    channel_plm: int = 768

    # Sub-module configurations
    input_embedder: input_embedder.InputEmbedder.Config
    protein_sequence_encoder: sequence_encoder.SequenceEncoder.Config
    protein_structure_encoder: structure_encoder.StructureEncoder.Config
    rna_sequence_encoder: sequence_encoder.SequenceEncoder.Config
    plm_module: PLMModuleConfig
    pairformer: PairformerConfig
    score_model: score_model.DiffusionModule.Config
    diffusion_head: sample_diffusion.BaseStructureModule.Config
    distogram_head: distogram_head.DistogramHead.Config
    confidence_head: confidence_head.ConfidenceHead.Config

    # Kernel configurations
    kernel_cuequivariance: bool = True

    # For training
    diffusion_conditioning_drop_rate: float = 0.0
    confidence_conditioning_drop_rate: float = 0.0


@MAIN_MODULE.register()
class KFold(torch.nn.Module):
    def __init__(self, config: KFoldConfig):
        super().__init__()
        self.config: KFoldConfig = config
        self.channel_s: int = config.channel_s
        self.channel_z: int = config.channel_z
        self.channel_plm: int = config.channel_plm

        kernel_config = {
            "cuequivariance": config.kernel_cuequivariance,
        }
        self.kernel_config = kernel_config

        # Initialize input featurizer.
        self.input_embedder = Registry.instantiate(config.input_embedder)

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

        def proj_plm(channel_in: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                LayerNorm(channel_in, create_offset=False),
                LinearNoBias(channel_in, self.channel_plm, init="relu"),
                torch.nn.ReLU(),
                LinearNoBias(self.channel_plm, self.channel_plm, init="relu"),
            )

        self.proj_prot_seq = proj_plm(self.prot_seq_encoder.d_model)
        self.proj_rna_seq = proj_plm(self.rna_seq_encoder.d_model)
        self.proj_prot_struct = proj_plm(self.prot_struct_encoder.d_model)

        def proj_attn(channel_in: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                LayerNorm(channel_in, create_offset=False),
                LinearNoBias(channel_in, self.channel_z, init="relu"),
                torch.nn.ReLU(),
                LinearNoBias(self.channel_z, self.channel_z, init="final"),
            )

        self.proj_prot_seq_attn = proj_attn(self.prot_seq_encoder.n_attns)
        self.proj_rna_seq_attn = proj_attn(self.rna_seq_encoder.n_attns)

        # Skip projection
        self.skip_plm_to_s = torch.nn.Sequential(
            LinearNoBias(self.channel_plm, self.channel_s, init="relu"),
            torch.nn.ReLU(),
            LinearNoBias(self.channel_s, self.channel_s, init="final"),
        )

        # Initialize trunk
        self.plm_module = plm_module.PLMModule(
            channel_s_inputs=self.channel_s,
            channel_plm=self.channel_plm,
            channel_z=self.channel_z,
            num_heads_attn=config.plm_module.num_heads_attn,
            num_heads_tri_attn=config.plm_module.num_heads_tri_attn,
            num_blocks=config.plm_module.num_blocks,
            dropout_plm=config.plm_module.dropout_plm,
            dropout_z=config.plm_module.dropout_z,
            blocks_per_ckpt=config.plm_module.blocks_per_ckpt,
        )

        self.pairformer_stack = pairformer.PairformerStack(
            channel_s=self.channel_s,
            channel_z=self.channel_z,
            num_heads_attn=config.pairformer.num_heads_attn,
            num_heads_tri_attn=config.pairformer.num_heads_tri_attn,
            num_blocks=config.pairformer.num_blocks,
            dropout=config.pairformer.dropout,
            blocks_per_ckpt=config.pairformer.blocks_per_ckpt,
        )

        # Recyling
        self.layernorm_s = LayerNorm(self.channel_s)
        self.layernorm_z = LayerNorm(self.channel_z)
        self.linear_s = LinearNoBias(self.channel_s, self.channel_s, init="final")
        self.linear_z = LinearNoBias(self.channel_z, self.channel_z, init="final")

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
        self.confidence_head = confidence_head.ConfidenceHead(
            config.confidence_head, kernel_config=kernel_config
        )
        self.is_compiled = False

    def do_compile(self, mode: str = "default", dynamic: bool = False):
        """Compile the trunk and score model."""
        opts = {"mode": mode, "dynamic": dynamic}
        self.is_compiled = True
        self.prot_seq_encoder = torch.compile(self.prot_seq_encoder, **opts)
        self.rna_seq_encoder = torch.compile(self.rna_seq_encoder, **opts)
        self.prot_struct_encoder = torch.compile(self.prot_struct_encoder, **opts)
        self.plm_module = torch.compile(self.plm_module, **opts)
        self.pairformer_stack = torch.compile(self.pairformer_stack, **opts)
        self.score_model.do_compile(**opts)
        self.confidence_head.do_compile(**opts)

    # ============================================================
    # Inference Methods
    # ============================================================
    @torch.inference_mode()
    def inference(
        self,
        f_input: FoldingInput,
        apo_dict: dict[int, dict[str, torch.Tensor]],
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
        apo_dict : dict[int, dict[str, torch.Tensor]]
            Dictionary mapping entity_id to apo structure information.
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

        # Tokenize apo structure and feed into structure encoder input features
        for entity_id, apo_info in apo_dict.items():  # noqa
            for k in ["aatypes", "coords", "mapping"]:
                if k not in apo_info:
                    raise KeyError(
                        f"Apo info for entity_id {entity_id} is missing key: {k}"
                    )
            aatypes, coords = apo_info["aatypes"], apo_info["coords"]
            seq_st, seq_ed, apo_st, apo_ed = apo_info["mapping"]
            seq_sl, apo_sl = slice(seq_st, seq_ed), slice(apo_st, apo_ed)
            bb_ids, fa_ids = self.protein_structure_encoder.tokenize(aatypes, coords)
            f_input.sequence.bb_struct_token_id[0, seq_sl] = bb_ids[apo_sl]
            f_input.sequence.fa_struct_token_id[0, seq_sl] = fa_ids[apo_sl]

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
        s_inputs, s_init, z_init = self.input_embedder(f_input)
        et = time.time()
        time_logs["input_embedder"] = et - st

        # Trunk with recycling
        st = time.time()
        s_trunk, z_trunk = self.run_trunk(s_inputs, s_init, z_init, f_input, num_recycles)
        s_trunk, z_trunk = s_trunk.float(), z_trunk.float()
        et = time.time()
        time_logs["trunk"] = et - st

        if return_embeddings:
            dict_out["trunk"] = {
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "z_trunk": z_trunk,
            }

        # Distogram head
        st = time.time()
        dict_out["distogram"] = self.distogram_head.forward_inference(
            f_input,
            z_trunk,
        )
        et = time.time()
        time_logs["distogram_head"] = et - st

        # Diffusion head
        # pred_atom_coords: [B, Nsample, La, 3]
        st = time.time()
        with torch.autocast(f_input.device.type, enabled=False):
            dict_out["diffusion"] = self.diffusion_head.sample_structure(
                f_input,
                s_inputs,
                s_trunk,
                z_trunk,
                num_steps,
                num_samples,
                return_traj=return_traj,
            )
        et = time.time()
        time_logs["diffusion_head"] = et - st

        st = time.time()
        dict_out["confidence"] = self.confidence_head.forward_inference(
            f_input,
            s_inputs,
            s_trunk,
            z_trunk,
            dict_out["diffusion"]["coordinates"],
        )
        et = time.time()
        time_logs["confidence_head"] = et - st

        # If the input was not batched, remove the batch dimension
        if not return_batched_output:
            for key in dict_out:
                dict_out[key] = {k: v.squeeze(0) for k, v in dict_out[key].items()}
        return dict_out, time_logs

    def run_trunk(
        self,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        f_input: FoldingInput,
        num_recycles: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s_inputs : torch.Tensor
            Tensor of shape (B, L, C_s) containing input single features
        s_init: torch.Tensor
            Tensor of shape (B, L, C_s) containing initial single representation
        z_init: torch.Tensor
            Tensor of shape (B, L, L, C_z) containing initial pair representation
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
        use_cuequiv_kernels = self.kernel_config.get("cuequivariance", False)

        def get_model(mod: torch.nn.Module) -> torch.nn.Module:
            return mod._orig_mod if (self.is_compiled and not self.training) else mod

        prot_seq_encoder = get_model(self.prot_seq_encoder)
        rna_seq_encoder = get_model(self.rna_seq_encoder)
        prot_struct_encoder = get_model(self.prot_struct_encoder)
        plm_module = get_model(self.plm_module)
        pairformer_stack = get_model(self.pairformer_stack)

        # Run the structure encoder once, without stochastic masking.
        x_prot_struct = prot_struct_encoder(f_input)

        # === Main trunk iteration with recycling === #
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        for i in range(0, num_recycles + 1):
            enable_grad = self.training and i == num_recycles
            with torch.set_grad_enabled(enable_grad):
                if enable_grad and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                _add = partial(add, inplace=not enable_grad)

                # Recycling
                s = s_init + self.linear_s(self.layernorm_s(s))
                z = z_init + self.linear_z(self.layernorm_z(z))

                # Initialize s_plm
                s_plm = self.proj_prot_struct(x_prot_struct)

                # Add sequence features with stochastic masking
                x_prot, attn_prot = prot_seq_encoder(f_input, mask_ratio=0.15)
                s_plm = _add(s_plm, self.proj_prot_seq(x_prot))
                z = _add(z, self.proj_prot_seq_attn(attn_prot.flatten(-2)))
                del x_prot, attn_prot

                x_rna, attn_rna = rna_seq_encoder(f_input, mask_ratio=0.15)
                s_plm = _add(s_plm, self.proj_rna_seq(x_rna))
                z = _add(z, self.proj_rna_seq_attn(attn_rna.flatten(-2)))
                del x_rna, attn_rna

                # Run trunk
                z = plm_module(
                    z,
                    s_inputs,
                    s_plm,
                    f_input.token.asym_id,
                    f_input.token.pad_mask,
                    use_cuequiv_kernels=use_cuequiv_kernels,
                )
                s, z = pairformer_stack(
                    s,
                    z,
                    f_input.token.pad_mask,
                    use_cuequiv_kernels=use_cuequiv_kernels,
                )

                # Skip connection to s_trunk
                s = _add(s, self.skip_plm_to_s(s_plm))

        return s, z

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
        s_inputs, s_init, z_init = self.input_embedder(f_input)
        # NOTE: cast to float32 for numerical stability in training.
        s_init, z_init = s_init.float(), z_init.float()

        # Trunk with recycling
        s_trunk, z_trunk = self.run_trunk(s_inputs, s_init, z_init, f_input, num_recycles)
        s_trunk, z_trunk = s_trunk.float(), z_trunk.float()

        # Distogram head
        dict_out["distogram"] = {
            "logits": self.distogram_head(z_trunk),
        }

        if train_diffusion_head:
            # Diffusion head
            _s_trunk, _z_trunk = s_trunk, z_trunk
            drop_rate = self.config.diffusion_conditioning_drop_rate
            if drop_rate > 0.0:
                drop_conditioning = torch.rand(batch_size, device=device) < drop_rate
                mask = (~drop_conditioning).to(z_trunk.dtype)  # [B,]
                _s_trunk = s_trunk * mask[:, None, None]
                _z_trunk = z_trunk * mask[:, None, None, None]

            # Forward pass through diffusion head for training.
            with torch.autocast(device.type, enabled=False):
                dict_out["diffusion"] = self.diffusion_head.training_step(
                    f_input,
                    s_inputs,
                    _s_trunk,
                    _z_trunk,
                    diffusion_batch_size,
                )

        if train_confidence_module:
            # Stop gradients to input features and trunk outputs.
            # Sample structures with diffusion mini-rollout.
            with torch.no_grad(), torch.autocast(device.type, enabled=False):
                coordinates = self.diffusion_head.sample_structure(
                    f_input=f_input,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    num_steps=num_mini_rollout_steps,
                    num_samples=num_mini_rollout_samples,
                )["coordinates"]  # [B, N_samples, Latom, 3]
            dict_out["sample"] = {
                "coordinates": coordinates,
            }
            _s_inputs = s_inputs.detach()
            _s_trunk = s_trunk.detach()
            _z_trunk = z_trunk.detach()

            # Randomly drop conditioning information for confidence head.
            drop_rate = self.config.confidence_conditioning_drop_rate
            if drop_rate > 0.0:
                drop_conditioning = torch.rand(batch_size, device=device) < drop_rate
                mask = (~drop_conditioning).to(z_trunk.dtype)  # [B,]
                # NOTE: Only drop the pair conditioning.
                _z_trunk = z_trunk * mask[:, None, None, None]

            # Forward pass through confidence head
            pae_logits, pde_logits, plddt_logits, resolved_logits = self.confidence_head(
                f_input,
                _s_inputs,
                _s_trunk,
                _z_trunk,
                coordinates,
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
            "proj_prot_seq",
            "proj_rna_seq",
            "proj_prot_struct",
            "proj_prot_seq_attn",
            "proj_rna_seq_attn",
            "skip_plm_to_s",
            "plm_module",
            "pairformer_stack",
            "layernorm_s",
            "layernorm_z",
            "linear_s",
            "linear_z",
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
