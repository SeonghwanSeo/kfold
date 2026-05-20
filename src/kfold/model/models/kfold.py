import dataclasses
import logging
import pathlib
import time
from collections.abc import Mapping
from typing import Self

import torch

import kfold.model.modules as submodules
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.kfold.plm_module import PLMInputEmbedder
from kfold.utils.registry import MAIN_MODULE, BaseConfig, Registry

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class KFoldConfig:
    input_embedder: BaseConfig
    sequence_encoder: BaseConfig
    structure_encoder: BaseConfig
    trunk: BaseConfig
    score_model: BaseConfig
    structure_module: BaseConfig
    distogram_head: BaseConfig
    confidence_head: BaseConfig

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
        kernel_config = {
            "cuequivariance": config.kernel_cuequivariance,
        }

        # Initialize encoders
        self.input_embedder: submodules.input_embedder.BaseInputEmbedder = (
            Registry.instantiate(config.input_embedder)
        )
        self.sequence_encoder: submodules.sequence_encoder.BaseSequenceEncoder = (
            Registry.instantiate(config.sequence_encoder)
        )
        self.structure_encoder: submodules.structure_encoder.BaseStructureEncoder = (
            Registry.instantiate(config.structure_encoder)
        )

        seq_enc, struct_enc = self.sequence_encoder, self.structure_encoder
        self.plm_input_embedder: PLMInputEmbedder = PLMInputEmbedder(
            channel_seq_emb=(seq_enc.n_layers, seq_enc.d_model),
            channel_seq_attn=(seq_enc.n_layers, seq_enc.n_heads),
            channel_struct_emb=struct_enc.d_model,
            channel_plm=config.trunk.channel_plm,  # type: ignore
        )

        # Initialize trunk
        self.trunk: submodules.trunk.BaseTrunk = Registry.instantiate(
            config.trunk, kernel_config=kernel_config
        )

        # Initialize prediction heads
        # NOTE: diffusion_head is not a torch.nn.Module
        self.score_model: submodules.score_model.BaseScoreModel = Registry.instantiate(
            config.score_model, kernel_config=kernel_config
        )
        self.diffusion_head: submodules.structure_module.BaseStructureModule = (
            Registry.instantiate(config.structure_module, score_model=self.score_model)
        )

        self.distogram_head: submodules.prediction_head.DistogramHead = (
            Registry.instantiate(config.distogram_head)
        )

        self.confidence_head: submodules.prediction_head.ConfidenceHead = (
            Registry.instantiate(config.confidence_head, kernel_config=kernel_config)
        )

    def do_compile(self, mode: str = "default", dynamic: bool = False):
        """Compile the trunk and score model."""
        kwargs = {"mode": mode, "dynamic": dynamic}
        self.trunk.do_compile(**kwargs)
        self.score_model.do_compile(**kwargs)
        self.confidence_head.do_compile(**kwargs)

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
            bb_ids, fa_ids = self.structure_encoder.tokenize(aatypes, coords)
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

        # Sequence encoder
        st = time.time()
        seq_emb, seq_attn = self.sequence_encoder(f_input)
        et = time.time()
        time_logs["sequence_encoder"] = et - st

        # Structure encoder
        st = time.time()
        struct_emb = self.structure_encoder(f_input)
        et = time.time()
        time_logs["structure_encoder"] = et - st

        st = time.time()
        plm_inputs, plm_attn = self.plm_input_embedder(seq_emb, seq_attn, struct_emb)
        z_init += plm_attn  # add PLM attention bias to pair representation
        del seq_emb, seq_attn, struct_emb, plm_attn  # free up memory
        et = time.time()
        time_logs["plm_input_embedder"] = et - st

        # Trunk with recycling
        st = time.time()
        s_trunk, z_trunk = self.trunk(
            s_inputs,
            s_init,
            z_init,
            f_input,
            num_recycles,
            plm_inputs=plm_inputs,
        )
        et = time.time()
        time_logs["trunk"] = et - st

        if return_embeddings:
            dict_out["trunk"] = {
                "s_inputs": s_inputs,
                "plm_inputs": plm_inputs,
                "s_trunk": s_trunk,
                "z_trunk": z_trunk,
            }
        del plm_inputs  # free up memory

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
        train_structure_module: bool = True,
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

        train_structure_module : bool, optional
            Whether to train structure module, by default True
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
        s_inputs, s_init, z_init = s_inputs.float(), s_init.float(), z_init.float()

        # Get PLM input embeddings
        seq_emb, seq_attn = self.sequence_encoder(f_input)
        struct_emb = self.structure_encoder(f_input)
        plm_inputs, plm_attn = self.plm_input_embedder(seq_emb, seq_attn, struct_emb)
        z_init = z_init + plm_attn  # add PLM attention bias to pair representation
        del seq_emb, seq_attn, struct_emb, plm_attn  # free up memory

        # Trunk with recycling
        s_trunk, z_trunk = self.trunk(
            s_inputs,
            s_init,
            z_init,
            f_input,
            num_recycles,
            plm_inputs=plm_inputs,
        )

        if train_structure_module:
            # Distogram head
            dict_out["distogram"] = {
                "logits": self.distogram_head(z_trunk),
            }

            # Diffusion head
            _s_trunk, _z_trunk = s_trunk, z_trunk
            drop_rate = self.config.diffusion_conditioning_drop_rate
            if drop_rate > 0.0:
                drop_conditioning = torch.rand(batch_size, device=device) < drop_rate
                mask = (~drop_conditioning).to(z_trunk.dtype)  # [B,]
                _s_trunk = s_trunk * mask[:, None, None]
                _z_trunk = z_trunk * mask[:, None, None, None]

            # Forward pass through diffusion head for training.
            dict_out["diffusion"] = self.diffusion_head.training_step(
                f_input,
                s_inputs,
                _s_trunk,
                _z_trunk,
                diffusion_batch_size,
            )

        if train_confidence_module:
            # Sample structures with diffusion mini-rollout.
            with torch.no_grad():
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

            # Stop gradients to input features and trunk outputs.
            _s_inputs = s_inputs.detach()
            _s_trunk = s_trunk.detach()
            _z_trunk = z_trunk.detach()

            # Randomly drop conditioning information for confidence head.
            drop_rate = self.config.confidence_conditioning_drop_rate
            if drop_rate > 0.0:
                drop_conditioning = torch.rand(batch_size, device=device) < drop_rate
                mask = (~drop_conditioning).to(z_trunk.dtype)  # [B,]
                _z_trunk = _z_trunk * mask[:, None, None, None]

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
                if not k.startswith(("sequence_encoder.", "structure_encoder."))
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
