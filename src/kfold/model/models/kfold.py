import time
from collections.abc import Mapping

import torch

import kfold.model.modules as submodules
from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import MAIN_MODULE, BaseConfig, Registry

from .base import BaseFoldingModel, BaseFoldingModelConfig


class KFoldConfig(BaseFoldingModelConfig):
    _class_: str = "KFold"
    sequence_encoder: BaseConfig
    structure_encoder: BaseConfig


@MAIN_MODULE.register()
class KFold(BaseFoldingModel):
    def __init__(self, config: KFoldConfig):
        super().__init__(config)
        self.sequence_encoder: submodules.sequence_encoder.BaseSequenceEncoder = (
            Registry.instantiate(config.sequence_encoder)
        )
        self.structure_encoder: submodules.structure_encoder.BaseStructureEncoder = (
            Registry.instantiate(config.structure_encoder)
        )

    def cast_to_bf16(self):
        """Cast model parameters to bfloat16 for faster inference."""
        super().cast_to_bf16()
        self.sequence_encoder = self.sequence_encoder.to(dtype=torch.bfloat16)
        self.structure_encoder.backbone = self.structure_encoder.backbone.to(
            dtype=torch.bfloat16
        )
        return self

    def forward(
        self,
        f_input: FoldingInput,
        num_recycles: int = 3,
        num_steps: int = 20,
        num_diffusion_samples: int = 1,
        diffusion_batch_size: int = 48,
        sample_structures: bool = True,
        train_structure_module: bool = True,
        train_confidence_module: bool = True,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass of KFold for model training.
        See Figure 2c in the main article of AlphaFold3.

        Parameters
        ----------
        f_input : FoldingInput
            Input data for folding model. Preferred to be batched.
        num_recycles : int
            Number of recycling cycles in trunk.

        # For diffusion sampling:
        num_steps : int
            Number of diffusion steps to sample structures:
            Used for validation and confidence module training.
        num_diffusion_samples : int
            Number of diffusion samples to sample structures for
            confidence module training.

        # For structure module training:
        diffusion_batch_size : int
            Batch size for diffusion training step.

        sample_structures : bool, optional
            Whether to sample structures for confidence module training,
        train_structure_module : bool, optional
            Whether to train structure module, by default True
        train_confidence_module : bool, optional
            Whether to train confidence module, by default True

        Returns
        -------
        model_out : dict[str, torch.Tensor]

            # When sample_structures is True:
            - sample:
                - coordinates: [B, N_samples, Ltoken, 3]
                    Sampled atom coordinates

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
            - confidence:
                # TODO
        """
        # Ensure batched input
        f_input = self.ensure_batched_input(f_input, do_warning=True)

        if train_confidence_module:
            assert sample_structures, (
                "To train confidence module, "
                "sample_structures must be True to provide sampled structures."
            )

        if not train_structure_module:
            # Set trunk and structure module to eval mode
            self.input_embedder.eval()
            self.trunk.eval()
            self.score_model.eval()

        # Output dictionary
        dict_out: dict[str, dict[str, torch.Tensor]] = {}

        s_inputs, s_init, z_init = self.input_embedder(f_input)

        seq_emb, seq_attn = self.sequence_encoder(f_input)
        struct_emb, _ = self.structure_encoder(f_input)

        # Trunk with recycling
        trunk_out = self.trunk(
            s_inputs,
            s_init.float(),
            z_init.float(),
            f_input,
            num_recycles,
            seq_emb=seq_emb,
            seq_attn=seq_attn,
            struct_emb=struct_emb,
        )
        s_trunk = trunk_out["s_trunk"]
        z_trunk = trunk_out["z_trunk"]

        if sample_structures:
            # Sample structures with Diffusion mini-rollout.
            # NOTE: We do not pass cache here to prevent that detached tensors
            # are stored in the model cache, which may lead to unexpected bugs with
            # diffusion module training. Instead, we construct cache inside
            # sample_structure method if necessary.
            self.score_model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
                coordinates = self.structure_module.sample_structure(
                    f_input=f_input,
                    s_inputs=s_inputs.detach(),
                    s_trunk=s_trunk.detach(),
                    z_trunk=z_trunk.detach(),
                    num_steps=num_steps,
                    num_diffusion_samples=num_diffusion_samples,
                    max_parallel_samples=None,
                )["sample_coordinates"]  # [B, N_samples, Ltoken, 3]
            sample_dict = {"coordinates": coordinates}
            dict_out["sample"] = sample_dict

        if train_structure_module:
            # Distogram head
            distogram_dict = {}
            distogram_dict["logits"] = self.distogram_head(z_trunk)
            if "z_aug" in trunk_out:
                # This is for distogram auxiliary loss with prime trunk output.
                z_aug = trunk_out["z_aug"]
                distogram_dict["logits_aug"] = self.distogram_head(z_aug)
            dict_out["distogram"] = distogram_dict

        if train_structure_module:
            # Diffusion head
            self.score_model.train()
            with torch.autocast("cuda", dtype=torch.float32):
                diffusion_dict = self.structure_module.training_step(
                    f_input,
                    s_inputs,
                    s_trunk,
                    z_trunk,
                    diffusion_batch_size,
                )
            dict_out["diffusion"] = diffusion_dict

        if train_confidence_module:
            # TODO: implement confidence prediction with mini-rollout
            coordinates = dict_out["sample"]["coordinates"]
            s_trunk_detached = s_trunk.detach()  # noqa
            z_trunk_detached = z_trunk.detach()  # noqa
            raise NotImplementedError("Confidence module is not implemented yet.")

        return dict_out

    def sample(
        self,
        f_input: FoldingInput,
        num_recycles: int = 10,
        num_steps: int = 200,
        num_diffusion_samples: int = 5,
        return_traj: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Forward pass of KFold model for model training.

        Parameters
        ----------
        f_input : FoldingInput
            Input data for folding model.
        num_recycles : int
            Number of recycling cycles in trunk.
        num_steps : int
            Number of diffusion steps for training.
        num_diffusion_samples : int
            Number of diffusion samples for training.
        return_traj : bool, optional
            Whether to return sampling trajectories.
        """
        dict_out: dict[str, torch.Tensor] = {}
        time_logs: dict[str, float] = {}

        # Indicate whether to return batched output
        return_batched_output = f_input.is_batched

        # Ensure batched input
        f_input = self.ensure_batched_input(f_input)

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
        struct_emb, _ = self.structure_encoder(f_input)
        et = time.time()
        time_logs["structure_encoder"] = et - st

        # Trunk with recycling
        st = time.time()
        trunk_out: dict[str, torch.Tensor] = self.trunk(
            s_inputs,
            s_init.float(),
            z_init.float(),
            f_input,
            num_recycles,
            seq_emb=seq_emb,
            seq_attn=seq_attn,
            struct_emb=struct_emb,
        )
        et = time.time()
        s_trunk = trunk_out["s_trunk"]
        z_trunk = trunk_out["z_trunk"]
        time_logs["trunk"] = et - st

        dict_out = {
            "seq_emb": seq_emb,
            "seq_attn": seq_attn,
            "s_trunk": s_trunk,
            "z_trunk": z_trunk,
        }

        # Distogram head
        st = time.time()
        dict_out["distogram_logits"] = self.distogram_head(z_trunk)
        et = time.time()
        time_logs["distogram_head"] = et - st

        # Diffusion head
        # pred_atom_coords: [B, Nsample, La, 3]
        st = time.time()
        with torch.autocast("cuda", dtype=torch.float32):
            dict_out.update(
                self.structure_module.sample_structure(
                    f_input,
                    s_inputs,
                    s_trunk,
                    z_trunk,
                    num_steps,
                    num_diffusion_samples,
                    return_traj=return_traj,
                )
            )
        et = time.time()
        time_logs["diffusion_head"] = et - st

        # TODO: Confidence head

        # If the input was not batched, remove the batch dimension
        if not return_batched_output:
            for key in dict_out:
                dict_out[key] = dict_out[key].squeeze(0)
        return dict_out, time_logs

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
            if missing_keys:
                missing_keys = {
                    key
                    for key in missing_keys
                    if not key.startswith(("sequence_encoder.", "structure_encoder."))
                }
            if missing_keys:
                raise KeyError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")
