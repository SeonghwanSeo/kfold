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

    def inference(
        self,
        f_input: FoldingInput,
        apo_dict: dict[int, dict[str, torch.Tensor]],
        num_recycles: int = 10,
        num_steps: int = 200,
        num_samples: int = 5,
        return_traj: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
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
        model_out : dict[str, torch.Tensor]
            Output dictionary containing sampled structures and intermediate features.
        time_logs : dict[str, float]
            Dictionary containing time taken for each module during sampling.
        """
        # Ensure batched input
        f_input = f_input.from_list([f_input]) if not f_input.is_batched else f_input

        # Sanity check: ensure batch size is 1 for sampling
        assert f_input.batch_size == 1, "Sampling currently only supports batch size of 1"

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
            return_traj=return_traj,
        )

        # remove batch dimension
        model_out = {k: v.squeeze(0) for k, v in model_out.items()}

        return model_out, time_logs

    def forward(
        self,
        f_input: FoldingInput,
        num_recycles: int = 3,
        num_steps: int = 20,
        num_samples: int = 1,
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
        num_samples : int
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

        # Output dictionary
        dict_out: dict[str, dict[str, torch.Tensor]] = {}

        s_inputs, s_init, z_init = self.input_embedder(f_input)

        seq_emb, seq_attn = self.sequence_encoder(f_input)
        struct_emb, _ = self.structure_encoder(f_input)

        # Trunk with recycling
        trunk_out = self.trunk(
            s_inputs,
            s_init,
            z_init,
            f_input,
            num_recycles,
            seq_emb=seq_emb,
            seq_attn=seq_attn,
            struct_emb=struct_emb,
        )
        s_trunk = trunk_out.pop("s_trunk").float()
        z_trunk = trunk_out.pop("z_trunk").float()

        if sample_structures:
            # Sample structures with Diffusion mini-rollout.
            # NOTE: We do not pass cache here to prevent that detached tensors
            # are stored in the model cache, which may lead to unexpected bugs with
            # diffusion module training. Instead, we construct cache inside
            # sample_structure method if necessary.
            coordinates = self.structure_module.sample_structure(
                f_input=f_input,
                s_inputs=s_inputs.detach(),
                s_trunk=s_trunk.detach(),
                z_trunk=z_trunk.detach(),
                num_steps=num_steps,
                num_samples=num_samples,
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
        num_samples: int = 5,
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
        num_samples : int
            Number of diffusion samples for training.
        return_traj : bool, optional
            Whether to return sampling trajectories.
        """
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
            s_init,
            z_init,
            f_input,
            num_recycles,
            seq_emb=seq_emb,
            seq_attn=seq_attn,
            struct_emb=struct_emb,
        )
        et = time.time()
        s_trunk = trunk_out["s_trunk"].float()
        z_trunk = trunk_out["z_trunk"].float()
        del trunk_out
        time_logs["trunk"] = et - st

        dict_out = {
            "seq_emb": seq_emb,
            "seq_attn": seq_attn,
            "struct_emb": struct_emb,
            "s_inputs": s_inputs,
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
        dict_out.update(
            self.structure_module.sample_structure(
                f_input,
                s_inputs,
                s_trunk,
                z_trunk,
                num_steps,
                num_samples,
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
            missing_keys = {
                k
                for k in missing_keys
                if not k.startswith(("sequence_encoder.", "structure_encoder."))
            }
            # Skip some fourier-related keys that are changed from
            # nn.Parameter(..., required_grad=False) to buffer. (Backward compatibility)
            unexpected_keys = {k for k in unexpected_keys if ".fourier_embed." not in k}

            if missing_keys:
                raise KeyError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")
