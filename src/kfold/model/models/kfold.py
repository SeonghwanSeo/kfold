import dataclasses

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import MAIN_MODULE

from .base import BaseFoldingModel, BaseFoldingModelConfig


@dataclasses.dataclass(kw_only=True)
class KFoldConfig(BaseFoldingModelConfig):
    _class_: str = "KFold"
    # TODO: define encoders
    # sequence_encoder: BaseConfig
    # structure_encoder: BaseConfig


@MAIN_MODULE.register()
class KFold(BaseFoldingModel):
    def forward(
        self,
        f_input: FoldingInput,
        num_recycles: int = 3,
        num_steps: int = 20,
        num_diffusion_samples: int = 1,
        diffusion_batch_size: int = 48,
        sample_structures: bool = True,
        train_structure_module: bool = True,
        train_interaction_head: bool = True,
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
        train_interaction_head : bool, optional
            Whether to produce interaction logits, by default True
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
            - interaction:
                - logits: [B, Ltoken, Ltoken, K]
                    Interaction logits (K = num pair interaction types)
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

        embed_out = self.input_embedder(f_input)
        s_inputs, s_init, z_init = embed_out[:3]
        extra_embed_args = embed_out[3:]

        # Trunk with recycling
        trunk_out = self.trunk(
            s_inputs,
            s_init,
            z_init,
            f_input,
            num_recycles,
            *extra_embed_args,
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
                # (SeonghwanSeo) Trick to avoid `ddp_unused_parameters` issues:
                # When training only the priming trunk (recycle=0), the refining trunk
                # is not used and its parameters are not updated. Instead, we use this
                # auxiliary head to ensure gradients flow to the refining trunk.
                z_aug = trunk_out["z_aug"]
                distogram_dict["logits_aug"] = self.distogram_head(z_aug)
            dict_out["distogram"] = distogram_dict

        if train_interaction_head and self.interaction_head is not None:
            # Interaction head
            interaction_dict = {}
            interaction_dict["logits"] = self.interaction_head(z_trunk)
            dict_out["interaction"] = interaction_dict

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
