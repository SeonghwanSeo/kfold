import time
import warnings

import torch
from omegaconf import DictConfig

import kfold.model.modules as submodules
from kfold.data.model_input import FoldingInput
from kfold.utils.registry import Registry


class KFold(torch.nn.Module):
    def __init__(self, global_config: DictConfig):
        super().__init__()
        self.config = global_config

        # Initialize sub-modules here using the config
        model_config = global_config.model
        # self.sequence_encoder: submodules.sequence_encoder.BaseSequenceEncoder = (
        #     Registry.instantiate(model_config.sequence_encoder)
        # )
        # self.structure_encoder: submodules.struct_encoder.BaseStructureEncoder = (
        #     Registry.instantiate(model_config.structure_encoder)
        # )

        self.input_embedder: submodules.input_embedder.BaseInputEmbedder = (
            Registry.instantiate(model_config.input_embedder)
        )

        self.trunk: submodules.trunk.BaseTrunk = Registry.instantiate(model_config.trunk)

        self.score_model: submodules.score_model.BaseScoreModel = Registry.instantiate(
            model_config.score_model
        )

        # NOTE: structure module is not a torch.nn.Module
        # This handles diffusion sampling as well
        self.structure_module: submodules.structure_module.BaseStructureModule = (
            Registry.instantiate(
                model_config.structure_module, score_model=self.score_model
            )
        )

        # Heads
        self.distogram_head: submodules.distogram_head.BaseDistogramHead = (
            Registry.instantiate(model_config.distogram_head)
        )

        # self.confidence_head: submodules.confidence_head.BaseConfidenceHead = (
        #     Registry.instantiate(model_config.confidence_head)
        # )

        # Compile submodules
        # NOTE: (SeonghwanSeo) This is very slow... Right now, just disable them.
        if getattr(model_config, "compile_trunk", False):
            self.trunk.compile(getattr(model_config, "compile_trunk", False))
        if getattr(model_config, "compile_score_model", False):
            self.score_model.compile(getattr(model_config, "compile_score_model", False))
        # if getattr(model_config, "compile_confidence_head", False):
        #     self.confidence_head.compile()

    def forward(
        self,
        f_input: FoldingInput,
        num_cycles: int = 4,
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
        num_cycles : int
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

        # Initialize model cache
        model_cache: dict = {}

        # Embed inputs
        s_inputs, s_init, z_init = self.input_embedder(
            f_input=f_input,
            model_cache=model_cache,
        )

        # Trunk with recycling
        # NOTE: In trunk, we do not use cache.
        s_trunk, z_trunk = self.trunk(
            s_inputs,
            s_init,
            z_init,
            f_input,
            num_cycles,
        )

        if sample_structures:
            # Sample structures with Diffusion mini-rollout.
            # NOTE: We do not pass cache here to prevent that detached tensors
            # are stored in the model cache, which may lead to unexpected bugs with
            # diffusion module training. Instead, we construct cache inside
            # sample_structure method if necessary.
            self.score_model.eval()
            with torch.no_grad() and torch.autocast("cuda", dtype=torch.float32):
                coordinates = self.structure_module.sample_structure(
                    f_input=f_input,
                    s_inputs=s_inputs.detach(),
                    s_trunk=s_trunk.detach(),
                    z_trunk=z_trunk.detach(),
                    num_steps=num_steps,
                    num_diffusion_samples=num_diffusion_samples,
                    max_parallel_samples=None,
                )  # [B, N_samples, Ltoken, 3]
            dict_out["sample"] = {
                "coordinates": coordinates,
            }

        if train_structure_module:
            # Distogram head
            dict_out["distogram"] = {
                "logits": self.distogram_head(z_trunk),
            }

            # Diffusion head
            self.score_model.train()
            with torch.autocast("cuda", dtype=torch.float32):
                dict_out["diffusion"] = self.structure_module.training_step(
                    f_input,
                    s_inputs,
                    s_trunk,
                    z_trunk,
                    diffusion_batch_size,
                )

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
        num_cycles: int,
        num_steps: int,
        num_diffusion_samples: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Forward pass of KFold model for model training.

        Parameters
        ----------
        f_input : FoldingInput
            Input data for folding model.
        num_cycles : int
            Number of recycling cycles in trunk.
        num_steps : int
            Number of diffusion steps for training.
        num_diffusion_samples : int
            Number of diffusion samples for training.
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

        # Trunk with recycling
        st = time.time()
        s_trunk, z_trunk = self.trunk(
            s_inputs,
            s_init,
            z_init,
            f_input,
            num_cycles,
        )
        et = time.time()
        time_logs["trunk"] = et - st

        dict_out = {
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
        dict_out["coordinates"] = self.structure_module.sample_structure(
            f_input,
            s_inputs,
            s_trunk,
            z_trunk,
            num_steps,
            num_diffusion_samples,
        )
        et = time.time()
        time_logs["diffusion_head"] = et - st

        # TODO: Confidence head

        # If the input was not batched, remove the batch dimension
        if not return_batched_output:
            for key in dict_out:
                dict_out[key] = dict_out[key].squeeze(0)
        return dict_out, time_logs

    # === Helper functions === #
    def ensure_batched_input(
        self,
        f_input: FoldingInput,
        do_warning: bool = False,
    ) -> FoldingInput:
        """Ensure the input is batched. If not, add batch dimension of size 1.

        Parameters
        ----------
        f_input : FoldingInput
            Input data for folding model.

        Returns
        -------
        f_input_batched : FoldingInput
            Batched input data for folding model.
        """
        if not f_input.is_batched:
            # If single example is given, make it batched.
            # However, this process copies tensors.
            if do_warning:
                warnings.warn(
                    "Input is not batched. Adding batch dimension of size 1."
                    " This copies tensors and may slow down the process."
                    " Please batch your inputs before moving to device:\n"
                    "\tf_input = FoldingInput.from_list([f_input])",
                    UserWarning,
                    stacklevel=2,
                )
            f_input = FoldingInput.from_list([f_input])
        return f_input
