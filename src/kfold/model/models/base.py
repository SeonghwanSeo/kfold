import dataclasses
import pathlib
import time
import warnings
from typing import Self

import torch

import kfold.model.modules as submodules
from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import MAIN_MODULE, BaseConfig, Registry


@dataclasses.dataclass(kw_only=True)
class KernelConfig:
    cuequivariance: bool = False


@dataclasses.dataclass(kw_only=True)
class BaseFoldingModelConfig:
    _class_: str = "BaseFoldingModel"
    compile_trunk: bool = False
    compile_score_model: bool = False
    compile_mode: str = "default"
    kernel: KernelConfig
    input_embedder: BaseConfig
    trunk: BaseConfig
    score_model: BaseConfig
    structure_module: BaseConfig
    distogram_head: BaseConfig
    interaction_head: BaseConfig | None = None
    # confidence_head: Baseconfig


@MAIN_MODULE.register()
class BaseFoldingModel(torch.nn.Module):
    def __init__(self, config: BaseFoldingModelConfig):
        super().__init__()
        self.config: BaseFoldingModelConfig = config
        kernel_config = config.kernel

        # Initialize sub-modules here using the config
        self.input_embedder: submodules.input_embedder.BaseInputEmbedder = (
            Registry.instantiate(config.input_embedder)
        )

        self.trunk: submodules.trunk.BaseTrunk = Registry.instantiate(
            config.trunk, kernel_config=kernel_config
        )

        self.score_model: submodules.score_model.BaseScoreModel = Registry.instantiate(
            config.score_model, kernel_config=kernel_config
        )

        # NOTE: structure module is not a torch.nn.Module
        # This handles diffusion sampling as well
        self.structure_module: submodules.structure_module.BaseStructureModule = (
            Registry.instantiate(config.structure_module, score_model=self.score_model)
        )

        # Heads
        self.distogram_head: submodules.distogram_head.BaseDistogramHead = (
            Registry.instantiate(config.distogram_head)
        )
        self.interaction_head: submodules.interaction_head.BaseInteractionHead | None = (
            Registry.instantiate(config.interaction_head)
            if config.interaction_head is not None
            else None
        )

        # self.confidence_head: submodules.confidence_head.BaseConfidenceHead = (
        #     Registry.instantiate(config.confidence_head)
        # )

        # Compile submodules
        # NOTE: (SeonghwanSeo) This is very slow... Right now, just disable them.
        self.trunk.compile(config.compile_trunk, config.compile_mode)
        self.score_model.compile(config.compile_score_model, config.compile_mode)

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
            dict_out["sample"] = {
                "coordinates": coordinates,
            }

        if train_structure_module:
            # Distogram head
            dict_out["distogram"] = {
                "logits": self.distogram_head(z_trunk),
            }

        if train_interaction_head and self.interaction_head is not None:
            dict_out["interaction"] = {
                "logits": self.interaction_head(z_trunk),
            }

        if train_structure_module:
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

        # Trunk with recycling
        st = time.time()
        trunk_out = self.trunk(
            s_inputs,
            s_init,
            z_init,
            f_input,
            num_recycles,
        )
        s_trunk, z_trunk = trunk_out[:2]
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

        if self.interaction_head is not None:
            st = time.time()
            dict_out["interaction_logits"] = self.interaction_head(z_trunk)
            et = time.time()
            time_logs["interaction_head"] = et - st

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

    @classmethod
    def from_checkpoint(
        cls,
        model_config: BaseFoldingModelConfig,
        ckpt_path: str | pathlib.Path,
        use_ema: bool = False,
        strict: bool = True,
    ) -> Self:
        """Load model from checkpoint."""
        # Initialize model
        model_cls = MAIN_MODULE[model_config._class_]
        model: torch.nn.Module = model_cls(model_config)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" not in ckpt:
            # Assume the checkpoint is a state_dict itself
            state_dict = ckpt
        elif use_ema:
            # Load EMA weights
            state_dict = ckpt["ema"]
        else:
            # Load regular weights
            state_dict = ckpt["state_dict"]

        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

        model.load_state_dict(state_dict, strict=strict)

        return model
