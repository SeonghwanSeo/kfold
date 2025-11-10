import warnings

import torch
from omegaconf import DictConfig

import kfold.model.modules as submodules
from kfold.data.model_input import FoldingInput
from kfold.utils.registry import MAIN_MODULE, Registry


@MAIN_MODULE.register()
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
        if hasattr(model_config, "compile_trunk"):
            self.trunk.compile(getattr(model_config, "compile_trunk", False))
        else:
            print(
                "Model config does not have 'compile_trunk' attribute."
                " Skipping trunk compilation."
            )

        if hasattr(model_config, "compile_score_model"):
            self.score_model.compile(getattr(model_config, "compile_score_model", False))
        else:
            print(
                "Model config does not have 'compile_score_model' attribute."
                " Skipping score model compilation."
            )
        # if getattr(model_config, "compile_confidence_head", False):
        #     self.confidence_head.compile()

    def forward(
        self,
        f_input: FoldingInput,
        num_recycles: int,
        num_steps: int,
        diffusion_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of KFold model for model training."""
        if not f_input.is_batched:
            # If single example, make it batched
            # However, this process copies tensors and may slow down the process
            warnings.warn(
                "Input is not batched. Adding batch dimension of size 1."
                " This copies tensors and may slow down the process."
                " Please batch your inputs before moving to device:"
                " f_input = FoldingInput.from_list([f_input])",
                UserWarning,
                stacklevel=2,
            )
            f_input = FoldingInput.from_list([f_input])

        # Embed inputs
        s_inputs, s_init, z_init = self.input_embedder(f_input)

        # Trunk with recycling
        s_trunk, z_trunk = self.trunk(
            s_inputs,
            s_init,
            z_init,
            f_input,
            num_recycles,
        )
        dict_out = {
            "s_trunk": s_trunk,
            "z_trunk": z_trunk,
        }

        # Distogram head
        dict_out["distogram_logits"] = self.distogram_head(z_trunk)

        # Diffusion head
        # t_hat: [B, N]
        # prior_atom_coords: [B, N, La, 3]
        # noised_atom_coords: [B, N, La, 3]
        # denoised_atom_coords: [B, N, La, 3]
        # label_atom_coords: [B, N, La, 3]
        dict_out |= self.structure_module.training_step(
            f_input,
            s_inputs,
            s_trunk,
            z_trunk,
            diffusion_batch_size,
        )

        # TODO: implement confidence prediction with mini-rollout
        return dict_out
