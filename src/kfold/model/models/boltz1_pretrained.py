import dataclasses
import time
import urllib.request
from pathlib import Path

import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.distogram_head.boltz1 import Boltz1DistogramHead
from kfold.model.modules.input_embedder.boltz1_embedder import Boltz1InputEmbedder
from kfold.model.modules.trunk.boltz1_trunk import Boltz1PairformerTrunk
from kfold.utils.registry import MAIN_MODULE

from .base import BaseFoldingModel, BaseFoldingModelConfig


@dataclasses.dataclass(kw_only=True)
class Boltz1PretrainedConfig(BaseFoldingModelConfig):
    _class_: str = "Boltz1Pretrained"
    proj_s_inputs: bool = False


@MAIN_MODULE.register()
class Boltz1Pretrained(BaseFoldingModel):
    input_embedder: Boltz1InputEmbedder  # type: ignore
    trunk: Boltz1PairformerTrunk  # type: ignore
    distogram_head: Boltz1DistogramHead  # type: ignore

    def __init__(self, config: Boltz1PretrainedConfig):
        super().__init__(config)

        # === Boltz-1 pretrained modules === #
        assert isinstance(self.input_embedder, Boltz1InputEmbedder)
        assert isinstance(self.trunk, Boltz1PairformerTrunk)
        assert isinstance(self.distogram_head, Boltz1DistogramHead)

        # Load Boltz-1 pretrained weights
        self.load_boltz_weights()

        # NOTE: additional projection layers for compatibility with KFold
        self.proj_s_inputs = None
        need_projection: bool = config.proj_s_inputs
        if need_projection:
            c_input_boltz = 384 + 33 * 2 + 1 + 4  # 459
            self.proj_s_inputs = torch.nn.Linear(
                c_input_boltz,
                config.score_model.channel_s,
                bias=False,
            )

    def load_boltz_weights(self):
        # cache_dir = Path("/cache/wykim_lab/boltz1_weights")
        cache_dir = Path("/mnt/parallel_storage/wykim_lab/icl_swkim/kfold")

        model_path = cache_dir / "boltz1_conf.ckpt"
        state_dict_path = cache_dir / "boltz1_state_dict.ckpt"

        MODEL_URL = (
            "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
        )
        # Download model
        if not state_dict_path.exists():
            print("Downloading Boltz-1 weights...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, str(model_path))  # noqa: S310

            print("Extracting Boltz-1 model weights...")
            # Load weights
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
            # Save state dict
            torch.save(state_dict, state_dict_path)
        else:
            state_dict = torch.load(state_dict_path, map_location="cpu")

        input_embedder_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k.startswith(
                (
                    "input_embedder",
                    "s_init",
                    "z_init_1",
                    "z_init_2",
                    "rel_pos",
                    "token_bonds",
                )
            )
        }
        self.input_embedder.load_state_dict(input_embedder_state_dict)
        for k in list(input_embedder_state_dict.keys()):
            state_dict.pop(k)

        trunk_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k.startswith(
                (
                    "pairformer_module",
                    "s_norm",
                    "z_norm",
                    "s_recycle",
                    "z_recycle",
                    "msa_module",
                )
            )
        }
        self.trunk.load_state_dict(trunk_state_dict, strict=True)
        for k in list(trunk_state_dict.keys()):
            state_dict.pop(k)

        distogram_module_state_dict = {
            k.replace("distogram_module.", ""): v
            for k, v in state_dict.items()
            if k.startswith("distogram_module")
        }
        self.distogram_head.load_state_dict(distogram_module_state_dict, strict=True)
        for k in list(distogram_module_state_dict.keys()):
            state_dict.pop("distogram_module." + k)

        # Remove unused keys
        for k in list(state_dict.keys()):
            if k.startswith(("structure_module", "confidence_module")):
                state_dict.pop(k)

        # Check that all keys have been used
        assert len(state_dict) == 0, f"Unused keys in state dict: {state_dict.keys()}"

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
        """Override forward pass of Boltz1 pretrained model for
        compatibility with KFold structure module input dimensions"""

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

        # Trunk with recycling
        s_trunk, z_trunk = self.trunk(
            s_inputs,
            s_init,
            z_init,
            f_input,
            num_recycles,
        )

        # ====================================================== #
        # NOTE: Only the difference is here: Project single features to match
        if self.proj_s_inputs is not None:
            s_inputs = self.proj_s_inputs(s_inputs)
        # ====================================================== #

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
        num_recycles: int,
        num_steps: int,
        num_diffusion_samples: int,
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
            num_recycles,
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

        # ====================================================== #
        # NOTE: Only the difference is here: Project single features to match
        if self.proj_s_inputs is not None:
            s_inputs = self.proj_s_inputs(s_inputs)
        # ====================================================== #

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
                num_diffusion_samples,
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

    def freeze_modules(self):
        """Freeze Boltz-1 pretrained modules."""
        for param in self.input_embedder.parameters():
            param.requires_grad_(False)
        for param in self.trunk.parameters():
            param.requires_grad_(False)
        for param in self.distogram_head.parameters():
            param.requires_grad_(False)
