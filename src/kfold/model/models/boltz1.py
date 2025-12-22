import dataclasses
import urllib.request
from pathlib import Path

import torch

from kfold.model.modules.distogram_head.boltz1 import Boltz1DistogramHead
from kfold.model.modules.input_embedder.boltz1_embedder import Boltz1InputEmbedder
from kfold.model.modules.score_model.boltz1_diffusion import Boltz1DiffusionModule
from kfold.model.modules.structure_module.boltz1_edm import Boltz1SampleDiffusion
from kfold.model.modules.trunk.boltz1_trunk import Boltz1PairformerTrunk
from kfold.utils.registry import MAIN_MODULE

from .base import BaseFoldingModel, BaseFoldingModelConfig


@dataclasses.dataclass(kw_only=True)
class Boltz1Config(BaseFoldingModelConfig):
    _class_: str = "Boltz1"
    load_weight: bool = True


@MAIN_MODULE.register()
class Boltz1(BaseFoldingModel):
    input_embedder: Boltz1InputEmbedder  # type: ignore
    trunk: Boltz1PairformerTrunk  # type: ignore
    distogram_head: Boltz1DistogramHead  # type: ignore
    score_model: Boltz1DiffusionModule  # type: ignore
    structure_module: Boltz1SampleDiffusion  # type: ignore

    def __init__(self, config: Boltz1Config):
        super().__init__(config)

        # === Boltz-1 pretrained modules === #
        assert isinstance(self.input_embedder, Boltz1InputEmbedder)
        assert isinstance(self.trunk, Boltz1PairformerTrunk)
        assert isinstance(self.distogram_head, Boltz1DistogramHead)
        assert isinstance(self.score_model, Boltz1DiffusionModule)
        assert isinstance(self.structure_module, Boltz1SampleDiffusion)

        # Load Boltz-1 pretrained weights
        if config.load_weight:
            self.load_boltz_weights()

    def load_boltz_weights(self):
        cache_dir = Path("/cache/wykim_lab/boltz1_weights")

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

        structure_module_state_dict = {
            k.replace("structure_module.score_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("structure_module.score_model")
        }
        self.score_model.diffusion_stack.load_state_dict(
            structure_module_state_dict, strict=True
        )
        for k in list(structure_module_state_dict.keys()):
            state_dict.pop("structure_module.score_model." + k)
        self.score_model.rel_pos_encoding.load_state_dict(
            self.input_embedder.rel_pos.state_dict(), strict=True
        )

        # Remove unused keys
        for k in list(state_dict.keys()):
            if k.startswith(
                (
                    "confidence_module",
                    "structure_module.out_token_feat_update",
                )
            ):
                state_dict.pop(k)

        # Check that all keys have been used
        assert len(state_dict) == 0, f"Unused keys in state dict: {state_dict.keys()}"

    def freeze_modules(self):
        """Freeze Boltz-1 pretrained modules."""
        for param in self.input_embedder.parameters():
            param.requires_grad_(False)
        for param in self.trunk.parameters():
            param.requires_grad_(False)
        for param in self.distogram_head.parameters():
            param.requires_grad_(False)
        for param in self.score_model.parameters():
            param.requires_grad_(False)
