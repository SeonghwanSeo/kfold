import urllib.request
from pathlib import Path

import torch
from omegaconf import DictConfig

import kfold.model.modules as submodules
from kfold.model.modules.distogram_head.boltz1 import Boltz1DistogramHead
from kfold.model.modules.input_embedder.boltz1_embedder import Boltz1InputEmbedder
from kfold.model.modules.trunk.boltz1_trunk import Boltz1PairformerTrunk
from kfold.utils.registry import MAIN_MODULE, Registry

from .kfold import KFold


@MAIN_MODULE.register()
class Boltz1(KFold):
    def __init__(self, global_config: DictConfig):
        torch.nn.Module.__init__(self)
        self.config = global_config

        # Initialize sub-modules here using the config
        model_config = global_config.model

        self.input_embedder: Boltz1InputEmbedder = Registry.instantiate(
            model_config.input_embedder
        )
        assert isinstance(self.input_embedder, Boltz1InputEmbedder)

        self.trunk: Boltz1PairformerTrunk = Registry.instantiate(model_config.trunk)
        assert isinstance(self.trunk, Boltz1PairformerTrunk)

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
        self.distogram_head: Boltz1DistogramHead = Registry.instantiate(
            model_config.distogram_head
        )
        assert isinstance(self.distogram_head, Boltz1DistogramHead)

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
                ("pairformer_module", "s_norm", "z_norm", "s_recycle", "z_recycle")
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
            if k.startswith(("msa_module", "structure_module", "confidence_module")):
                state_dict.pop(k)

        # Check that all keys have been used
        assert len(state_dict) == 0, f"Unused keys in state dict: {state_dict.keys()}"
        return
