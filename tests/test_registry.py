import torch

from kfold.config import _resolve_registry_defaults
from kfold.model.modules.sequence_encoder import BaseSequenceEncoder
from kfold.utils.registry import SEQUENCE_ENCODER, BaseConfig, Registry


@SEQUENCE_ENCODER.register()
class ExampleSequenceEncoder(BaseSequenceEncoder):
    class Config(BaseConfig):
        d_model: int = 128
        n_layers: int = 4

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        sequence_id: torch.Tensor,
        chain_id: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError


if __name__ == "__main__":
    from omegaconf import OmegaConf

    # Print all registered modules
    Registry.print_all_registered()

    # create example yaml file
    cfg = {
        "model": {
            "_registry_": "sequence_encoder",
            "_class_": "ExampleSequenceEncoder",
            "d_model": 256,
        },
        "train": {
            "optimizer": "adam",
            "lr_scheduler": "cosine",
            "lr": 1e-3,
        },
    }

    cfg = OmegaConf.create(cfg)
    print("Original Config:")
    print(OmegaConf.to_yaml(cfg))
    print()

    overridden_config = _resolve_registry_defaults(cfg)

    print("Overridden Config:")
    print(OmegaConf.to_yaml(overridden_config))

    # Create model
    model = Registry.instantiate(overridden_config.model)
    print(type(model))
