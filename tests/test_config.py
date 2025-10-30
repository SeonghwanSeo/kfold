from omegaconf import OmegaConf

from kfold.config import load_config

if __name__ == "__main__":
    example_config_path = "configs/example.yaml"

    print("Input Config:")
    print(OmegaConf.to_yaml(OmegaConf.load(example_config_path)))

    print("Resolved Config:")
    config = load_config("configs/example.yaml")
    print(OmegaConf.to_yaml(config))
