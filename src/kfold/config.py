from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from kfold.utils.registry import Registry


def load_config(path: str | Path) -> DictConfig:
    """
    Load a configuration file from the given path with recursive _yaml_ inheritance.

    Args:
        path (str | Path): The path to the configuration file.

    Returns:
        DictConfig: The loaded configuration as a DictConfig object.
    """
    config: DictConfig = OmegaConf.load(path)
    config = _resolve_yaml_inheritance(config, Path(path).parent)
    config = _resolve_registry_defaults(config)
    return config


def _resolve_yaml_inheritance(config: DictConfig, base_path: Path) -> DictConfig:
    container: dict = OmegaConf.to_container(config, resolve=True)

    def _resolve_yaml_inheritance(obj: Any, base_path: Path) -> Any:
        """Recursively resolve _yaml_ inheritance in nested structures."""
        if not isinstance(obj, dict):
            return obj

        # Check if current dict has _yaml_ and resolve it first
        if "_yaml_" in obj:
            yaml_path = base_path / obj.pop("_yaml_")
            base_config: DictConfig = load_config(yaml_path)
            OmegaConf.set_struct(base_config, True)  # avoid invalid override
            obj = OmegaConf.to_container(OmegaConf.merge(base_config, obj))

        # Recursively process all nested dicts
        resolved = {}
        for key, value in obj.items():
            resolved[key] = _resolve_yaml_inheritance(value, base_path)

        return resolved

    resolved_container = _resolve_yaml_inheritance(container, base_path)
    return OmegaConf.create(resolved_container)


def _resolve_registry_defaults(config: DictConfig) -> DictConfig:
    """Resolve and merge registry-based default configurations recursively."""

    def _resolve_nested(obj: Any) -> Any:
        """Recursively resolve registry defaults in nested structures."""
        if not isinstance(obj, dict):
            return obj

        # Check if current dict has registry info and resolve it first
        if "_registry_" in obj or "_class_" in obj:
            assert "_registry_" in obj and "_class_" in obj, (
                "Both '_registry_' and '_class_' must be specified in the config."
            )
            registry_name = obj["_registry_"]
            type_name = obj["_class_"]
            registry: Registry = Registry.get_register(registry_name)
            config_cls = registry.__config_dict__.get(type_name, None)
            try:
                if config_cls is not None:
                    # Merge with registry defaults
                    default_cfg = OmegaConf.structured(config_cls)
                    # add _registry_ and _class_ (ClassVar)
                    OmegaConf.set_struct(default_cfg, False)
                    default_cfg._registry_ = config_cls._registry_
                    default_cfg._class_ = config_cls._class_
                    OmegaConf.set_struct(default_cfg, True)  # avoid invalid override
                    obj = OmegaConf.to_container(OmegaConf.merge(default_cfg, obj))
            except Exception as e:
                raise Exception(
                    f"Failed to merge defaults for {registry_name}.{type_name} - {e}"
                ) from e

        # Recursively process all nested dicts
        resolved = {}
        for key, value in obj.items():
            resolved[key] = _resolve_nested(value)

        return resolved

    container: dict = OmegaConf.to_container(config)
    resolved_container = _resolve_nested(container)
    return OmegaConf.create(resolved_container)
