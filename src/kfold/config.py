import copy
from pathlib import Path
from typing import Any

from omegaconf import MISSING, DictConfig, OmegaConf

from kfold.utils.registry import Registry


def load_config(
    path: str | Path,
    override_args: list[str] | None = None,
    override_registry_defaults: bool = True,
) -> DictConfig:
    """
    Load a configuration file from the given path with recursive _yaml_ inheritance.

    Args:
        path (str | Path): The path to the configuration file.
        override_args (list[str] | None): A list of dotlist strings to override specific
            configuration values.
        override_registry_defaults (bool): Whether to override registry defaults.

    Returns:
        DictConfig: The loaded configuration as a DictConfig object.
    """
    config: DictConfig = OmegaConf.load(path)

    if override_args is not None:
        # Override specific arguments in the config
        overrides = OmegaConf.from_dotlist(override_args)
        config = OmegaConf.merge(config, overrides)

    config = _resolve_yaml_inheritance(config, Path(path).parent)
    if override_registry_defaults:
        config = _resolve_registry_defaults(config)
    return config


def print_config(config: DictConfig) -> None:
    """Print the configuration in a human-readable format."""
    print(OmegaConf.to_yaml(config))


def save_config(config: DictConfig, save_path: str | Path) -> None:
    """Save the configuration to a YAML file."""
    OmegaConf.save(config, save_path)


def to_dict(config: DictConfig) -> dict:
    """Convert a DictConfig to a standard Python dictionary."""
    return OmegaConf.to_container(config, resolve=True)


def _resolve_yaml_inheritance(config: DictConfig, base_path: Path) -> DictConfig:
    container: dict = OmegaConf.to_container(config, resolve=True)

    def _resolve_yaml_inheritance(obj: Any, base_path: Path) -> Any:
        """Recursively resolve _yaml_ inheritance in nested structures."""
        if not isinstance(obj, dict):
            return obj

        # Check if current dict has _yaml_ and resolve it first
        if "_yaml_" in obj:
            yaml_path = base_path / obj.pop("_yaml_")
            base_config = load_config(yaml_path, override_registry_defaults=False)
            obj = OmegaConf.merge(base_config, obj)

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
        if isinstance(obj, list):
            return [_resolve_nested(item) for item in obj]

        if not isinstance(obj, dict):
            return obj

        # Check if current dict has registry info and resolve it first
        is_registry_obj = "_registry_" in obj or "_class_" in obj
        if not is_registry_obj:
            return {key: _resolve_nested(value) for key, value in obj.items()}

        # Ensure both keys are present
        assert "_registry_" in obj and "_class_" in obj, (
            "Both '_registry_' and '_class_' must be specified in the config."
        )
        registry_name = obj["_registry_"]
        type_name = obj["_class_"]
        registry: Registry = Registry.get_register(registry_name)
        config_cls = registry.__config_dict__.get(type_name, None)

        if config_cls is None:
            # If no config class is registered, treat as normal dict
            return {key: _resolve_nested(value) for key, value in obj.items()}

        # Build resolved config
        resolved = {}
        provided_keys = set(obj.keys())

        # Process all fields from config class
        for field_name, field_info in config_cls.__dataclass_fields__.items():
            # Check if field is required (no default value)

            if field_name in obj:
                # User provided value - recursively resolve it
                resolved[field_name] = _resolve_nested(obj[field_name])
                provided_keys.discard(field_name)
            else:
                if field_info.default is not MISSING:
                    resolved[field_name] = _resolve_nested(
                        copy.deepcopy(field_info.default)
                    )
                elif field_info.default_factory is not MISSING:
                    resolved[field_name] = _resolve_nested(field_info.default_factory())
                else:
                    raise ValueError(
                        f"Missing required field '{field_name}' for "
                        f"{registry_name}.{type_name}."
                    )

        # Preserve registry metadata
        resolved["_registry_"] = registry_name
        resolved["_class_"] = type_name
        provided_keys.discard("_registry_")
        provided_keys.discard("_class_")

        # Check for unknown fields
        if provided_keys:
            raise ValueError(
                f"Unknown fields {provided_keys} for {registry_name}.{type_name}."
                f" Valid fields: {set(config_cls.__dataclass_fields__.keys())}"
            )
        return resolved

    container: dict = OmegaConf.to_container(config)
    resolved_container = _resolve_nested(container)
    return OmegaConf.create(resolved_container)
