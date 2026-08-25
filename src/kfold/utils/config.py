"""Configuration loading and typed configuration helpers."""

import copy
import dataclasses
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf

from .registry import Registry

ConfigT = TypeVar("ConfigT")
ClassT = TypeVar("ClassT", bound=type)


def load_config(
    path: str | Path,
    override_args: list[str] | None = None,
) -> DictConfig:
    """Load a configuration file with recursive ``_yaml_`` inheritance."""
    config: DictConfig = OmegaConf.load(path)

    if override_args is not None:
        overrides = OmegaConf.from_dotlist(override_args)
        config = OmegaConf.merge(config, overrides)

    config = _resolve_yaml_inheritance(config, Path(path).parent)
    return _resolve_registry_defaults(config)


def print_config(config: DictConfig) -> None:
    """Print the configuration in a human-readable format."""
    print(OmegaConf.to_yaml(config))


def save_config(config: DictConfig, save_path: str | Path) -> None:
    """Save a configuration to a YAML file."""
    OmegaConf.save(config, save_path)


def to_dict(config: DictConfig) -> dict:
    """Convert a ``DictConfig`` to a resolved Python dictionary."""
    return OmegaConf.to_container(config, resolve=True)


def _resolve_yaml_inheritance(config: DictConfig, base_path: Path) -> DictConfig:
    container: dict = OmegaConf.to_container(config, resolve=True)

    def resolve(obj: Any, current_base_path: Path) -> Any:
        if isinstance(obj, dict):
            if "_yaml_" in obj:
                yaml_path = current_base_path / obj.pop("_yaml_")
                base_config = load_config(yaml_path)
                obj = OmegaConf.to_container(
                    OmegaConf.merge(base_config, obj), resolve=True
                )
            return {key: resolve(value, current_base_path) for key, value in obj.items()}
        if isinstance(obj, list):
            return [resolve(item, current_base_path) for item in obj]
        return obj

    return OmegaConf.create(resolve(container, base_path))


def _resolve_registry_defaults(config: DictConfig) -> DictConfig:
    """Resolve and merge registry-backed dataclass defaults recursively."""

    def resolve(obj: Any) -> Any:
        if isinstance(obj, list):
            return [resolve(item) for item in obj]
        if not isinstance(obj, dict):
            return obj

        is_registry_config = "_registry_" in obj or "_class_" in obj
        if not is_registry_config:
            return {key: resolve(value) for key, value in obj.items()}

        if "_registry_" not in obj or "_class_" not in obj:
            raise ValueError(
                "Both '_registry_' and '_class_' must be specified in the config."
            )

        registry_name = obj["_registry_"]
        type_name = obj["_class_"]
        registry = Registry.get_register(registry_name)
        config_cls = registry.__config_dict__.get(type_name)
        if config_cls is None:
            return {key: resolve(value) for key, value in obj.items()}

        resolved = {}
        provided_keys = set(obj)
        for field_name, field_info in config_cls.__dataclass_fields__.items():
            if field_name in obj:
                resolved[field_name] = resolve(obj[field_name])
                provided_keys.discard(field_name)
            elif field_info.default is not dataclasses.MISSING:
                resolved[field_name] = resolve(copy.deepcopy(field_info.default))
            elif field_info.default_factory is not dataclasses.MISSING:
                resolved[field_name] = resolve(field_info.default_factory())
            else:
                raise ValueError(
                    f"Missing required field '{field_name}' for "
                    f"{registry_name}.{type_name}."
                )

        resolved["_registry_"] = registry_name
        resolved["_class_"] = type_name
        provided_keys.difference_update(("_registry_", "_class_"))
        if provided_keys:
            valid_fields = set(config_cls.__dataclass_fields__)
            raise ValueError(
                f"Unknown fields {provided_keys} for {registry_name}.{type_name}."
                f" Valid fields: {valid_fields}"
            )
        return resolved

    container: dict = OmegaConf.to_container(config)
    return OmegaConf.create(resolve(container))


def resolve_config(
    config_cls: type[ConfigT],
    config: ConfigT | DictConfig | Mapping[str, Any] | None,
) -> ConfigT:
    """Merge a partial config with dataclass defaults and return a typed object."""
    if not dataclasses.is_dataclass(config_cls):
        raise TypeError(f"Config class must be a dataclass, got {config_cls!r}.")
    if isinstance(config, config_cls):
        return config

    schema = OmegaConf.structured(config_cls)
    merged = schema if config is None else OmegaConf.merge(schema, config)
    resolved = OmegaConf.to_object(merged)
    if not isinstance(resolved, config_cls):
        raise TypeError(f"Expected {config_cls.__name__}, got {type(resolved).__name__}.")
    return resolved


def configurable(cls: ClassT) -> ClassT:
    """Convert the first constructor config argument to ``cls.Config``."""
    config_cls = getattr(cls, "Config", None)
    if config_cls is None:
        raise TypeError(f"@configurable requires {cls.__name__}.Config.")
    if not dataclasses.is_dataclass(config_cls):
        raise TypeError(f"{cls.__name__}.Config must be a dataclass.")

    original_init = cls.__init__

    @wraps(original_init)
    def wrapped_init(self, cfg=None, *args, **kwargs):
        resolved_cfg = resolve_config(config_cls, cfg)
        original_init(self, resolved_cfg, *args, **kwargs)

    cls.__init__ = wrapped_init
    return cls
