"""Typed configuration helpers."""

import dataclasses
from collections.abc import Mapping
from functools import wraps
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf

ConfigT = TypeVar("ConfigT")
ClassT = TypeVar("ClassT", bound=type)


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
