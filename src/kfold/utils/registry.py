from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

from omegaconf import OmegaConf

C = TypeVar("C", bound=type[Any])
ConfigT = TypeVar("ConfigT", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base configuration class for registry objects."""

    _registry_: str
    _class_: str


class Registry:
    """Registry to store class with config"""

    __obj_dict__: dict[str, "Registry"] = {}

    def __init__(self, name: str):
        assert name not in self.__obj_dict__, f'Registry name "{name}" already exists!'
        self.name: str = name
        self.__obj_dict__[name] = self  # add registry

        self.__config_dict__: dict[str, type[BaseConfig]] = {}
        self._module_dict: dict[str, Any] = dict()

    @classmethod
    def get_register(cls, name: str) -> "Registry":
        assert name in cls.__obj_dict__, f"no registry with name '{name}' found"
        return cls.__obj_dict__[name]

    def __len__(self):
        return len(self._module_dict)

    def __getitem__(self, name: str) -> Any:
        module = self._module_dict.get(name, None)
        if module is None:
            raise KeyError(
                f"No object named '{name}' found in '{self.name}' registry"
                + f"Registry: {set(self._module_dict.keys())}"
            )
        return module

    def __contains__(self, name: str) -> bool:
        return name in self._module_dict

    @classmethod
    def instantiate(cls, config: BaseConfig, **kwargs) -> Any:
        """Instantiate a module from the registry using the provided config.

        Parameters
        ----------
        config : BaseConfig
            The configuration object containing `_registry_` and `_class_`
            attributes to identify the module to instantiate.

        Returns
        -------
        Any
            An instance of the requested module, initialized with the provided
            configuration.

        Raises
        ------
        AssertionError
            If the `_registry_` in the config does not match this registry's name.
        KeyError
            If the specified class is not found in this registry.
        """
        registry = cls.get_register(config._registry_)
        module_cls = registry[config._class_]
        return module_cls(config, **kwargs)

    def register(
        self, name: str | None = None, config_cls: type[BaseConfig] | None = None
    ) -> Callable[[C], C]:
        """Decorator to register a module in the registry.

        Parameters
        ----------
        name : str, optional
            The name to register the module under. If None, the module's
            `__name__` attribute will be used. Defaults to None.

        Returns
        -------
        Callable[[C], C]
            A decorator that registers the module.
        """

        def decorator(module_to_register: C) -> C:
            # Pass the module, explicit name (if any), and config class
            # to the internal registration method.
            return self._do_register(
                module_to_register,
                name_override=name,
                config_cls=config_cls,
            )

        return decorator

    def _do_register(
        self,
        module: C,
        name_override: str | None,
        config_cls: type[BaseConfig] | None,
    ) -> C:
        """Registers the module and optionally wraps.

        This method is called internally by the `register` decorator. It handles
        the actual registration of the module into the `_module_dict`.

        Parameters
        ----------
        module : C
            The class (module) to register. It is expected to be a type.
        name_override : str or None
            If provided, this name is used for registration. Otherwise,
            the `__name__` attribute of the `module` is used as the
            registration key.
        config_cls : type[BaseConfig] or None
            An optional configuration class to associate with the module.

        Returns
        -------
        C
            The registered module. if module has Config subclass, its __init__
            method is wrapped to accept a config object.
        Raises
        ------
        AssertionError
            If a module with the determined name (either `name_override` or
            `module.__name__`) is already registered in this registry.
        """
        # Determine the actual name to use for registration
        name = name_override if name_override is not None else module.__name__

        # add module to registry
        assert name not in self._module_dict, (
            f"An object named '{name}' was already registered in '{self.name}' registry!"
        )
        self._module_dict[name] = module

        if hasattr(module, "Config"):
            """If inner Config class exists, wrap the __init__ method to accept
            a config object and merge it with the default configuration.
            e.g.:
            class MyModule:
                class Config(BaseConfig):
                    param1: int = 10
                    param2: str = "default"

                def __init__(self, config: MyModule.Config):
                    self.config = config
            """
            assert issubclass(module.Config, BaseConfig), (
                f"The 'Config' attribute of module '{name}' must be"
                " a subclass of 'BaseConfig'."
            )
            config_cls = module.Config

        # If a config class is provided, wrap the __init__ method
        if config_cls is not None:
            # store the config class in the registry
            self.__config_dict__[name] = config_cls

            # wrap the __init__ method
            original_init = module.__init__

            @wraps(original_init)
            def wrapped_init(instance, config: Any, *args, **kwargs):
                # create a default configuration from the provided config dataclass
                merged_config = OmegaConf.merge(config_cls, config)
                merged_config = OmegaConf.structured(merged_config)
                return original_init(instance, merged_config, *args, **kwargs)

            module.__init__ = wrapped_init

        return module

    def print_registered(self) -> None:
        """Print all registered modules in the registry."""
        print(f"<Registry: {self.name}>")
        for name in self._module_dict.keys():
            print(f"  - {name}")

    @classmethod
    def print_all_registered(cls) -> None:
        """Print all registered modules in all registries."""
        for registry in cls.__obj_dict__.values():
            registry.print_registered()
            print()


# data
DATAMODULE = Registry("datamodule")
DATASET = Registry("dataset")
DATA_FILTER = Registry("data_filter")
DATA_SAMPLER = Registry("data_sampler")

# K-Fold module
MAIN_MODULE = Registry("main_module")

# Input encoder
SEQUENCE_ENCODER = Registry("sequence_encoder")
STRUCTURE_ENCODER = Registry("structure_encoder")

# Section 3.1 Algorithm 2 InputFeatureEmbedder,
INPUT_EMBEDDER = Registry("input_embedder")

# Section 3.6 Algorithm 17 Pairformer
TRUNK = Registry("trunk")

# Section 3.7 Algorithm 18 SampleDiffusion
STRUCTURE_MODULE = Registry("structure_module")
# Section 3.7 Algorithm 20 DiffusionModule
SCORE_MODEL = Registry("score_model")

# Section 3 Algorithm 1 Inference Loop
DISTOGRAM_HEAD = Registry("distogram_head")

CONFIDENCE_HEAD = Registry("confidence_head")

AFFINITY_HEAD = Registry("affinity_head")
