# from . import <files>
import importlib
from pathlib import Path

from .kfold import KFold, KFoldConfig

# Get all Python module names in current directory
_package_dir = Path(__file__).parent
_modules = [f.stem for f in _package_dir.glob("*.py") if f.stem != "__init__"]

# Import all modules
for _module in _modules:
    importlib.import_module(f".{_module}", package=__package__)

# Clean up
del Path, importlib, _package_dir, _modules

__all__ = ["KFold", "KFoldConfig"]


def load_model(model_config: KFoldConfig) -> KFold:
    """Utility function to load a folding model from its config."""
    from kfold.utils.registry import MAIN_MODULE

    model_cls = MAIN_MODULE[model_config._class_]
    return model_cls(model_config)
