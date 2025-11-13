"""Simple auto-import script for package initialization."""

import importlib
from pathlib import Path

from . import *  # noqa
from .base import BaseCropper

# Get all Python module names in current directory
_package_dir = Path(__file__).parent
_modules = [f.stem for f in _package_dir.glob("*.py") if f.stem != "__init__"]

# Import all modules
for _module in _modules:
    importlib.import_module(f".{_module}", package=__package__)

# Clean up
del Path, importlib, _package_dir, _modules
