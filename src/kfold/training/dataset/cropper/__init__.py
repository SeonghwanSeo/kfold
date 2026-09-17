# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple auto-import script for package initialization."""

import importlib
from pathlib import Path

from .base import BaseCropper

# Get all Python module names in current directory
_package_dir = Path(__file__).parent
_modules = [f.stem for f in _package_dir.glob("*.py") if f.stem != "__init__"]

# Import all modules
for _module in _modules:
    importlib.import_module(f".{_module}", package=__package__)

# Clean up
del Path, importlib, _package_dir, _modules
