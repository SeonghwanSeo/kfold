# K-Fold Developer Guide

This document provides a comprehensive overview of the K-Fold project, including its architecture, training pipeline, and inference procedures.
It serves as a reference for developers and contributors working with the K-Fold framework.

**Note:** The notations and variable names used in this codebase largely follow those in the Alphafold3 paper for consistency.

## Quick Start

### Installation

#### Docker

TODO write...


#### Conda Environment
```bash
conda create -n kfold python=3.11
pip install -e .
pre-commit install
```

#### UV Environment
```bash
uv venv --python 3.11
uv pip install -e .
pre-commit install
```

## Project Architecture

### Core Components

- **`src/kfold/`**: Core package containing the model implementation
  - **`constants/`**: Constants used across the codebase
    - `chain.py`: Chain type definitions
    - `residue.py`: Residue type definitions and mappings
    - `atom.py`: Atom type definitions and mappings

  - **`data/`**: Core data structures and representations
    - `tokenized.py`: High-level tokenized numpy array interfaces (`TokenizedStructure`).
    - `model_input.py`: High-level (batched) tensor interfaces (`FoldingInput`).
    - `metadata.py`: Metadata structures for datasets.
    - `featurize.py`: Functions to convert tokenized representations and model inputs.
    - `sequence_tokenizer.py`: Tokenizers for protein, DNA and RNA

  - **`inference/`**: TODO: implement API for inference

  - **`models/`**: Core K-Fold model implementations
    - `models/`: Main K-Fold model classes
    - `modules/`: Submodules used in the K-Fold model (e.g., sequence encoder, ...)

  - **`training/`**: Training pipeline components (pytorch-lightning)

- **`scripts/`**: Utility scripts for data processing, training, and evaluation.
  - **`train.py`**: Script to train the K-Fold model.

## Training Pipeline

```bash
python scripts/train.py configs/af3.yaml
```

## Inference Framework

### Benchmarking

## Implementation Guidelines

**TODO Writing...**

### Module Structure
I introduce `Registry` to manage different implementations of modules such as sequence encoders, structure modules, etc.
When you want to add a new module, please cite the following code snippets:

#### Implementing a new Structure Module
- Separate Config class style
  ```python
  from kfold.models.modules.registry import STRUCTURE_MODULES, BaseConfig

  from .base import BaseStructureModule

  class MyStructureModuleConfig(BaseConfig):
    # Define your config parameters here
    # This class is automatically decorated by @dataclasses.dataclass
    param1: int = 128
    param2: float = 0.1

  @STRUCTURE_MODULES.register_module()
  class MyStructureModule(BaseStructureModule):
    def __init__(self, config: MyStructureModuleConfig):
      super().__init__(config)
  ```

- Nested Config class style
  ```python
  @STRUCTURE_MODULES.register_module()
  class MyStructureModule(BaseStructureModule):
    class Config(BaseConfig):
      param1: int = 128
      param2: float = 0.1

    def __init__(self, config: MyStructureModuleConfig):
      super().__init__(config)
  ```

#### Using the registered module in model

```python
from kfold.models.modules.registry import Registry

model_config = DictConfig({
    "_register_": "structure_module",
    "_class_": "MyStructureModule",
    "param1": 256,
    "param2": 0.2,
})
model = Registry.instantiate(model_config)
