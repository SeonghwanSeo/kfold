# K-Fold Developer Guide

Author: Seonghwan Seo (Prof. Woo Youn Kim's Lab)

This document provides a comprehensive overview of the K-Fold project, including its architecture, training pipeline, and inference procedures.
It serves as a reference for developers and contributors working with the K-Fold framework.

**Note:** The notations and variable names used in this codebase largely follow those in the Alphafold3 paper for consistency.

## Contents
- [Installation](#installation)
- [Project Architecture](#project-architecture)
- [Training Pipeline](#training-pipeline)
- [Inference Framework](#inference-framework) (TODO)
- [Implementation Guidelines](#implementation-guidelines)

## Installation

- **Docker**
    ```bash
    TODO write...
    ```
- **Conda**
  ```bash
  conda create -n kfold python=3.11
  pip install -e '.[train,dev]'
  pre-commit install
  ```
- **UV Virtual Python Environment**
  ```bash
  uv venv --python 3.11
  uv pip install -e '.[train,dev]'
  pre-commit install
  ```

## Project Architecture

### Core Components

- **`configs/`**: Configuration files for training and model settings.
  - `model/`: Model architecture configurations.
  - `train/`: Training pipeline configurations.

- **`src/kfold/`**: Core package containing the model implementation
  - **`constants/`**: Constants used across the codebase
    - `chain.py`: Chain type definitions
    - `residue.py`: Residue type definitions and mappings
    - `atom.py`: Atom type definitions and mappings
    - `bond.py`: Bond type definitions and mappings
    - `training.py`: Training-related constants
    - `constraint.py`: TODO: Constraint-related constants

  - **`data/`**: Core data structures and representations
    - `structure.py`: High-level numpy array interfaces (`TokenizedStructure`).
    - `model_input.py`: High-level (batched) tensor interfaces (`FoldingInput`).
    - `metadata.py`: Metadata structures for datasets.
    - `featurize.py`: A module and functions to convert `TokenizedStructure` to `FoldingInput`.
    - `apo_perturbation.py`: A module and functions for apo structure perturbation and augmentation.
    - `sequence_tokenizer.py`: Tokenizers for protein, DNA and RNA

  - **`inference/`**: TODO: implement API for inference

  - **`models/`**: Core K-Fold model implementations
    - `models/`: Main K-Fold model classes
    - `modules/`: Submodules used in the K-Fold model (e.g., sequence encoder, ...)
    - `layers/`: Layer implementations for co-folding
      - `primitives/`: Basic building blocks (e.g., linear, attention, ...)
      - `alphafold3/`: Layers based on the Alphafold3 architecture
      - `boltz1/`: Fork of Boltz1 layers modified for K-Fold compatibility
      - `kfold/`: Custom layers specific to K-Fold architecture

  - **`training/`**: Training pipeline components (pytorch-lightning)

- **`scripts/`**: Utility scripts for data processing, training, and evaluation.
  - **`train.py`**: Script to train the K-Fold model.
  - **`validate.py`**: Script to validate the trained model.

## Training Pipeline

### Model Training Command

```bash
python scripts/train.py --config configs/train-af3.yaml --wandb
```

### Model Validaiton Command

```bash
python scripts/validate.py --config configs/train-af3.yaml --checkpoint /path/to/checkpoint.ckpt
```

## Inference Framework

### Benchmarking

## Implementation Guidelines

**TODO Writing...**

### Module Structure
I introduce `Registry` to manage different implementations of modules such as sequence encoders, structure modules, etc.
When you want to add a new module, please cite the following code snippets:

#### Implementing a new Structure Module
- Separate config class style
  ```python
  from kfold.models.modules.registry import STRUCTURE_MODULES, BaseConfig

  from .base import BaseStructureModule

  class MyStructureModuleConfig(BaseConfig):
    # Define your config parameters here
    # This class is automatically decorated by @dataclasses.dataclass
    param1: int = 128
    param2: float = 0.1

  @STRUCTURE_MODULES.register(config_cls=MyStructureModuleConfig)
  class MyStructureModule(BaseStructureModule):
    def __init__(self, config: MyStructureModuleConfig):
      super().__init__(config)
  ```

- Nested config class style
  ```python
  @STRUCTURE_MODULES.register()
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
