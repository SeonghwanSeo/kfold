# K-Fold Developer Guide

Author: Seonghwan Seo (Prof. Woo Youn Kim's Lab)

This document provides a comprehensive overview of the K-Fold project, including its architecture, training pipeline, and inference procedures.
It serves as a reference for developers and contributors working with the K-Fold framework.

**Note:** The notations and variable names used in this codebase largely follow those in the Alphafold3 paper for consistency.

## Contents
- [Installation](#installation)
- [Project Architecture](#project-architecture)
- [Training Pipeline](#training-pipeline)
- [Inference Framework](#inference-framework)
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

- **`configs/`**: Configuration files for training and model settings.
  - `model/`: Model architecture configurations.
  - `train/`: Training pipeline configurations.

- **`src/kfold/`**: Core package containing the model implementation
  - **`constants/`**: Constants used across the codebase
    - `chain.py`: Chain type definitions
    - `residue.py`: Residue type definitions and mappings
    - `ccd.py`: Common chemical component (CCD) definitions
    - `atom.py`: Atom type definitions and mappings
    - `bond.py`: Bond type definitions and mappings
    - `sequence.py`: Sequence model-related constants
    - `training.py`: Training-related constants
    - `constraint.py`: TODO: Constraint-related constants

  - **`data/`**: Core data structures and representations
    - **`types/`**: High-level data structures
      - `structure.py`: High-level biomolecular structure (`RefStructure`)
      - `tokenized.py`: High-level numpy array representation (`TokenizedStructure`).
      - `model_input.py`: High-level (batched) tensor representation (`FoldingInput`).
      - `metadata.py`: Metadata of each biomolecular structure.
    - **`pipelines/`**: Data processing pipelines
      - `structure_preparation.py`: Structure preparation pipeline.
      - `cif_factory.py`: Training data processing pipeline from mmCIF files.
      - `prior_sampling.py`: Prior structure sampling and augmentation.
      - `tokenization.py`: Tokenization pipeline converting `RefStructure` to `TokenizedStructure`.
      - `featurization.py`: Featurization pipeline converting `TokenizedStructure` to `FoldingInput`.
    - **`utils/`**: Utility functions for data processing

  - **`inference/`**:
      - `query.py`: Inference query representation (`Query`) and related pipelines (from `input_yaml` to `Query`).
      - `data_pipeline.py`: Inference data processing pipeline (from `Query` to `FoldingInput`).
      - `dataset.py`: Inference dataset and dataloader implementations.
      - `pl_client.py`: PyTorch Lightning client for inference with multi-GPU support.

  - **`models/`**: Core K-Fold model implementations
    - `models/`: Main K-Fold model classes
    - `modules/`: Submodules used in the K-Fold model (e.g., sequence encoder, structure module, ...)
    - `layers/`: Layer implementations for co-folding
      - `primitives/`: Basic building blocks (linear, attention, triangle updates, AdaLN)
      - `alphafold3/`: Layers based on the Alphafold3 architecture (pairformer, diffusion, embeddings)
      - `kfold/`: Custom layers specific to K-Fold architecture (PLM module, ECSI modules)

  - **`training/`**: Training pipeline components (pytorch-lightning)
    - `loss/`: Loss function implementations (WeightedMSE, BondLoss, SmoothLDDTLoss)
    - `metrics/`: Validation metric implementations
    - `training_module.py`: Main LightningModule implementation

- **`scripts/`**: Utility scripts for data processing, training, and evaluation.
  - **`train.py`**: Script to train the K-Fold model.
  - **`validate.py`**: Script to validate the trained model on validation datasets.
  - **`inference.py`**: Script to perform inference using the trained model.
  - **`inference_multigpu.py`**: Script for multi-GPU inference.

---

## Training Pipeline

See [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) for detailed instructions on preparing datasets and training the K-Fold model.

---

## Inference Framework

### Inference

```bash
python scripts/inference.py --config <config_path> --checkpoint <checkpoint_path> --input <input_yaml> --out_dir <output_directory>
```

### Benchmark

(TODO: `EVALUATION_GUIDE.md` for detailed instructions on benchmarking the K-Fold model.)

---

## Implementation Guidelines

**TODO Writing...**

### Module Structure
Model submodules are instantiated directly from their typed configs. The registry is
reserved for structure modules because ECSI and EDM are selected at runtime.

#### Implementing a model submodule

```python
from dataclasses import dataclass

from kfold.utils.config import configurable

@configurable
class MyModule:
  @dataclass(kw_only=True)
  class Config:
    width: int = 128

  def __init__(self, cfg: Config, runtime_dependency):
    self.cfg = cfg
```

`@configurable` accepts a typed `Config`, mapping, or `DictConfig`. Partial mappings
override dataclass defaults and are converted to the module's `Config` object.

#### Implementing a new Structure Module
- Separate config class style
  ```python
  from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig

  from .base import BaseStructureModule

  class MyStructureModuleConfig(BaseConfig):
    # Define your config parameters here
    # This class is automatically decorated by @dataclasses.dataclass
    param1: int = 128
    param2: float = 0.1

  @STRUCTURE_MODULE.register(config_cls=MyStructureModuleConfig)
  class MyStructureModule(BaseStructureModule):
    def __init__(self, config: Config):
      super().__init__(config)
  ```

- Nested config class style
  ```python
  @STRUCTURE_MODULE.register()
  class MyStructureModule(BaseStructureModule):
    class Config(BaseConfig):
      param1: int = 128
      param2: float = 0.1

    def __init__(self, config: MyStructureModuleConfig):
      super().__init__(config)
  ```

#### Using the registered module in model

```python
from kfold.utils.registry import Registry

model_config = DictConfig({
    "_registry_": "structure_module",
    "_class_": "MyStructureModule",
    "param1": 256,
    "param2": 0.2,
})
model = Registry.instantiate(model_config, score_model=score_model)
