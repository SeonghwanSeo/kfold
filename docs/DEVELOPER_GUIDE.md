# K-Fold Developer Guide

This document provides a comprehensive overview of the K-Fold project, including its architecture, training pipeline, and inference procedures.
It serves as a reference for developers and contributors working with the K-Fold framework.

**Note:** The notations and variable names used in this codebase largely follow those in the Alphafold3 paper for consistency. 

## Quick Start

### Installation

#### Docker
```bash
uv venv --python 3.11
uv pip install -e .
pre-commit install
```

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
    - `model_input.py`: High-level tokenized data interfaces (`ChainInput`, `FoldingInput`).
    - `sequence_tokenizer.py`: Tokenizers for protein, DNA and RNA

  - **`inference/`**: TODO: implement API for inference

  - **`models/`**: Core UBD model implementations
    - `models/`: Main K-Fold model classes
    - `modules/`: Submodules used in the K-Fold model (e.g., sequence encoder, ...)


## Data Preparation

### Dataset Splitting and Filtering

## Training Pipeline

### Data Loading Architecture


## Inference Framework

### Benchmarking

