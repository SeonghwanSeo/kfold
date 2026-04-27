# K-Fold Training Guide

Author: Seonghwan Seo (Prof. Woo Youn Kim's Lab)

This document provides detailed instructions on setting up the environment, preparing the dataset, and running training using the K-Fold framework.

## Contents
- [Installation](#installation)
- [Preparing the Dataset](#preparing-the-dataset)
- [Training](#training)

## Installation
Before starting, ensure that the K-Fold package and its dependencies for training are installed.

#### Docker

TODO write...

#### Conda Environment
```bash
conda create -n kfold python=3.11
pip install -e '.[train,dev]'
pre-commit install
```

#### UV Environment
```bash
uv venv --python 3.11
uv pip install -e '.[train,dev]'
pre-commit install
```

## Preparing the Dataset

### For K-Fold Consortium Users (Internal)

#### Option A: Use Pre-built Dataset (Recommended)

Pre-generated datasets are available on the internal server:
`/storage/wykim_lab/icl_shwan/dataset/`.

You can use this path as an argument for the training script directly and skip the data generation steps below.
```bash
export KFOLD_DATA_DIR=/cache/wykim_lab/kfold_data/

# Navigate to the data directory
cd $KFOLD_DATA_DIR

# Clone pre-built dataset (latest: v260310)
cp -r /storage/wykim_lab/icl_shwan/dataset/v260310/ .

# Extract the datasets you need
cd v260130/dataset
tar --zstd -xvf rcsb-train.tar.zst
tar --zstd -xvf rcsb-val.tar.zst
tar --zstd -xvf NaturalAb.tar.zst
...
```

#### Option B: Create New Dataset

If you need to regenerate the dataset from scratch, follow these steps.

**Source Data Paths:**
- RCSB: `/storage/wykim_lab/icl_shwan/data/rcsb-260109/`

See [`scripts/process/rcsb/README.md`](../../scripts/process/rcsb/README.md) for instructions on downloading and preparing the RCSB PDB dataset.


### For Community Users (Public)

TODO: Add public data links.

## Training

Once the environment is set up and the dataset is prepared, you can start training by running the `train.py` script.

Modify `config file` to match your training environment.

- Debug mode (single GPU, no workers, no safe data-loading):
  ```bash
  python scripts/train.py --config ./configs/train-af3-tiny.yaml --debug

  # If you want to skip validation (e.g., validation not implemented yet), use --skip_val
  python ./scripts/train.py --config ./configs/train-af3-tiny.yaml --debug --skip_val
  ```

- Full training mode with prepared config file:
  ```bash
  python scripts/train.py \
    --config ./configs/train-af3.yaml \
    --wandb
  ```

- Advanced training using `--override` flag:
  ```bash
  python scripts/train.py \
    --config ./configs/train-af3.yaml \
    --wandb \
    --override \
      train.trainer.max_epochs=10 \
      train.global_hparams.max_tokens=512 ...
  ```
