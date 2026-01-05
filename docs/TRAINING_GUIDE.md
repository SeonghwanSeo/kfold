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
The training dataset is prepared based on the pre-processed dataset provided by [Boltz1 Official Github](https://github.com/jwohlwend/boltz/blob/v1.0.0/docs/training.md#download-the-pre-processed-data).

### For K-Fold Consortium Users (Internal)

#### Option A: Use Pre-built Dataset (Recommended)

Pre-generated datasets are available on the internal server:
`/storage/wykim_lab/icl_shwan/dataset/`.

You can use this path as an argument for the training script directly and skip the data generation steps below.
```bash
export KFOLD_DATA_DIR=/cache/wykim_lab/kfold_data/

# Navigate to the data directory
cd $KFOLD_DATA_DIR

# Clone pre-built dataset (v260103)
cp -r /storage/wykim_lab/icl_shwan/dataset/v260103/ .

# Extract the datasets you need
cd v260103/dataset
tar --zstd -xvf rcsb-train.tar.zst
tar --zstd -xvf rcsb-val.tar.zst
tar --zstd -xvf NaturalAb.tar.zst
...
```

#### Option B: Create New Dataset

If you need to regenerate the dataset from scratch, follow these steps.

**Source Data Paths:**
- RCSB:
  - mmCIF files: `/mnt/parallel_storage/wykim_lab/icl_shwan/raw_data/rcsb/rcsb.tar`
  - apo structures: `/mnt/parallel_storage/wykim_lab/icl_shwan/raw_data/rcsb/apo.tar`

##### Dataset Creation Steps:
To create the dataset, execute scripts in `scripts/process/rcsb/` sequentially as follows

TODO write...

##### Custom Dataset Creation Steps:

To create the manifest files to train with subsets of the dataset, execute scripts in `scripts/process/manifest/` sequentially as follows:

TODO write...


### For Community Users (Public)

TODO

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
