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

K-Fold's training datasets are not distributed. Prepare your own predicted
structures as `RefStructure` records and use `type: distillation` for any chain
composition. See [Custom distillation datasets](CUSTOM_DATASET.md) for the
LMDB/manifest contract, configuration, and optional apo/prior inputs.

Experimental RCSB structures use `type: rcsb`. See the
[RCSB preprocessing guide](../../scripts/process/rcsb/README.md) for its current
train/validation commands starting from an RCSB mmCIF directory and `components.cif`.
Run CCD preparation (`a`) and sequence extraction (`b`), process both splits to NPZ (`c`), then
build train with `d1 → d2` before selecting/packing validation with `e1 → e2`.
For each split, follow with `f1 → [f2 → f3]`, `g`, and `h1 → [h2] → h3`
to prepare apo predictions, prior stacks, and structure tokens. The guide also
describes the validation selector's network dependency and apo length constraints.

Protein apo augmentation uses BioPrior. Set `prob_perturbation` to control how
often it is applied and `apo_perturb.bioprior` to configure it. The RCSB preset
uses probability `0.9` and `max_steps: 15`. Missing apo atoms remain masked after
augmentation; if BioPrior fails, the original coordinates are used.

## Training

Once the environment is set up and the dataset is prepared, you can start training by running the `train.py` script.

Modify `config file` to match your training environment.

- Debug mode (single GPU, no workers, no safe data-loading):
  ```bash
  python scripts/train.py --config configs/train/stage_1.yaml --debug

  # Disable validation.
  python scripts/train.py --config configs/train/stage_1.yaml --debug \
    --override train.trainer.limit_val_batches=0
  ```

- Full training mode with prepared config file:
  ```bash
  python scripts/train.py \
    --config configs/train/stage_1.yaml \
    --wandb
  ```

- Advanced training using `--override` flag:
  ```bash
  python scripts/train.py \
    --config configs/train/stage_1.yaml \
    --wandb \
    --override \
      train.trainer.max_epochs=10 \
      train.data.max_tokens=512 ...
  ```

### Batch size and epoch length

Set `train.data.train_batch_size` per GPU, `train.trainer.accumulate_grad_batches`
for gradient accumulation, and `train.trainer.limit_train_batches` for batches
per GPU per epoch. Effective global batch size is batch size × GPU count × node
count × accumulation. These values are used directly unless `--global_batch_size` is supplied. That
option computes gradient accumulation from the GPU/node count and per-GPU batch
size; the global batch size must be divisible by their product. It does not change
the number of batches per epoch.

`--batch_size` and `--num_batches_per_epoch` override the corresponding settings.
For example, 8 GPUs with batch size 1 and accumulation 32 give a global batch size
of 256; 32,000 batches per epoch give 1,000 optimizer steps.
