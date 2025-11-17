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

### For K-Fold Contributors (Elice B200 server):

#### Option A: Use Pre-built Dataset (Recommended)

A pre-generated LMDB dataset is available on the internal server:
`/mnt/parallel_storage/wykim_lab/icl_shwan/data/`

You can use this path as an argument for the training script directly and skip the data generation steps below.
```bash
export KFOLD_DATA_DIR=/cache/wykim_lab/kfold_data/

cd $KFOLD_DATA_DIR
cp -r /mnt/parallel_storage/wykim_lab/icl_shwan/data/structures/kfold_rcsb_processed_v251116.lmdb ./
cp -r /mnt/parallel_storage/wykim_lab/icl_shwan/data/manifests/ ./
```

#### Option B: Create New Dataset

If you need to regenerate the dataset from scratch, follow these steps.

**Source Data Paths:**
- Boltz1 Data: `/mnt/parallel_storage/wykim_lab/icl_mseok/BOLTZ1/rcsb_processed_targets/`
- ESMFold apo structures: `/mnt/parallel_storage/wykim_lab/icl_shwan/data/rcsb_apo_esmfold.tar.zstd`

##### Dataset Creation Steps:
To create the dataset, execute the following commands.

1. Copy source data to high-speed cache and extract: (You can skip each step if the files are already in the cache directory.)
  ```bash
  export CACHE_DIR=/cache/wykim_lab/
  export KFOLD_DATA_DIR=$CACHE_DIR/kfold_data/

  # Copy Boltz1 data
  cd $CACHE_DIR
  cp -r /mnt/parallel_storage/wykim_lab/icl_mseok/BOLTZ1/rcsb_processed_targets ./

  # Copy ESMFold apo structures
  cp /mnt/parallel_storage/wykim_lab/icl_shwan/data/rcsb_apo_esmfold.tar.zstd ./

  # Extract ESMFold apo structures
  tar --zstd -xvf rcsb_apo_esmfold.tar.zstd
  ```

2. Run the pre-processing script (approx. 5-6 mins with 192 CPUs):
  ```bash
  python ./scripts/process/boltz_rcsb/a1_preprocess_rcsb.py \
    --boltz_structure_dir $CACHE_DIR/rcsb_processed_targets/structures/ \
    --apo_structure_dir $CACHE_DIR/rcsb_apo_esmfold/ \
    --output_dir $KFOLD_DATA_DIR/kfold_rcsb_processed_v251116_npz/ \
    --num_cpus 192
  ```

3. Combine NPZ files into an LMDB database:
  ```bash
  python ./scripts/process/boltz_rcsb/a2_combine_lmdb.py \
    --npz_dir $KFOLD_DATA_DIR/kfold_rcsb_processed_v251116_npz/ \
    --lmdb_path $KFOLD_DATA_DIR/kfold_rcsb_processed_v251116.lmdb
  ```

4. Copy the final LMDB file from cache storage to persistent storage:
  ```bash
  cp $KFOLD_DATA_DIR/kfold_rcsb_processed_v251116.lmdb /scratch/<YOUR_DIRECTORY>/
  ```

##### Manifest Creation Steps:

To create the manifest file, run the following commands.

1. Construct entire manifest file:
  ```bash
  python ./scripts/process/boltz_rcsb/b_get_manifest.py \
    --boltz_manifest_path $CACHE_DIR/rcsb_processed_targets/manifest.json \
    --output_path $KFOLD_DATA_DIR/manifests/all_manifest.pkl
  ```

2. Construct AF3 manifest file (less than 300 chains):
  ```bash
  python ./scripts/process/boltz_rcsb/b_get_manifest.py \
    --boltz_manifest_path $CACHE_DIR/rcsb_processed_targets/manifest.json \
    --output_path $KFOLD_DATA_DIR/manifests/af3_manifest.pkl \
    --exclude_large_complex
  ```

3. Construct **complex only** manifest file:
  ```bash
  python ./scripts/process/boltz_rcsb/b_get_manifest.py \
    --boltz_manifest_path $CACHE_DIR/rcsb_processed_targets/manifest.json \
    --output_path $KFOLD_DATA_DIR/manifests/complex_manifest.pkl \
    --exclude_large_complex \
    --exclude_single_chain
  ```

4. Construct **P-P & P-L complex only** manifest file:
  ```bash
  python ./scripts/process/boltz_rcsb/b_get_manifest.py \
    --boltz_manifest_path $CACHE_DIR/rcsb_processed_targets/manifest.json \
    --output_path $KFOLD_DATA_DIR/manifests/pp_pl_manifest.pkl \
    --exclude_large_complex \
    --exclude_single_chain \
    --exclude_nucleic_acids
  ```


### For Community Users (Public)

TODO

## Training

Once the environment is set up and the dataset is prepared, you can start training by running the `train.py` script.

Modify `config file` to match your training environment.

- Debug mode (single GPU, no workers, no safe data-loading):
  ```bash
  python scripts/train.py --config ./configs/train-af3-mini.yaml --debug
  ```

- Full training mode
  ```bash
  python scripts/train.py \
    --config ./configs/train-af3.yaml \
    --wandb
  ```
