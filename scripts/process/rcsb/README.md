# RCSB Data Preprocessing Scripts

## Contents

- [Overview](#overview)
- [Download the pre-processed data](#download-the-pre-processed-data)
- [Processing from raw RCSB data](#processing-from-raw-rcsb-data)
  - [Raw data download and preparation](#raw-data-download-and-preparation)
  - [Data pre-processing steps](#data-pre-processing-steps)
    - [Step 1: Create output directories](#step-1-create-output-directories)
    - [Step 2: Prepare CCD](#step-2-prepare-ccd)
    - [Step 3: Prepare training set](#step-3-prepare-training-set)
    - [Step 4-1: Prepare validation split](#step-4-1-prepare-validation-split)
    - [Step 4-2: Construct validation set lmdb database](#step-4-2-construct-validation-set-lmdb-database)
  - [Pre-trained embedding extraction](#pre-trained-embedding-extraction)

## Overview

The pre-processed dataset structure for RCSB PDB data is as follows:

```
/data/processed/
    ccd-train.pkl               # CCD for model training
    ccd-test.pkl                # CCD for model inference
    /dataset/
        /rcsb-train/
            /apo/               # Raw apo/prior archives by chain type and source
            /apo_lmdb/          # Source-specific apo LMDBs
            /prior_lmdb/        # Chain-type prior stack LMDBs
            /apo_tok_lmdb/      # Protein apo structure tokens by source
            apo_multimer_lookup.msgpack
            /apo_multimer_lmdb/ # Source-specific multimer apo LMDBs
            /prior_multimer_lmdb/
            /embedding/
              /sequence/        # Pre-trained sequence embeddings
              /structure/       # Pre-trained structure embeddings
            structure.lmdb      # Training set lmdb database
            metadata.json       # Metadata file with cluster ids
            metadata.msgpack    # Binary metadata file
        /rcsb-val/
            /apo/               # Raw apo/prior archives by chain type and source
            /apo_lmdb/          # Source-specific apo LMDBs
            /prior_lmdb/        # Chain-type prior stack LMDBs
            /apo_tok_lmdb/      # Protein apo structure tokens by source
            apo_multimer_lookup.msgpack
            /apo_multimer_lmdb/ # Source-specific multimer apo LMDBs
            /prior_multimer_lmdb/
            /embedding/
              /sequence/        # Pre-trained sequence embeddings
              /structure/       # Pre-trained structure embeddings
            validation_ids.txt  # Validation pdb ids
            structure.lmdb      # Validation set lmdb database
            metadata.json       # Metadata file
            metadata.msgpack    # Binary metadata file
```

## Download the pre-processed data

TODO

## Processing from raw RCSB data

The following instructions guide you through the steps to process raw RCSB PDB data into a training set suitable for model training and validation.

Prerequisites:
- Environment with necessary dependencies installed `pip install -e .[train,dev]`
- Access to `mmseqs2` binary for clustering and similarity searches
- Sufficient disk space for raw and processed data (~100GB)
    - After removing intermediate files, ~40GB will be used

### Raw data download and preparation
You can download the raw data files from the RCSB PDB website using the following script:

```bash
# Create your own raw data directory
mkdir /raw_data/RCSB

# mmCIF files for each structure
rsync -rlpt -v -z --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ /raw_data/RCSB/mmCIF/

# Component mmCIF file
wget "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif" -O /raw_data/RCSB/components.cif
```


### Data pre-processing steps

**Step 1**: Create output directories

```bash
# All processed data will be saved under /data/processed
mkdir -p /data/processed
```

**Step 2**: Prepare CCD

Run the following script to prepare the Chemical Component Dictionary (CCD):

```bash
# Run the preparation script
# For data preprocessing and inference
python scripts/process/rcsb/a_process_ccd.py \
    --cif_path /raw_data/RCSB/components.cif \
    --out_path /data/processed/ccd-test.pkl

# For model training (up to 10 cached conformers per molecule)
python scripts/process/rcsb/a_process_ccd.py \
    --cif_path /raw_data/RCSB/components.cif \
    --out_path /data/processed/ccd-train.pkl \
    --train
```

**Step 3**: Pre-process cif files

Run the following script to process mmCIF files.

```bash
# Step 3-1. Extract sequences
python scripts/process/rcsb/b_extract_all_sequence.py \
    --cif_dir /raw_data/RCSB/mmCIF/ \
    --data_dir /data/processed/dataset \
    --split train \
    --num_workers 128

python scripts/process/rcsb/b_extract_all_sequence.py \
    --cif_dir /raw_data/RCSB/mmCIF/ \
    --data_dir /data/processed/dataset \
    --split val \
    --num_workers 128

# Step 3-2: Pre-process structures as npz files
python scripts/process/rcsb/c_process_cifs.py \
    --cif_path /raw_data/RCSB/mmCIF/ \
    --data_dir /data/processed/dataset \
    --split train \
    --num_workers 128

python scripts/process/rcsb/c_process_cifs.py \
    --cif_path /raw_data/RCSB/mmCIF/ \
    --data_dir /data/processed/dataset \
    --split val \
    --num_workers 128
```

**Step 3**: Construct training set

Run the following script to construct the training set lmdb database:

```bash
# Step 3-1: Run clustering and save metadata with cluster ids
python scripts/process/rcsb/d1_cluster.py \
    --data_dir /data/processed/dataset \
    --mmseqs "mmseqs2-binary-path" \
    --num_workers 128

# Step 3-2: Combine all npz files into a single lmdb database
python scripts/process/rcsb/d2_construct_training_set.py \
    --data_dir /data/processed/dataset
```

Finally, you can get the training set lmdb database at `/data/processed/dataset/rcsb-train/structure.lmdb` and the metadata file at `/data/processed/dataset/rcsb-train/metadata.json`.

**Step 4**: Construct validation set

Run the following scripts to construct the validation set lmdb database:

```bash
# Step 4-1: Get validation pdb ids
python scripts/process/rcsb/e1_get_val_ids.py \
    --data_dir /data/processed/dataset \
    --ccd_path /data/processed/ccd-test.pkl \
    --mmseqs "mmseqs2-binary-path" \
    --num_workers 128

# Step 4-2: Construct validation set
python scripts/process/rcsb/e2_construct_val_set.py \
    --data_dir /data/processed/dataset

# Step 4-3: (Optional) Get validation set statistics
python scripts/process/rcsb/e3_get_val_statistics.py \
    --data_dir /data/processed/dataset
```

Finally, you can get the validation set lmdb database at `/data/processed/dataset/rcsb-val/structure.lmdb` and the metadata file at `/data/processed/dataset/rcsb-val/metadata.json`.
The validation pdb ids are saved at `/data/processed/dataset/rcsb-val/validation_ids.txt`.

### Pre-trained embedding extraction

After Step 3 and Step 4, you can access unique polymer sequences from the training and validation sets: `/data/processed/dataset/rcsb-train/sequences/` and `/data/processed/dataset/rcsb-val/sequences/`.
You can use these sequences to extract pre-trained embeddings using your preferred protein language model (e.g., ESM).

```bash
# Example command for ESM-2 embedding extraction
esm-extract \
    --model esm2_t33_650M_UR50D \
    --input_fasta /data/processed/dataset/rcsb-train/sequences/unique_proteins.fasta \
    --output_dir /data/processed/dataset/rcsb-train/embedding/sequence/esm2-650m/ \
    ...
```

### Deterministic DNA apo structure preparation

DNA apo records are generated for DNA entities in `all_sequences.fasta` as
idealized single-stranded helices. By default the script writes
zstd-compressed `{pdb_id}_{entity_id}.pdb.zst` files under `apo/dna/dna_helix/`,
and the apo/prior LMDB creation step parses them into atom29 records.

```bash
python scripts/process/rcsb/f1_create_dna_apo.py \
    --data_dir /data/processed/dataset \
    --split train

python scripts/process/rcsb/f1_create_dna_apo.py \
    --data_dir /data/processed/dataset \
    --split val
```

### Sampler Apo And Prior LMDBs

Raw sampler outputs are stored under:

```text
rcsb-{split}/apo/
  protein/{source}/{pdb_id}_{entity_id}/...
  rna/{source}/{pdb_id}_{entity_id}/...
  dna/dna_helix/{pdb_id}_{entity_id}.pdb.zst
```

Build the source-specific apo LMDBs and chain-type prior stack LMDBs:

```bash
python scripts/process/rcsb/f2_create_apo_prior_lmdb.py \
    --data_dir /data/processed/dataset \
    --split train \
    --num_workers 128 \
    --map_size_gb 1024 \
    --overwrite
```

This writes:

```text
rcsb-{split}/apo_lookup.msgpack
rcsb-{split}/apo_lmdb/{chain_type}/{source}.lmdb
rcsb-{split}/prior_lmdb/{chain_type}.lmdb
```

### Antibody/Protein Multimer Apo And Prior LMDBs

SAbDab heavy/light pairs are resolved during preprocessing and stored as
runtime lookup metadata.  Multimer CIF chain IDs are sequence-matched to the
SAbDab H/L sequences before records are written with internal integer
`asym_id` keys.

```bash
python scripts/process/rcsb/g1_extract_sabdab_pairs.py \
    --data_dir /data/processed/dataset \
    --splits train val \
    --out_path /data/processed/dataset/rcsb-train/sequences/sabdab_heavy_light_pairs.csv

python scripts/process/rcsb/g2_create_apo_multimer_lookup.py \
    --data_dir /data/processed/dataset \
    --split train \
    --source prot_m_sampler \
    --source_dir /data/source/rcsb-train/apo-sampler/protein_multimer/prot_m_sampler_seed1-5_step200

python scripts/process/rcsb/g3_create_apo_multimer_lmdb.py \
    --data_dir /data/processed/dataset \
    --split train \
    --source prot_m_sampler \
    --source_dir /data/source/rcsb-train/apo-sampler/protein_multimer/prot_m_sampler_seed1-5_step200 \
    --num_workers 64 \
    --overwrite
```

This writes:

```text
rcsb-{split}/apo_multimer_lookup.msgpack
rcsb-{split}/apo_multimer_lmdb/protein/{source}.lmdb
rcsb-{split}/prior_multimer_lmdb/protein/{source}.lmdb
```

Protein apo structure tokens are generated from `apo_lmdb/protein/*.lmdb`:

```bash
python scripts/process/rcsb/h1_tokenize_apo_monomer.py --data_dir /data/processed/dataset --split train
python scripts/process/rcsb/h2_tokenize_apo_multimer.py --data_dir /data/processed/dataset --split train
python scripts/process/rcsb/h3_combine_apo_token_lmdb.py --data_dir /data/processed/dataset --split train
```
