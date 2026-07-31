# JASPAR Preprocessing

This pipeline builds the JASPAR protein-DNA distillation dataset from the raw
synthetic outputs under `JASPAR/`.

Default root data directory used in examples:

```bash
/cache/wykim_lab/icl_shwan/source
```

All scripts take the root directory through `--data_dir` and append `JASPAR/`
internally.

## Required Inputs

```text
JASPAR/metadata.csv
JASPAR/cif/{data_idx}.cif
```

By default, scripts only use rows with `distillation == 1`.

## Steps

1. Convert CIF files to RefStructure NPZ files.

```bash
.venv/bin/python scripts/process/JASPAR/a2_process_cifs.py \
  --data_dir /cache/wykim_lab/icl_shwan/source \
  --ccd_path /path/to/ccd.pkl \
  --num_workers 64 \
  --overwrite
```

If `--ccd_path` is omitted, the script uses `/cache/wykim_lab/icl_shwan/source/ccd.pkl`.

2. Extract protein and DNA sequences for successfully processed samples.

```bash
.venv/bin/python scripts/process/JASPAR/a1_extract_sequences.py \
  --data_dir /cache/wykim_lab/icl_shwan/source \
  --processed_only \
  --overwrite
```

Outputs:

```text
JASPAR/sequences/all_sequences.fasta
JASPAR/sequences/sequence.fasta
JASPAR/sequences/unique_protein_sequences.fasta
JASPAR/sequences/unique_dna_sequences.fasta
JASPAR/sequences/sequence_mapping.tsv
```

3. Pack NPZ files into training artifacts.

```bash
.venv/bin/python scripts/process/JASPAR/a3_construct_training_set.py \
  --data_dir /cache/wykim_lab/icl_shwan/source \
  --num_workers 64 \
  --overwrite
```

Outputs:

```text
JASPAR/structure.lmdb
JASPAR/manifest.msgpack
JASPAR/manifest.json
```

4. Create protein apo lookup/LMDB and prior LMDB from materialized apo outputs.

```bash
.venv/bin/python scripts/process/JASPAR/a4_create_apo_prior_lmdb.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset \
  --num_workers 128 \
  --overwrite
```

Outputs:

```text
JASPAR/apo_lookup.msgpack
JASPAR/apo_lmdb/protein/{source}.lmdb
JASPAR/prior_lmdb/protein.lmdb
```

5. Tokenize protein apo structures.

```bash
.venv/bin/python scripts/process/JASPAR/a5_tokenize_apo.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset \
  --ckpt_path /path/to/structure_encoder.ckpt \
  --chunk 0 \
  --num_chunk 1
```

For Slurm arrays, shard by `--chunk` and `--num_chunk`.

6. Combine apo token chunk LMDBs.

```bash
.venv/bin/python scripts/process/JASPAR/a6_combine_apo_token_lmdb.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset
```

Output:

```text
JASPAR/apo_tok_lmdb/protein/{source}.lmdb
```
