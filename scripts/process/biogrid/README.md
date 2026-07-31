# BioGRID PPI distillation preprocessing

This pipeline selects T2/T3 Protenix-v1 predictions with:

```text
criterion_ipsae_max >= 0.7
criterion_iptm >= 0.7
criterion_pdockq2_fast_max >= 0.49
```

Every passing `(PPI, seed)` is a separate entry named
`{data_idx}__seed_{structure_idx}`. All chains and the interface receive
`cluster_id={data_idx}`. With `ClusterSampler`, this normalizes the aggregate
sampling mass of a PPI by its number of passing seeds.

The examples use:

```bash
DATA=/cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset
```

## 1. Select metadata and extract CIFs

The extractor streams each `.tar.zst` once and writes only selected CIFs.

```bash
.venv/bin/python scripts/process/biogrid/a1_select_extract.py \
  --data_dir "$DATA"
```

Outputs:

```text
Biogrid/metadata.csv
Biogrid/cif/{data_idx}__seed_{seed}.cif
```

## 2. Parse CIFs and construct training artifacts

```bash
.venv/bin/python scripts/process/biogrid/a2_process_cifs.py \
  --data_dir "$DATA" \
  --num_workers 64

.venv/bin/python scripts/process/biogrid/a3_construct_training_set.py \
  --data_dir "$DATA" \
  --num_workers 64
```

Important outputs:

```text
Biogrid/structure.lmdb
Biogrid/manifest.{json,msgpack}
Biogrid/sequences/unique_protein_sequences.fasta
Biogrid/sequences/sequence_mapping.tsv
```

## 3. Apo sampling and LMDB construction

Run the apo sampler externally with `unique_protein_sequences.fasta`, then put
its outputs under one directory per sampler source:

```text
Biogrid/uniq_apo/protein/{source}/uniq_protein_1/...
Biogrid/uniq_apo/protein/{source}/uniq_protein_2/...
```

Before LMDB construction, materialize the unique predictions for each
sample/entity under the following contract:

```text
Biogrid/apo/protein/{source}/{entry_id}_{entity_id}/
```

The one-off materialization helper used for this dataset was intentionally
removed after preprocessing completed.

Build apo and prior LMDBs:

```bash
.venv/bin/python scripts/process/biogrid/a5_create_apo_prior_lmdb.py \
  --data_dir "$DATA" \
  --num_workers 128
```

## 4. Tokenize apo structures by chunk

Run this step externally, typically as a Slurm array:

```bash
.venv/bin/python scripts/process/biogrid/a6_tokenize_apo.py \
  --data_dir "$DATA" \
  --ckpt_path /path/to/structure_encoder.ckpt \
  --chunk 0 \
  --num_chunk 16
```

## 5. Combine token chunks

```bash
.venv/bin/python scripts/process/biogrid/a7_combine_apo_token_lmdb.py \
  --data_dir "$DATA"
```

The final token LMDBs are written to:

```text
Biogrid/apo_tok_lmdb/protein/{source}.lmdb
```
