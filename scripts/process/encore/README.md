# ENCORE protein–RNA distillation preprocessing

This pipeline selects Protenix-v1 predictions using:

```text
distillation == 1
ipTM >= 0.6
interface PDE (chain_pair_gpde_offdiag_mean) <= 3.0
```

Every passing prediction is a separate entry. Predictions with the same CIF
provider sample name share chain/interface `cluster_id`. `ClusterSampler` then
normalizes the aggregate sampling mass by the number of passing seeds.

The examples use:

```bash
DATA=/cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset
SOURCE=/cache/wykim_lab/icl_shwan/source/ENCORE
```

## 1. Select metadata and copy CIFs

```bash
.venv/bin/python scripts/process/encore/a1_select_copy.py \
  --source_dir "$SOURCE" \
  --data_dir "$DATA"
```

Outputs:

```text
ENCORE/metadata.csv
ENCORE/cif/{entry_id}.cif
```

The default cutoff selects 2,275 structures in 1,116 provider sample clusters.

## 2. Parse CIFs and construct training artifacts

```bash
.venv/bin/python scripts/process/encore/a2_process_cifs.py \
  --data_dir "$DATA" \
  --num_workers 64

.venv/bin/python scripts/process/encore/a3_construct_training_set.py \
  --data_dir "$DATA" \
  --num_workers 64
```

Important outputs:

```text
ENCORE/structure.lmdb
ENCORE/manifest.{json,msgpack}
ENCORE/sequences/unique_protein_sequences.fasta
ENCORE/sequences/unique_rna_sequences.fasta
ENCORE/sequences/sequence_mapping.tsv
```

## 3. Create apo and prior LMDBs

Run the protein apo sampler externally with `unique_protein_sequences.fasta`,
then materialize its outputs per training entity under:

```text
ENCORE/apo/protein/{source}/{entry_id}_{entity_id}/...
```

Build the apo lookup, apo LMDBs, and prior LMDBs:

```bash
.venv/bin/python scripts/process/encore/a5_create_apo_prior_lmdb.py \
  --data_dir "$DATA" \
  --num_workers 128
```

This creates source-specific protein apo and prior LMDBs:

```text
ENCORE/apo_lmdb/protein/{source}.lmdb
ENCORE/prior_lmdb/protein.lmdb
```

## 4. Tokenize apo structures by chunk

Run this step externally, typically as a Slurm array:

```bash
.venv/bin/python scripts/process/encore/a6_tokenize_apo.py \
  --data_dir "$DATA" \
  --ckpt_path /path/to/structure_encoder.ckpt \
  --chunk 0 \
  --num_chunk 16
```

## 5. Combine token chunks

```bash
.venv/bin/python scripts/process/encore/a7_combine_apo_token_lmdb.py \
  --data_dir "$DATA"
```

The final token LMDBs are written to the path below.

```text
ENCORE/apo_tok_lmdb/protein/{source}.lmdb
```
