# TPD Preprocessing

This pipeline builds the TPD distillation dataset from the raw synthetic
outputs under `raw_data/`.

Default root data directory used in examples:

```bash
/cache/wykim_lab/icl_shwan/kfold_data/v260701_af3
```

All scripts take the root directory through `--data_dir` and append `tpd/`
internally.

## Steps

1. Start from filter-passing CIF files and metadata prepared under `tpd/`.

Required inputs:

```text
tpd/cif/{subdb}/{subdb}__{data_idx}_{structure_idx}.cif
tpd/metadata/filtered_metadata.csv
```

2. Extract sequences for apo sampling.

```bash
.venv/bin/python scripts/process/tpd/a1_extract_sequences.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3 \
  --overwrite
```

Outputs:

```text
tpd/sequences/all_sequences.fasta
tpd/sequences/unique_protein_sequences.fasta
tpd/sequences/unique_rna_sequences.fasta
tpd/sequences/sequence_mapping.tsv
```

3. Convert CIF files to RefStructure NPZ files.

```bash
.venv/bin/python scripts/process/tpd/a2_process_cifs.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3 \
  --num_workers 64 \
  --overwrite
```

4. Pack NPZ files into training artifacts.

```bash
.venv/bin/python scripts/process/tpd/a3_construct_training_set.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3 \
  --num_workers 64 \
  --overwrite
```

Outputs:

```text
tpd/structure.lmdb
tpd/manifest.msgpack
tpd/manifest.json
```

5. Create apo lookup/LMDB and prior LMDB from generated apo structures.

Before this step, materialize apo sampler outputs under:

```text
tpd/apo/protein/{source}/{entry_id}_{entity_id}/
```

Then run:

```bash
.venv/bin/python scripts/process/tpd/a4_create_apo_prior_lmdb.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3 \
  --num_workers 128 \
  --overwrite
```

Outputs:

```text
tpd/apo_lookup.msgpack
tpd/apo_lmdb/protein/{source}.lmdb
tpd/prior_lmdb/protein.lmdb
```

6. Tokenize protein apo structures.

```bash
.venv/bin/python scripts/process/tpd/a5_tokenize_apo.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3 \
  --ckpt_path /path/to/structure_encoder.ckpt \
  --chunk 0 \
  --num_chunk 1
```

For Slurm arrays, shard by `--chunk` and `--num_chunk`.

7. Combine apo token chunk LMDBs.

```bash
.venv/bin/python scripts/process/tpd/a6_combine_apo_token_lmdb.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3
```

Output:

```text
tpd/apo_tok_lmdb/protein/{source}.lmdb
```
