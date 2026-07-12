# AFDB Heterodimer Preprocessing

This pipeline builds the simplified distillation artifacts consumed by
`HeterodimerDistillationDataset`. The input AF-M structures are assumed to have
already passed ipSAE/filtering, so this step only converts file format and
writes sequence files for apo/prior construction.

Raw CIF files are expected under:

```text
/scratch/icl_shwan/afdb_m/cifs
```

Gzip-compressed files should use the `.cif.gz` suffix.

Run:

```bash
.venv/bin/python scripts/process/afdb_heterodimer/a1_construct_training_set.py \
  --cif_dir /scratch/icl_shwan/afdb_m/cifs \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset \
  --num_workers 64 \
  --overwrite
```

Outputs:

```text
AFDB-heterodimer/structure.lmdb
AFDB-heterodimer/manifest.msgpack
AFDB-heterodimer/manifest.json
AFDB-heterodimer/sequences/all_sequences.fasta
AFDB-heterodimer/sequences/sequence.fasta
AFDB-heterodimer/sequences/unique_protein_sequences.fasta
AFDB-heterodimer/sequences/sequence_mapping.tsv
```

The output directory name intentionally matches the dataset name used by
`configs/dataset/distill-heterodimer.yaml`.

After placing the AF-M metadata table at:

```text
AFDB-heterodimer/metadata.csv
```

filter the manifest by ipSAE >= 0.8:

```bash
.venv/bin/python scripts/process/afdb_heterodimer/a2_filter_manifest.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset \
  --overwrite
```

The default metadata columns are `modelEntityId` for the manifest id and
`max_ipSAE` for the score. This writes `manifest_08.msgpack` and, if
`manifest.json` exists, `manifest_08.json`.
Because the manifest ids include the AF-M suffix, `-model_v1` is appended to
`modelEntityId` by default before matching.

Before building apo/prior LMDBs, materialize the unique apo predictions against
the full manifest:

```text
AFDB-heterodimer/apo/protein/<source>/<entry_id>_<entity_id>/
```

Run:

```bash
.venv/bin/python scripts/process/afdb_heterodimer/a3_materialize_apo_sources.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset \
  --source_dir /cache/wykim_lab/icl_shwan/source/hetero/apo_uniq \
  --overwrite
```

Create apo lookup, source-specific apo LMDBs, and prior stack LMDBs:

```bash
.venv/bin/python scripts/process/afdb_heterodimer/a3_create_apo_prior_lmdb.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset \
  --num_workers 128 \
  --overwrite
```

The default manifest is the full `manifest.msgpack`, so apo/prior artifacts are
available independently of a particular training threshold. This writes:

```text
AFDB-heterodimer/apo_lookup.msgpack
AFDB-heterodimer/apo_lookup.json
AFDB-heterodimer/apo_lmdb/protein/<source>.lmdb
AFDB-heterodimer/prior_lmdb/protein.lmdb
```

Tokenize apo structures with the protein structure tokenizer:

```bash
.venv/bin/python scripts/process/afdb_heterodimer/a4_tokenize_apo.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset \
  --ckpt_path /path/to/structure_encoder.ckpt \
  --chunk 0 \
  --num_chunk 1
```

Combine token chunk LMDBs:

```bash
.venv/bin/python scripts/process/afdb_heterodimer/a5_combine_apo_token_lmdb.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3/dataset
```

This writes `AFDB-heterodimer/apo_tok_lmdb/protein/<source>.lmdb`.
