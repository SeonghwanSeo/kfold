# RFAM Preprocessing

RFAM RNA monomer distillation keeps trunk apo inputs empty.  Only the prior
side is populated, using RNA sampler structures stored as
`prior_lmdb/rna.lmdb`.

## Sequence Extraction

```bash
.venv/bin/python scripts/process/rfam/a1_extract_sequences.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3
```

This writes:

```text
rfam/sequences/rfam_sequences.fasta
```

## Prior LMDB

```bash
.venv/bin/python scripts/process/rfam/a2_create_prior_lmdb.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3 \
  --num_workers 64 \
  --overwrite
```

By default the script reads RNA sampler outputs from:

```text
rfam/apo/rna/rna_sampler_seed1_step50/
```

If the sampler outputs are archived as `rna_sampler_seed1_step50.tar.zst`,
extract the archive first; this script intentionally reads only the directory
layout above.

Each output record is keyed as `{rfam_sample_id}_1`, matching the monomer
dataset entity id.
