# Disordered PDB Preprocessing

These scripts build the `disordered_pdb` training dataset from mmCIF files and
RCSB train apo products.

## Expected Layout

Raw CIF files are expected under a directory such as:

```text
/cache/wykim_lab/icl_shwan/disordered_pdb/
  1abc/1abc.cif
  2xyz/2xyz.cif
```

Processed outputs are written under:

```text
{data_dir}/disordered_pdb/
  npz/
  structure.lmdb
  manifest.msgpack
  sequences/
    all_sequences.fasta
    uniq_sequences.fasta
  apo_lookup.msgpack
  apo_lmdb/{protein,dna,rna}/{source}.lmdb
  prior_lmdb/{protein,dna,rna}.lmdb
  apo_tok_lmdb/protein/{source}.lmdb
```

## Structure Dataset

```bash
python scripts/process/disordered_pdb/a1_process_cifs.py \
  --cif_dir /cache/wykim_lab/icl_shwan/disordered_pdb \
  --ccd_path /path/to/ccd-train.pkl \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3 \
  --num_workers 128

.venv/bin/python scripts/process/disordered_pdb/a2_extract_manifest_from_rcsb_train.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3 \
  --num_workers 128

python scripts/process/disordered_pdb/a3_construct_training_set.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3
```

## Apo LMDB

Materialize apo structures from already processed RCSB train apo archives.
This matches by entry id and exact polymer sequence, writes copied structures
to `disordered_pdb/apo/{chain_type}/{source}/`, and writes
`disordered_pdb/sequences/rcsb_apo_mapping.msgpack` for lookup construction.
If `rcsb-train/apo_tok_lmdb/protein/{source}.lmdb` exists, protein apo tokens
are copied directly to `disordered_pdb/apo_tok_lmdb/protein/{source}.lmdb`.
Matched RCSB prior stacks are also copied to
`disordered_pdb/prior_lmdb/{protein,dna,rna}.lmdb`.

```bash
.venv/bin/python scripts/process/disordered_pdb/b1_fetch_rcsb_train_apo.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3 \
  --overwrite
```

Default sources are `esmfold`, `prot_sampler_*`, `rna_sampler_seed1_step100`,
and `dna_helix`. `afdb` is not used because its residue mapping is more
complicated.

Then build lookup and source-specific apo LMDBs:

```bash
python scripts/process/disordered_pdb/b2_make_lookup.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3

python scripts/process/disordered_pdb/b3_create_apo_lmdb.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3 \
  --num_workers 128 \
  --overwrite

python scripts/process/disordered_pdb/b4_create_prior_lmdb.py \
  --data_dir /cache/wykim_lab/icl_shwan/kfold_data/v260701_af3 \
  --overwrite
```

## Apo Tokens

Protein apo tokens are fetched by `b1_fetch_rcsb_train_apo.py` when
`rcsb-train/apo_tok_lmdb/protein/{source}.lmdb` exists. No local tokenization
script is kept in this directory.
