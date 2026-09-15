# RCSB preprocessing

Create `rcsb-train` and `rcsb-val` from RCSB mmCIF files and `components.cif`.
Run the commands from the repository root in your processing jobs.

## Paths

```bash
RCSB_MMCIF_DIR="/absolute/path/to/raw_rcsb_mmcif"
COMPONENTS_CIF="/absolute/path/to/components.cif"
DATA_DIR="/absolute/path/to/processed_kfold_data"
SABDAB_CSV="/absolute/path/to/sabdab_summary.csv"  # Optional multimer input
```

`DATA_DIR` is the output parent for both splits. Keep `components.cif` outside
`RCSB_MMCIF_DIR` and use a fresh output directory for a new build.

Preparation requires K-Fold, `pdbeccdutils`, and MMseqs2. Prediction requires
AtlasFold and a GPU; tokenization also requires a GPU. Adjust `--num_workers`
to your CPU allocation.

## 1. Prepare CCD

```bash
python scripts/process/rcsb/a_process_ccd.py \
  --cif_path "$COMPONENTS_CIF" --out_path "$DATA_DIR/ccd-train.pkl" \
  --train --date_cutoff 2021-09-30 --num_workers 32
```

## 2. Extract sequences

This creates sequence files for both splits in one pass.

```bash
python scripts/process/rcsb/b_extract_all_sequence.py \
  --cif_dir "$RCSB_MMCIF_DIR" --data_dir "$DATA_DIR" --num_workers 32
```

The current split dates are train through **2021-09-30** and val from
**2021-10-01 to 2023-01-12**. To change them, edit `SPLITS` in the sequence
extraction and CIF processing scripts.

## 3. Process structures

```bash
for split in train val; do
  python scripts/process/rcsb/c_process_cifs.py \
    --cif_dir "$RCSB_MMCIF_DIR" --data_dir "$DATA_DIR" --split "$split" \
    --ccd_path "$DATA_DIR/ccd-train.pkl" --num_workers 32
done
```

## 4. Build train

```bash
python scripts/process/rcsb/d1_cluster.py \
  --data_dir "$DATA_DIR" --mmseqs mmseqs --num_workers 32

python scripts/process/rcsb/d2_construct_training_set.py --data_dir "$DATA_DIR"
```

## 5. Build val

Validation selection uses the training sequences and queries the RCSB API for
ligand quality scores, so this step requires network access.

```bash
python scripts/process/rcsb/e1_get_val_ids.py \
  --data_dir "$DATA_DIR" --ccd_path "$DATA_DIR/ccd-train.pkl" \
  --mmseqs mmseqs --num_workers 32

python scripts/process/rcsb/e2_construct_val_set.py --data_dir "$DATA_DIR"
```

Validation requires protein priors. Use `--max_length 2048` for both validation
apo prediction and LMDB construction below.

## 6. Predict apo structures

### Monomer

Reads `sequences/unique_protein_sequences.fasta` directly, without manifest
filtering. Results use the FASTA ID (the first word after `>`) as the directory
name. Repeated sequences use their first ID. LMDB construction maps entries to
these IDs by sequence.

```bash
python scripts/process/rcsb/f1_predict_apo.py \
  --data_dir "$DATA_DIR" --split train --seeds 1 --max_length 1280

python scripts/process/rcsb/f1_predict_apo.py \
  --data_dir "$DATA_DIR" --split val --seeds 1 --max_length 2048
```

The monomer length limit is **1280 residues for train** and **2048 for val** in
these commands; longer inputs are skipped.
Seeds default to `[1]`. For multiple seeds, use `--seeds 1 2 3`.

### Multimer (optional, train)

Export heavy/light groups from a SAbDab CSV with `PDB,Hchain,Lchain` columns:

```bash
python scripts/process/rcsb/f2_extract_sabdab_pairs.py \
  --data_dir "$DATA_DIR" --split train --sabdab_path "$SABDAB_CSV" \
  --out_path "$DATA_DIR/rcsb-train/sequences/multimer_groups.csv"
```

You can instead supply your own group CSV and skip the export:

```csv
group_id,entry_id,asym_ids
my-antibody,1abc,7;11
```

Use the prepared dataset's integer `asym_id` values. Each group must contain two
protein chains with distinct entities; groups must not overlap chains.

```bash
python scripts/process/rcsb/f3_predict_apo_multimer.py \
  --data_dir "$DATA_DIR" --split train \
  --groups_csv "$DATA_DIR/rcsb-train/sequences/multimer_groups.csv" \
  --seeds 1 --num_samples 5
```

The combined chain length must be at most 1280 residues. The current validation
loader uses monomer inputs, so val does not need multimer predictions.

## 7. Build apo/prior LMDBs

Use the same seeds, sample count, and monomer `--max_length` as prediction.
If you skipped multimer
prediction, omit `--groups_csv` from the train command.

```bash
python scripts/process/rcsb/g_create_apo_prior_lmdb.py \
  --data_dir "$DATA_DIR" --split train --max_length 1280 \
  --groups_csv "$DATA_DIR/rcsb-train/sequences/multimer_groups.csv" \
  --seeds 1 --num_samples 5 --num_workers 32 --map_size_gb 128

python scripts/process/rcsb/g_create_apo_prior_lmdb.py \
  --data_dir "$DATA_DIR" --split val --max_length 2048 \
  --seeds 1 --num_samples 5 --num_workers 32 --map_size_gb 128
```

Apo stores contain rank 1; prior stores contain all ranks, grouped by seed.
`--overwrite` rebuilds the apo/prior stores and lookups. Regenerate tokens afterward.

## 8. Tokenize

Tokenizer weights are downloaded from Hugging Face automatically. Optionally use
`--cache_dir` to choose the cache location.

```bash
for split in train val; do
  python scripts/process/rcsb/h1_tokenize_apo_monomer.py \
    --data_dir "$DATA_DIR" --split "$split" --chunk 0 --num_chunk 1
done

# Skip this command if multimer prediction was skipped.
python scripts/process/rcsb/h2_tokenize_apo_multimer.py \
  --data_dir "$DATA_DIR" --split train --chunk 0 --num_chunk 1
```

After tokenization finishes, combine the shards:

```bash
for split in train val; do
  python scripts/process/rcsb/h3_combine_apo_token_lmdb.py \
    --data_dir "$DATA_DIR" --split "$split" --num_chunk 1
done
```

For parallel prediction or tokenization jobs, assign each job a different
`--chunk` with the same `--num_chunk`. Pass the tokenization shard count to the
combiner and wait for all shards to finish.

## Output layout

```text
$DATA_DIR/
  ccd-train.pkl
  rcsb-train/
    npz/
    sequences/
      unique_protein_sequences.fasta
      all_sequences.fasta
      multimer_groups.csv             # Optional
    manifest.json
    manifest.msgpack
    structure.lmdb/
    apo_lookup.msgpack
    apo_multimer_lookup.msgpack
    apo/
      protein/atlasfold-seed1/
      protein-multimer/atlasfold-m-seed1/
    apo_lmdb/
      protein/atlasfold-seed1.lmdb/
      protein-multimer/atlasfold-m-seed1.lmdb/
    prior_lmdb/
      protein/atlasfold-seed1.lmdb/
      protein-multimer/atlasfold-m-seed1.lmdb/
    apo_tok_lmdb/
      protein/atlasfold-seed1.lmdb/
      protein-multimer/atlasfold-m-seed1.lmdb/
  rcsb-val/
    validation_ids.txt
    # Same layout, without multimer predictions/stores.
```

Keep `all_sequences.fasta`: validation selection and apo entity mapping use it.
