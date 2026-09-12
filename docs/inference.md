# K-Fold inference

K-Fold accepts YAML or JSON files describing proteins, DNA, RNA, and ligands.
Inference requires a CUDA GPU; see the [installation instructions](../README.md#installation) for setup.
No MSA search or sequence database is required.

- [Running predictions](#running-predictions)
- [Multi-GPU inference](#multi-gpu-inference)
- [Command-line options](#command-line-options)
- [Input format](#input-format)
- [Apo structures and priors](#apo-structures-and-priors)
- [Outputs and confidence](#outputs-and-confidence)
- [Python API](python_api.md)

## Running predictions

Save the [input example](#input-format) as `query.yaml`, then run:

```bash
kfold predict --input query.yaml --out-dir predictions/
```

By default, K-Fold generates missing protein apo structures with AtlasFold and predicts five complex structures with seed 1.
Model weights and the chemical component dictionary (CCD) are downloaded automatically.
Use `--cache-dir` to select a download cache.

From the repository root, the script entry point provides the same interface:

```bash
python run_kfold.py predict --input query.yaml --out-dir predictions/
```

### Multiple queries

Pass a directory to process its immediate `.yaml`, `.yml`, and `.json` files.
Each file contains one query, and query names must be unique:

```bash
kfold predict --input queries/ --out-dir predictions/
```

### Checking and resuming runs

Validate query files and referenced paths before loading models:

```bash
kfold predict --input queries/ --out-dir predictions/ --dry-run
```

A dry run reports pending and completed jobs without downloading models or writing outputs.
Structure contents and CCD availability are checked during inference.

Completed query/seed directories contain `done.txt` and are skipped on subsequent runs.
Incomplete jobs restart from the beginning.
Use `--overwrite` after changing an input, sampling settings, or requested output files for an existing seed.
Matching files are replaced; the output directory and other files are retained.

To add predictions, request new seeds.
Include old and new seeds in the same command to rank their results together: with `--seed 1 2 3`, a completed seed 1 is skipped while seeds 2 and 3 run.

## Multi-GPU inference

Use `--gpu-ids` to distribute queries and seeds across the selected GPUs:

```bash
kfold predict --input queries/ --out-dir predictions/ \
  --seed 1 2 3 --gpu-ids 0 1
```

Each GPU handles independent query/seed jobs sequentially and reuses its loaded models.
A single prediction runs on one GPU.
You can also pass a single query file with multiple seeds.
Without `--gpu-ids`, inference uses GPU 0.
Pass a single ID, such as `--gpu-ids 2`, to select one GPU.
IDs refer to visible CUDA devices; with `CUDA_VISIBLE_DEVICES=2,3`, `--gpu-ids 0 1` selects physical GPUs 2 and 3.

## Command-line options

Run `kfold predict --help` for help at the command line.

| Option | Default | Description |
| --- | --- | --- |
| `-i`, `--input` | Required | Query file or directory of query files. |
| `-o`, `--out-dir` | Required | Root output directory. |
| `--seed` | `1` | One or more unique positive seeds. |
| `--num-apos` | `1` | Generated apo structures per protein entry without supplied structures; 1–5. |
| `--num-samples` | `5` | Complex predictions per query/seed. |
| `--num-recycles` | `10` | Model recycling iterations. |
| `--num-steps` | `100` | Diffusion steps. |
| `--gpu-ids` | `0` | One or more unique, non-negative visible CUDA indices for independent query/seed jobs. |
| `--disable-struct-encoder` | Off | Disable the pretrained protein structure encoder to reduce memory use. |
| `--disable-rna-encoder` | Off | Disable the RNA encoder; use only when queries contain no RNA. |
| `--cache-dir` | Hugging Face default | Download cache for model weights and CCD. |
| `--dry-run` | Off | Validate query files and paths, and report pending jobs. |
| `--overwrite` | Off | Rerun completed query/seed jobs. |
| `--save-confidence` | Off | Save per-atom and token-pair confidence arrays. |
| `--save-embeddings` | Off | Save single and pair embeddings. |
| `--save-distogram` | Off | Compute and save distogram arrays. |
| `--save-trajectory` | Off | Compute and save diffusion trajectories. |

### Sampling

Use `--seed` for independent runs and `--num-samples` for the number of predictions per seed:

```bash
kfold predict --input query.yaml --out-dir predictions/ \
  --seed 1 2 3 --num-samples 5
```

This produces 15 predictions.
Each seed has its own prepared inputs and output directory.
`--num-apos` separately controls how many apo structures are generated per protein entry; see [apo structures and priors](#apo-structures-and-priors).

## Input format

Every query requires a `name` and a non-empty `sequences` list.
Each entry contains one component type: `protein`, `protein_pair`, `dna`, `rna`, or `ligand`.
YAML and JSON use the same schema.

```yaml
name: test
sequences:
  - protein:
      id: [A, B]
      sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL
```

`name` determines the output directory.
It must start with a letter or digit and contain only letters, digits, underscores, dots, and hyphens.
Chain IDs must start with a letter and contain only letters, digits, and underscores.
IDs are case-sensitive and must be unique across the query.

Use `id: A` for one chain or `id: [A, B]` for identical copies.
Every component may include an optional string `description`.
Sequences must be uppercase and contain no whitespace; CCD codes use uppercase letters and digits.
Omit unused optional fields instead of setting them to `null`.
Unknown fields are rejected.

### Proteins

| Field | Required | Description |
| --- | --- | --- |
| `id` | Yes | Chain ID or list of IDs for identical copies. |
| `sequence` | Yes | Amino-acid sequence; `X` represents an unknown residue. |
| `modifications` | No | List of [residue modifications](#residue-modifications). |
| `apo` | No | List of apo structure paths. Generated when omitted. |
| `prior` | No | List of separate prior structure paths; requires `apo`. |

The copies in one entry use the same set of apo and prior structures.
To supply different structures for chains with the same sequence, put them in separate entries.

### Protein pairs

Use `protein_pair` for two proteins whose relative positions are preserved in the starting structure, such as the heavy and light chains of a Fab.
Missing pair structures are generated with AtlasFold-M.

| Field | Required | Description |
| --- | --- | --- |
| `id` | Yes | `[H, L]` for one pair, or `[[H, L], [M, N]]` for two copies. |
| `sequence1`, `sequence2` | Yes | Amino-acid sequences in component order. |
| `modifications1`, `modifications2` | No | Residue modifications for each component. |
| `apo` | No | List of two-chain structure paths. Generated when omitted. |
| `prior` | No | List of separate two-chain prior paths; requires `apo`. |

For `id: [[H, L], [M, N]]`, chains H and M use `sequence1`, and chains L and N use `sequence2`.
The two copies are placed independently.
Supplied structures must contain exactly two non-empty protein chains in `sequence1`, `sequence2` order.

### DNA and RNA

DNA and RNA entries require `id` and `sequence`, and accept optional `modifications`.
DNA uses `A`, `C`, `G`, `T`, and `N`; RNA uses `A`, `C`, `G`, `U`, and `N`.

```yaml
name: nucleic_acids
sequences:
  - dna:
      id: A
      sequence: ACGTACGT
  - rna:
      id: B
      sequence: ACGUACGU
```

### Ligands

Ligands require `id` and exactly one of `ccd` or `smiles`.
Use a CCD code for a known chemical component, including ions, or a SMILES string for a custom ligand.
The following entries can be added to a query's `sequences` list:

```yaml
- ligand:
    id: C
    ccd: ATP
- ligand:
    id: [D, E]
    ccd: MG
- ligand:
    id: F
    smiles: 'CC(=O)Oc1ccccc1C(=O)O'
```

`ccd` also accepts a non-empty list such as `[ATP]`, or multiple CCD codes for a multi-residue component.
CCD codes must exist in the loaded dictionary.

### Residue modifications

Add `modifications` to a protein, DNA, or RNA entry.
Each modification specifies a 1-based `residue_index` within the sequence and the replacement `ccd` code.
For example, this replaces the fourth residue with phosphoserine:

```yaml
name: modified_protein
sequences:
  - protein:
      id: A
      sequence: MKTSA
      modifications:
        - residue_index: 4
          ccd: SEP
```

Each residue index may appear only once.
Protein pairs use the same format in `modifications1` and `modifications2` for their respective sequences.

### Covalent bonds

The optional top-level `bonds` list connects pairs of atoms.
Each atom reference is `[chain_id, residue_index, atom_name]`, with 1-based residue indices:

```yaml
bonds:
  - [[A, 20, NZ], [C, 1, C08]]
```

This connects atom `NZ` of chain A residue 20 to atom `C08` of chain C residue 1.
The chains, residues, and atom names must exist in the query's prepared structure.
Self-bonds and duplicate bonds are rejected.

## Apo structures and priors

K-Fold uses protein apo structures as structural input and prior structures to prepare the starting coordinates for diffusion.
Both are prepared automatically when `apo` is omitted.

### Automatic generation

`protein` entries use AtlasFold; `protein_pair` entries use AtlasFold-M.
For each entry without supplied apo structures, `--num-apos N` runs AtlasFold with N seeds and five candidates per seed.
The highest-ranked candidate from each seed becomes an apo structure, and all candidates become prior structures.

```bash
kfold predict --input query.yaml --out-dir predictions/ \
  --num-apos 3
```

With `--num-apos 3`, each protein entry without supplied structures gets three apo structures and fifteen prior structures.
The number of K-Fold predictions is controlled separately by `--num-samples`.
`--num-apos` must be between 1 and 5 and does not limit the number of supplied structures.

### Supplying structures

Use `apo` to supply structures from an existing prediction, such as AlphaFold2, or an experiment:

```yaml
name: supplied_apo
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL
      apo:
        - structures/protein.pdb
```

`apo` must be a non-empty list of existing file paths.
Relative paths resolve from the query file's directory.
PDB and mmCIF files may contain multiple models; every model is used, in file and model order.
Duplicate paths within each list are rejected.

Supplied apo structures are used directly and also provide the starting structures for diffusion.
To use separate starting structures, optionally add `prior` alongside `apo` using the same list-of-paths format.

For a `protein`, the first subchain of each model is used.
For a `protein_pair`, both chains are used in file order.
Structure sequences are aligned to the query; missing residues are masked, and source insertions without a query position are omitted.

## Outputs and confidence

For a query named `test`, the default command writes:

```text
predictions/test/
├── test_model.cif
├── test_confidence.json
├── test_summary.csv
└── test_seed-1/
    ├── done.txt
    ├── query.json
    ├── apo/
    │   ├── seq-0-apo.pdb
    │   └── seq-0-prior.pdb
    ├── test_seed-1_sample-0_model.cif
    ├── test_seed-1_sample-0_confidence.json
    ├── ...
    ├── test_seed-1_sample-4_model.cif
    └── test_seed-1_sample-4_confidence.json
```

Each seed directory contains all predictions and prepared inputs.
Sample indices start at 0.
`query.json` references the saved apo and prior PDB files using relative paths; move the whole seed directory to keep these references valid.
In `seq-{i}-apo.pdb` and `seq-{i}-prior.pdb`, `i` is the zero-based position of the protein or pair entry in `sequences`.

### Ranking and confidence summaries

After all jobs finish, the CLI copies the highest-ranked structure and its confidence JSON to the query directory.
The summary CSV lists `seed`, `sample`, `ranking_score`, `plddt`, `ptm`, `iptm`, `pde`, and `has_clash`, sorted by descending ranking score.

Confidence JSON files group scores under `complex`, `chains`, and `interfaces`.
Chain summaries contain mean pLDDT, mean PDE, and pTM; interface summaries contain pairwise ipTM.
The main confidence measures are:

A token represents one standard protein, DNA, or RNA residue, or one atom in a ligand or modified residue.

| Metric | Meaning | Scale |
| --- | --- | --- |
| `plddt` | Local confidence, predicted per atom and averaged in summaries. | 0–100; higher is better. |
| `ptm` | Predicted TM-score for overall structure accuracy. | 0–1; higher is better. |
| `iptm` | Interface predicted TM-score for relative chain placement. | 0–1; higher is better. |
| `pae` | Predicted aligned error between tokens, available in raw confidence arrays. | Å; lower is better. |
| `pde` | Predicted distance error between tokens, averaged in summaries. | Å; lower is better. |
| `has_clash` | Whether the predicted complex triggers the inter-chain clash check. | 0 or 1. |

The complex ranking score is `0.8 * iptm + 0.2 * ptm - 100 * has_clash`.
Structure mmCIF files store per-atom pLDDT in the B-factor field.

Ranking includes only the seeds requested in the current command and saved sample indices below `--num-samples`, including completed jobs that were skipped.
Top-level results are refreshed even when no new inference is needed.

### Optional outputs

Structure mmCIF and confidence-summary JSON files are always saved.
Enable additional outputs with these flags:

| Flag | File suffix | Contents |
| --- | --- | --- |
| `--save-confidence` | `_sample-{i}_confidence.npz` | Per-atom `plddt` and token-pair `pae` and `pde` arrays. |
| `--save-embeddings` | `_embeddings.npz` | Unpadded `s_inputs`, `s_lm`, and `z` representations, shared across samples. |
| `--save-distogram` | `_distogram.npz` | Shared `logits`, `bin_edges`, `asym_ids`, and `res_ids`. |
| `--save-trajectory` | `_sample-{i}_trajectory.cif` | Diffusion trajectory for each sample. |

All filenames begin with `<name>_seed-<seed>`.
If the best prediction has a saved confidence NPZ, it is also copied to `<name>_confidence.npz` in the query directory.
Embeddings and distogram logits are FP16; confidence arrays, coordinates, trajectories, and distance bin boundaries are FP32.
Distogram indices are INT32.
