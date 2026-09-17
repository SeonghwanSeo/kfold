# K-Fold inference

K-Fold accepts YAML or JSON files describing proteins, DNA, RNA, and ligands.
Inference requires a CUDA GPU.
No MSA search or sequence database is required.

- [Installation](#installation)
- [Running predictions](#running-predictions)
- [Selecting stages](#selecting-stages)
- [Multi-GPU inference](#multi-gpu-inference)
- [Command-line options](#command-line-options)
- [Input format](#input-format)
- [Apo structures](#apo-structures)
- [Outputs and confidence](#outputs-and-confidence)
- [Python API](python_api.md)

## Installation

K-Fold requires Python 3.11 or later.

Install from PyPI:

```bash
pip install 'kfold[cuequiv]'
```

Or install from source:

```bash
git clone https://github.com/SeonghwanSeo/kfold.git
cd kfold
pip install '.[cuequiv]'
```

The `cuequiv` extra installs cuEquivariance kernels for faster inference on NVIDIA GPUs.

## Running predictions

Save the [input example](#input-format) as `query.yaml`, then run:

```bash
kfold --input query.yaml --out-dir predictions/
```

By default, K-Fold generates protein apo structures for all selected queries with [AtlasFold](https://github.com/SeonghwanSeo/atlasfold), then predicts five complex structures per query with seed 1.
You can also provide apo structures from experiments or other prediction tools.
Model weights and the chemical component dictionary (CCD) are downloaded automatically.
Use `--cache-dir` to select a download cache.

From the repository root, the script entry point provides the same interface:

```bash
python run_kfold.py --input query.yaml --out-dir predictions/
```

### Multiple queries

Pass a directory to process its immediate `.yaml`, `.yml`, and `.json` files.
Each file contains one query, and query names must be unique:

```bash
kfold --input queries/ --out-dir predictions/
```

### Checking and resuming runs

Validate query files and referenced paths before loading models:

```bash
kfold --input queries/ --out-dir predictions/ --dry-run
```

A dry run checks query files, structure paths, and CCD codes, and reports pending and completed jobs without loading models or writing prediction outputs.
The CCD is downloaded if needed; structure contents are checked during inference.

Rerun the same command to continue an interrupted run.
Completed predictions and apo structures are reused; unfinished predictions are rerun.

Use `--overwrite` to replace existing results after changing inputs, sampling settings, or output options.

To add predictions and rank them with existing results, include both old and new seeds: with `--seeds 1 2 3`, a completed seed 1 is skipped while seeds 2 and 3 run.

## Selecting stages

The default `--stage all` prepares apo structures and predicts complexes.
Use `--stage apo` to prepare apos only, or `--stage complex` to predict complexes from prepared apos:

```bash
kfold --stage apo --input query.yaml --out-dir predictions/ --seeds 1
kfold --stage complex --input query.yaml --out-dir predictions/ --seeds 1
```

Use the same input, output directory, and seeds for both commands.
With `--share-apo-seeds`, pass the same shared apo seeds to both stages; the complex stage can use any inference seeds specified by `--seeds`.
`--stage complex` reads the prepared queries and structures from `--out-dir` for the queries selected by `--input`.

## Multi-GPU inference

Use `--gpu-ids` to distribute queries and seeds across the selected GPUs:

```bash
kfold --input queries/ --out-dir predictions/ --seeds 1 2 3 --gpu-ids 0 1
```

Both stages use the selected GPUs.
Each prediction runs entirely on one GPU; additional GPUs process other queries or seeds.
You can also pass a single query file with multiple seeds.
Without `--gpu-ids`, inference uses GPU 0.
Pass a single ID, such as `--gpu-ids 2`, to select one GPU.
IDs refer to visible CUDA devices; with `CUDA_VISIBLE_DEVICES=2,3`, `--gpu-ids 0 1` selects physical GPUs 2 and 3.

## Command-line options

Run `kfold --help` for all options.
The default `--stage all` prepares apos and predicts complexes.

| Option | Default | Description |
| --- | --- | --- |
| `-i`, `--input` | Required | Query JSON/YAML file or directory. |
| `-o`, `--out-dir` | Required | Output directory; prepared inputs for `--stage complex`. |
| `--stage` | `all` | Both stages (`all`), apo preparation (`apo`), or prediction from prepared inputs (`complex`). |
| `--seeds` | `1` | One or more unique positive complex inference seeds. |
| `--apo-config` | Built-in defaults | YAML settings for apo batching and AtlasFold sampling. |
| `--num-apos` | `1` | Generated apos per protein entry per inference seed (1–5); excludes `--share-apo-seeds`. |
| `--share-apo-seeds` | Off | Unique positive seeds for apos shared across inference seeds; excludes `--num-apos`. |
| `--num-samples` | `5` | Complex predictions per query/seed. |
| `--num-recycles` | `10` | Model recycling iterations. |
| `--num-steps` | `100` | Diffusion steps. |
| `--conditioning` | `prior_only` | Assembly intermediate reuse: ECSI only; chainwise trunk re-encoding with `prior_and_trunk`; or joint protein-complex structure representation with `prior_and_trunk_multichain`. |
| `--provided-intermediates` | Off | Read atom-mapped `QUERY/STAGE.npz` structures instead of predicting non-final assembly stages. |
| `--gpu-ids` | `0` | Unique non-negative visible CUDA device IDs. |
| `--disable-struct-encoder` | Off | Disable the protein structure encoder to save memory. |
| `--disable-rna-encoder` | Off | Disable the RNA encoder; only for queries without RNA. |
| `--cpu-offload` | Off | Offload encoders to CPU: less GPU memory, more CPU memory and transfer time. |
| `--cache-dir` | Hugging Face default | Download cache for model weights and CCD. |
| `--dry-run` | Off | Validate query files and paths, and report pending jobs. |
| `--overwrite` | Off | Rerun completed query/seed jobs. |
| `--save-confidence` | Off | Save per-atom and token-pair confidence arrays. |
| `--save-embeddings` | Off | Save single and pair embeddings. |
| `--save-distogram` | Off | Compute and save distogram arrays. |
| `--save-trajectory` | Off | Compute and save diffusion trajectories. |

### Sampling

Use `--seeds` for independent runs and `--num-samples` for the number of predictions per seed:

```bash
kfold --input query.yaml --out-dir predictions/ --seeds 1 2 --num-samples 5
```

### Apo configuration

Use `--apo-config apo.yaml` to customize AtlasFold settings.
The default configuration is shown below.

```yaml
max_tokens_per_batch: 1024
monomer:
  num_recycles: 4
  mlm_prob: 0.15
  num_samples: 5
  num_steps: null # AtlasFold's dynamic scheduling
multimer:
  num_recycles: 4
  mlm_prob: 0.15
  num_samples: 5
  num_steps: 100
```

`max_tokens_per_batch` controls the trade-off between throughput and memory use ([see AtlasFold](https://github.com/SeonghwanSeo/atlasfold/tree/main#performance-and-gpu-memory)).

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

### Sequential assembly

An optional `assembly` plan predicts one or more partial complexes before the
full system. Each stage selects confidence top-1 across all requested seeds and
samples, then uses that structure in the next stage:

```yaml
assembly:
  stages:
    - id: bind_antibody
      chains: [H, L]
  selection: confidence_top1
```

The full-system `final` stage is appended automatically. Existing
`protein_pair` objects cannot be split by a stage. Protein and protein-pair apo
structures remain optional: the normal apo stage generates missing inputs with
AtlasFold or AtlasFold-Multimer before sequential K-Fold inference.

`--conditioning prior_only` carries the selected structure into the ECSI
initialization. `--conditioning prior_and_trunk` also replaces compatible apo
coordinates and regenerates protein structure tokens before rerunning the full
trunk. `--conditioning prior_and_trunk_multichain` additionally tokenizes all
protein chains in each selected object in their shared coordinate frame and
allows cross-chain attention only in the protein structure encoder. All seeds
for one assembly query run on one GPU so selection is global
for each stage. See [Sequential complex priors](sequential_assembly.md) for the
artifact layout and exact conditioning behavior.

### Proteins

| Field | Required | Description |
| --- | --- | --- |
| `id` | Yes | Chain ID or list of IDs for identical copies. |
| `sequence` | Yes | Amino-acid sequence; `X` represents an unknown residue. |
| `modifications` | No | List of [residue modifications](#residue-modifications). |
| `apo` | No | Optional apo structure paths to use instead of automatic generation. |
| `prior` | No | List of separate prior structure paths; requires `apo`. |

The copies in one entry use the same set of apo and prior structures.
To provide different structures for chains with the same sequence, put them in separate entries.

### Protein pairs

Use `protein_pair` for two proteins whose relative positions are preserved in the starting structure, such as the heavy and light chains of a Fab.
Pair structures are generated automatically with AtlasFold-Multimer, or you can provide your own.

| Field | Required | Description |
| --- | --- | --- |
| `id` | Yes | `[H, L]` for one pair, or `[[H, L], [M, N]]` for two copies. |
| `sequence1`, `sequence2` | Yes | Amino-acid sequences in component order. |
| `modifications1`, `modifications2` | No | Residue modifications for each component. |
| `apo` | No | Optional two-chain structure paths to use instead of automatic generation. |
| `prior` | No | List of separate two-chain prior paths; requires `apo`. |

For `id: [[H, L], [M, N]]`, chains H and M use `sequence1`, and chains L and N use `sequence2`.
The two copies are placed independently.
Provided structures must contain exactly two non-empty protein chains in `sequence1`, `sequence2` order.

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

## Apo structures

K-Fold uses apo structures as starting structures for proteins.
These are generated automatically, or you can provide your own.

### Automatic generation

K-Fold generates structures with AtlasFold for individual proteins and AtlasFold-Multimer for protein pairs.
By default, apo structures are generated separately for each complex inference seed.
Use `--num-apos` to choose how many apo structures to generate for each protein or pair per inference seed (1–5; default: 1).

With `--num-apos N`, AtlasFold runs with N distinct seeds for each complex inference seed, generating five predictions per AtlasFold seed by default.
The highest-ranked prediction from each seed provides an apo structure used to condition K-Fold, while all predictions across these seeds form the prior ensemble from which starting coordinates are sampled for the diffusion bridge.

Use `--share-apo-seeds` to generate one apo ensemble per target and reuse its apo and prior structures across all complex inference seeds.
Reusing the same ensemble avoids repeating apo generation for every inference seed.
This is particularly useful for relatively rigid apo structures or runs with many inference seeds.

```bash
kfold --input query.yaml --out-dir predictions/ --seeds 1 2 3 4 5 --share-apo-seeds 7 11 42
```

This generates apos using AtlasFold seeds 7, 11, and 42, then reuses them for complex inference seeds 1 through 5.
The number of shared seeds determines the apo count for each automatically generated protein entry.
Provide at least one unique positive integer after `--share-apo-seeds`; it cannot be combined with `--num-apos`.
Additional inference seeds can reuse the prepared shared ensemble without further apo generation.
Use `--overwrite` with `--stage apo` or `--stage all` when changing the shared seeds or switching between shared and per-inference-seed apo generation.

### Providing apo structures

Use `apo` to provide experimental structures or predictions from tools such as AlphaFold2:

```yaml
name: provided_apo
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL
      apo:
        - structures/protein.pdb
```

List one or more PDB or mmCIF files under `apo`.
Relative paths are resolved from the query file's directory.
If a file contains multiple models, all are used.
Provided structures replace automatic generation for that protein or pair.

## Outputs and confidence

For a query named `test`, the default command writes:

```text
predictions/test/
├── test_model.cif
├── test_confidence.json
├── test_summary.csv
├── kfold_settings.json
└── test_seed-1/
    ├── done.txt
    ├── query.json
    ├── kfold_settings.json
    ├── apo_setting.json
    ├── apo/
    │   ├── monomer-1.done
    │   ├── monomer-1_apo.pdb
    │   └── monomer-1_prior.pdb
    ├── test_seed-1_sample-0_model.cif
    ├── test_seed-1_sample-0_confidence.json
    ├── ...
    ├── test_seed-1_sample-4_model.cif
    └── test_seed-1_sample-4_confidence.json
```

By default, each seed directory contains its predictions and prepared inputs.
With `--share-apo-seeds`, `query.json`, `apo_setting.json`, and `apo/` are saved directly under `predictions/test/`; predictions remain in their respective seed directories.
Sample indices start at 0.
`query.json` references the saved apo and prior PDB files using relative paths; move the whole seed directory, or the whole target directory when sharing apos, to keep these references valid.
Apo files use `monomer-{i}_apo.pdb` and `monomer-{i}_prior.pdb`, or `multimer-{i}_apo.pdb` and `multimer-{i}_prior.pdb` for protein pairs.
The number i is the 1-based position in the full `sequences` list, including non-protein entries.

### Ranking and confidence summaries

After all jobs finish, the CLI copies the highest-ranked structure and its confidence JSON to the query directory.
The summary CSV lists `seed`, `sample`, `ranking_score`, `plddt`, `ptm`, `iptm`, `pde`, and `has_clash`, sorted by descending ranking score.

Confidence JSON files group scores under `complex`, `chains`, and `interfaces`.
Chain summaries contain mean pLDDT, mean PDE, and pTM; interface summaries contain pairwise ipTM.
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

Ranking covers the requested seeds and sample count, including reused results.

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
