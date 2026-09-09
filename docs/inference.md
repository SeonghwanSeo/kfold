# KFold Inference Guide

KFold accepts a YAML or JSON description of a biomolecular complex and predicts
its three-dimensional structure. Use `kfold prepare` to generate missing protein
apo structures, or provide your own apo inputs. A two-chain protein source, such
as an antibody Fab heavy/light pair, is represented by `multimer_sequences`.

## Running inference

Save a query as `query.yaml`:

```yaml
name: example
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL
```

Prepare missing apo inputs, then run KFold on a CUDA GPU:

```bash
kfold prepare --input query.yaml --out-dir prepared/ --seed 1 2 3 4 5
kfold predict --input prepared/ --out-dir results/ --seed 1 2 3 4 5
```

Both commands default to five samples per seed. This example prepares 25 protein
structures and produces 25 KFold predictions. `--input` also accepts a directory
of YAML/JSON files.

You can also use `python run_kfold.py` in an environment where KFold is installed.

Run wrapper commands from the repository root. For example:

```bash
python run_kfold.py predict --help
```

The wrapper calls the KFold CLI directly and accepts the same options.

If your queries already contain protein apo inputs, pass them directly to
`predict`. Prediction does not generate apo structures or copy input structures.

Models and their matching configuration are downloaded from `SeonghwanSeo/kfold`;
CCD is downloaded from `SeonghwanSeo/kfold-assets`. Use `--cache-dir` to choose
the Hugging Face cache for models, encoders, tokenizers and CCD.

To predict on multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 kfold predict \
  --input prepared/ \
  --out-dir results/ \
  --seed 1 2 3 4 \
  --num-gpus 4
```

Prediction distributes `(query, seed)` jobs across GPUs, so one query with
several seeds can use several GPUs. Completed jobs are skipped before assignment.
Each worker loads its own model; jobs are balanced by count, not sequence length.

Prediction options:

| Option | Default | Description |
| :--- | :--- | :--- |
| `--seed` | `1` | One or more KFold seeds. |
| `--num-apo` | `3` | Maximum apo structures used per protein entry. |
| `--num-samples` | `5` | Diffusion samples per query and seed. |
| `--num-recycles` | `10` | Trunk recycle count. |
| `--num-steps` | `100` | Diffusion step count. |
| `--num-gpus` | `1` | Number of GPU workers. |
| `--dry-run` | `False` | Validate prepared inputs on CPU without prediction models. |
| `--overwrite` | `False` | Recompute completed predictions. |
| `--save-confidence` | `False` | Save raw confidence arrays. |
| `--save-distogram` | `False` | Save distogram logits and bin edges. |
| `--save-trajectory` | `False` | Save the diffusion trajectory. |

Optional `--weight` and `--config` override the released weights and YAML
configuration independently. See `kfold predict --help` for all options.

## Input format

The top-level input contains `sequences`, `multimer_sequences`, or both. Every
physical chain ID must be unique across both sections. Give each query a unique
`name`; if omitted, the input filename stem is used.

Omit `apo` and `prior` when asking `prepare` to generate structures. To supply
your own structures, add paths as in this example:

```yaml
name: Example_Complex
sequences:
  - protein:
      id: A
      sequence: MKTAYIAK
      apo:
        - structures/protein_a_apo_1.pdb
        - structures/protein_a_apo_2.pdb
      prior:
        - structures/protein_a_prior_1.pdb
        - structures/protein_a_prior_2.pdb
  - dna:
      id: B
      sequence: ACGTAA
  - ligand:
      id: C
      ccd: MOV

multimer_sequences:
  - protein:
      id: H:L
      sequence: EVQLVESGG:DIQMTQSP
      apo:
        - structures/fab_apo_1.pdb
        - structures/fab_apo_2.pdb
      prior:
        - structures/fab_prior_1.pdb
        - structures/fab_prior_2.pdb
```

An `id` may be a string or a list. Lists create copies of the same entity:

```yaml
sequences:
  - protein:
      id: [A, B]
      sequence: MKTAYIAK
      apo: monomer.pdb

multimer_sequences:
  - protein:
      id: [H:L, M:N]
      sequence: EVQLVESGG:DIQMTQSP
      apo: fab.pdb
```

The second example creates two physical copies of the same two-entity pair:
`H:L` and `M:N`.

### Protein monomers

Protein entries under `sequences` use the following fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | One or more physical chain IDs. |
| `sequence` | `str` | Amino-acid sequence. |
| `modifications` | `dict[int, str]` | Optional 1-based residue index to CCD code mapping. |
| `apo` | `str` or `list[str]` | Monomer apo structure(s); required by `predict`, optional for `prepare`. |
| `prior` | `list[str]` | Optional custom monomer prior ensemble. |

Custom structures are aligned to the input sequence. Missing residues are masked
and sequence mismatches produce an alignment warning.

### Protein multimers

`multimer_sequences` currently supports exactly two protein components per
entry. It is intended for a shared-frame custom source such as an antibody Fab,
but it is not antibody-specific.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | Pair IDs in `chain1:chain2` format. |
| `sequence` | `str` | Two sequences in `sequence1:sequence2` format. |
| `modifications1` | `dict[int, str]` | Optional modifications for component 1. |
| `modifications2` | `dict[int, str]` | Optional modifications for component 2. |
| `apo` | `str` or `list[str]` | Two-chain apo structure(s); required by `predict`, optional for `prepare`. Each file must contain both components in one shared frame. |
| `prior` | `list[str]` | Optional custom two-chain prior ensemble. |

`--num-apo` defaults to **3** and limits the apo structures used per protein entry.
If more are provided, candidates are sampled without replacement for each KFold
seed. An entry with fewer candidates uses all of them. A single path is equivalent
to a one-element list.

Each custom multimer file must contain exactly two non-empty protein chains in
the same order as the two input sequences. Each component is aligned
to its corresponding input sequence. The pair remains in one shared rigid frame during prior
sampling.

### Apo and prior resolution

Protein monomers and protein multimers follow one source-resolution contract:

| Input | Resolved apo | Resolved priors |
| :--- | :--- | :--- |
| `apo` and `prior` | `apo` | `prior` |
| `apo` only | `apo` | `apo` |

`predict` requires apo inputs for every protein entry. Use `prepare` to fill
missing apo inputs first.

For each configured prior sample, one candidate structure is selected randomly
for each physical chain. Chains in the same rigid group, including the two
components of a multimer pair, select the same candidate index. `predict --num-samples`
sets both the number of sampled diffusion priors and the number of output samples
per query and seed. It does not change the candidate files listed in the input.

Structure paths may be absolute, relative to the current working directory, or
relative to the input YAML/JSON file.

### DNA and RNA

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | One or more physical chain IDs. |
| `sequence` | `str` | DNA uses A/C/G/T; RNA uses A/C/G/U. |
| `modifications` | `dict[int, str]` | Optional 1-based residue index to CCD code mapping. |

DNA and RNA entries do not accept `apo` or `prior` fields. Their apo features
remain unset, and their prior coordinates are initialized by the existing prior
sampler.

### Ligands

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | One or more physical chain IDs. |
| `smiles` | `str` | A SMILES representation. |
| `ccd` | `str` or `list[str]` | One CCD code or a multi-residue CCD sequence. |

Specify exactly one of `smiles` and `ccd`.

### Covalent bonds

Use the optional top-level `bonds` field to specify covalent connections:

```yaml
bonds:
  - [[A, 20, NZ], [C, 1, C08]]
```

Each atom reference uses
`[chain_id, one_based_residue_index, atom_name]`. Contact and distance
constraints are not supported.

## Apo preparation

`prepare` uses AtlasFold for ordinary proteins and AtlasFold-M for
`multimer_sequences` protein pairs. Its `--seed` defaults to `1`, and
`--num-samples` defaults to `5` per seed. With seeds 1–5, it produces **5 apo
candidates and 25 prior candidates** per protein entry:

- Apo: the highest-confidence structure from each seed.
- Prior: all structures, ordered by seed and then confidence rank.

Monomers are ranked by mean pLDDT; multimers by complex ranking score.
Preparation uses four recycles for both models. AtlasFold-M uses 100 diffusion
steps; AtlasFold uses its length-dependent sampling defaults.

Prepared query YAML files are saved directly in `--out-dir`, with generated
**PDB** structures under `apo/`. Existing apo inputs are preserved, and existing
priors are kept even when apo is generated. Referenced files are not copied;
their paths are made absolute. Source queries are unchanged.

Use the same seed list in both commands to match preparation and prediction
seeds, or choose them independently. With `--num-gpus`, preparation distributes
query files; all seeds of a single query stay on one GPU.

Preparation skips each seed with a `done.txt` marker, written only after its
outputs are saved. Input settings and result files are not rechecked; use
`--overwrite` with the original query when changing generation inputs or settings.
It does not replace apo paths already supplied in a query. Fully prepared queries
require no GPU or model loading. See `kfold prepare --help` for all options.

## Python API

```python
from kfold import KFoldRunner

runner = KFoldRunner(device="cuda:0", cache_dir="/data/huggingface")
seeds = [1]
paths = runner.prepare("queries/", "prepared/", seeds=seeds)
runner.release_apo_models()
for path in paths:
    queries = runner.read_queries(path, seeds=seeds)
    for query in queries:
        result = runner.fold(query, num_samples=5)
        print(result.coordinates.shape, result.confidence_summary)
        result.save("results/")
```

`runner.fold(query)` predicts one query and returns one `FoldingResult`.
Results contain CPU coordinates, confidence summaries and arrays,
plus optional distogram and trajectory data. Writing files is explicit via `save`.
`prepare_document(mapping, out_dir)` supports in-memory query dictionaries.

Models load on first use. `runner.model`, `runner.atlasfold.model` and
`runner.atlasfold_multimer.model` share the same `runner.atlaslm` instance;
`runner.ccd` is also loaded once. Keep one runner per device/process and reuse it
across preparation and prediction. `release_apo_models()` frees the AtlasFold
models while retaining AtlasLM and CCD; `fold()` also calls it before prediction.
Use the same seed list for preparation and KFold, or provide separate lists.

## Output

Each query gets its own directory containing the stored query description and
predicted mmCIF structures:

```text
prepared/
├── Example_Complex.yaml
└── apo/Example_Complex/sequences-0/seed-1/
    ├── rank_1.pdb       # apo and prior
    ├── rank_1.json      # seed, rank, original sample index and ranking score
    ├── ...
    ├── rank_5.pdb       # prior
    ├── rank_5.json
    └── done.txt

results/
└── Example_Complex/
    ├── Example_Complex_seed-1_query.yaml
    └── Example_Complex_seed-1/
        ├── Example_Complex_seed-1_sample-0.cif
        ├── Example_Complex_seed-1_sample-0_confidences.json
        ├── Example_Complex_seed-1_sample-0_confidences.npz  # with --save-confidence
        └── done.txt
```

Every successful sample writes an mmCIF file and a confidence-summary JSON file.
`--save-confidence` adds pLDDT/PAE/PDE arrays in NPZ format;
`--save-trajectory` adds a trajectory PDB. `--save-distogram` adds one NPZ per
query/seed containing logits, bin edges and token identifiers.

Prediction writes `done.txt` only after all requested outputs for a query/seed
are saved. Subsequent runs check only this marker. Use `--overwrite` or a new
output directory when changing the query, model, sample count or output options.
