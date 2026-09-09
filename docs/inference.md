# K-Fold Inference Guide

K-Fold accepts YAML or JSON queries describing proteins, DNA, RNA and ligands.
Prepare apo structures, then predict on a CUDA GPU. Both commands accept a query
file or directory through `--input` and save results to `--out-dir`.

## Input format

Save a query as `query.yaml`:

```yaml
name: example
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL
```

See [examples](../examples/README.md) for antibody–antigen, protein–RNA and
protein–ligand inputs. YAML and JSON use the same fields. Give each query a unique
`name`; if omitted, the filename stem is used. Chain IDs must be unique across
the query.

### Proteins

Use `protein` entries under `sequences`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | Chain ID. A list such as `[A, B]` creates copies. |
| `sequence` | `str` | Amino-acid sequence. |
| `modifications` | `dict[int, str]` | Optional 1-based residue index to CCD code mapping. |
| `apo` | `str` or `list[str]` | Apo structure paths; required by `predict`, optional for `prepare`. |
| `prior` | `list[str]` | Optional separate prior ensemble; defaults to the supplied apo structures. |

To use your own apo structures, add `apo: structures/protein.pdb` or a list of
paths to the protein entry. Paths may be absolute or relative to the working
directory or query file.

### Protein pairs

Use `multimer_sequences` for two proteins whose apo structure contains both
chains, such as a Fab heavy/light pair:

```yaml
multimer_sequences:
  - protein:
      id: H:L
      sequence: EVQLVESGG:DIQMTQSP
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | Pair IDs such as `H:L`; `[H:L, M:N]` creates two copies. |
| `sequence` | `str` | Two sequences separated by `:`. |
| `modifications1`, `modifications2` | `dict[int, str]` | Optional residue modifications for each component. |
| `apo` | `str` or `list[str]` | Two-chain structure paths; required by `predict`, optional for `prepare`. |
| `prior` | `list[str]` | Optional separate two-chain prior ensemble; defaults to the supplied apo structures. |

Each supplied structure must contain exactly two protein chains in the same
order as the input sequences. Ordinary proteins and pairs can appear together
in `sequences` and `multimer_sequences`.

### DNA and RNA

Use `dna` or `rna` entries under `sequences`. Neither accepts `apo` or `prior`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | One or more chain IDs. |
| `sequence` | `str` | DNA uses A/C/G/T; RNA uses A/C/G/U. |
| `modifications` | `dict[int, str]` | Optional 1-based residue index to CCD code mapping. |

### Ligands

Use `ligand` entries under `sequences`, specifying exactly one of `smiles` or `ccd`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | One or more chain IDs. |
| `smiles` | `str` | A SMILES representation. |
| `ccd` | `str` or `list[str]` | One CCD code or a multi-residue CCD sequence. |

### Covalent bonds

The optional top-level `bonds` field connects atoms using
`[chain_id, one_based_residue_index, atom_name]`:

```yaml
bonds:
  - [[A, 20, NZ], [C, 1, C08]]
```

## Preparation

Preparation uses AtlasFold for ordinary proteins and AtlasFold-M for protein
pairs. Existing apo and prior inputs are preserved.
If every protein already has apo inputs, go directly to [Prediction](#prediction).

```bash
kfold prepare --input query.yaml --out-dir prepared/ --seed 1 2 3 4 5
```

For each protein entry being generated, the example produces 25 PDB structures.
The highest-confidence structure from each seed becomes an apo candidate
(5 total); all samples become prior candidates (25 total).

The prepared YAML queries and generated PDB structures are saved in `prepared/`.

AtlasFold weights download automatically.

| Option | Default | Description |
| :--- | :--- | :--- |
| `--seed` | `1` | One or more AtlasFold seeds. |
| `--num-samples` | `5` | Structures generated per seed. |
| `--num-gpus` | `1` | Number of GPUs to use. |
| `--cache-dir` | Hugging Face default | Cache for downloaded models. |
| `--overwrite` | `False` | Regenerate predicted structures from the original query. |

See `kfold prepare --help` for all options.

## Prediction

```bash
kfold predict --input prepared/ --out-dir results/ --seed 1 2 3 4 5
```

With five seeds and five samples per seed, this produces 25 predictions per query.

Model weights/config and CCD download automatically from `SeonghwanSeo/kfold`
and `SeonghwanSeo/kfold-assets`.

To use multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 kfold predict \
  --input prepared/ --out-dir results/ --seed 1 2 3 4 --num-gpus 4
```

Prediction options:

| Option | Default | Description |
| :--- | :--- | :--- |
| `--seed` | `1` | One or more K-Fold seeds. |
| `--num-apo` | `3` | Maximum apo structures used per protein entry. |
| `--num-samples` | `5` | Diffusion samples per query and seed. |
| `--num-recycles` | `10` | Trunk recycle count. |
| `--num-steps` | `100` | Diffusion step count. |
| `--num-gpus` | `1` | Number of GPUs to use. |
| `--cache-dir` | Hugging Face default | Cache for downloaded models and CCD. |
| `--dry-run` | `False` | Validate prepared inputs on CPU without prediction models. |
| `--overwrite` | `False` | Recompute completed predictions. |
| `--save-confidence` | `False` | Save raw confidence arrays. |
| `--save-distogram` | `False` | Save distogram logits and bin edges. |
| `--save-trajectory` | `False` | Save the diffusion trajectory. |

Optional `--weight` and `--config` override the release weights and configuration.
See `kfold predict --help` for all options. To use the wrapper, replace `kfold`
with `python run_kfold.py` from the repository root.

### Output

Results are saved in `results/`. Each sample produces an mmCIF structure and a
confidence-summary JSON. Use `complex.ranking_score` in the JSON to compare
predictions for the same query; higher is better. Optional confidence arrays
and distograms use NPZ; trajectories use PDB.

## Python API

```python
from kfold import KFoldRunner

runner = KFoldRunner(device="cuda:0")
paths = runner.prepare("query.yaml", "prepared/", seeds=[1])
for path in paths:
    for query in runner.read_queries(path, seeds=[1]):
        result = runner.fold(query, num_samples=5)
        result.save("results/")
```

`prepare` and `read_queries` accept YAML/JSON paths. `fold` takes one parsed
`Query` and returns a `FoldingResult`.
