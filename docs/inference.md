# K-Fold Inference Guide

K-Fold accepts a YAML or JSON description of a biomolecular complex and predicts
its three-dimensional structure. Protein and RNA inputs currently require a custom
apo structure, a custom prior ensemble, or both. A two-chain protein source, such
as an antibody Fab heavy/light pair, is represented by `multimer_sequences`.

## Input format

The top-level input contains `sequences`, `multimer_sequences`, or both. Every
physical chain ID must be unique across both sections.

```yaml
name: Example_Complex
sequences:
  - protein:
      id: A
      sequence: MKTAYIAK
      apo: structures/protein_a_apo.pdb
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
      apo: structures/fab_apo.pdb
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
| `apo` | `str` | Optional custom monomer apo structure. |
| `prior` | `list[str]` | Optional custom monomer prior ensemble. |

Custom structures are globally aligned to the input sequence. Source insertions
are ignored, and input residues missing from the source receive `NaN`
coordinates and are excluded from apo structure-token mapping. A warning reports
the alignment summary whenever the sequences differ.

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
| `apo` | `str` | Optional custom two-chain apo structure. |
| `prior` | `list[str]` | Optional custom two-chain prior ensemble. |

Each custom multimer file must contain exactly two non-empty protein chains in
the same order as the two input sequences. Each component is globally aligned
to its corresponding input sequence using the same insertion/deletion behavior
as monomer sources. The pair remains in one shared rigid frame during prior
sampling.

### Apo and prior resolution

Protein monomers, RNA chains, and protein multimers follow one source-resolution
contract:

| Input | Resolved apo | Resolved priors |
| :--- | :--- | :--- |
| `apo` and `prior` | `apo` | `prior` |
| `apo` only | `apo` | `[apo]` |
| `prior` only | `prior[0]` with a warning | `prior` |
| neither | Error until the external apo sampler is integrated | — |

Different prior ensemble sizes are independently shuffled from the query seed
and cycled to the largest ensemble size. `--num-samples` controls diffusion
samples only; it does not change the number of prior sources.

Structure paths may be absolute, relative to the current working directory, or
relative to the input YAML/JSON file.

### DNA and RNA

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | One or more physical chain IDs. |
| `sequence` | `str` | DNA uses A/C/G/T; RNA uses A/C/G/U. |
| `modifications` | `dict[int, str]` | Optional 1-based residue index to CCD code mapping. |

DNA does not accept custom apo or prior files. Its apo coordinates are left
missing and its prior coordinates are initialized by the existing prior sampler.
RNA accepts `apo` and `prior` with the source-resolution rules above, and at
least one of them is required until the external apo sampler is integrated.

### Ligands

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | One or more physical chain IDs. |
| `smiles` | `str` | A SMILES representation. |
| `ccd` | `str` or `list[str]` | One CCD code or a multi-residue CCD sequence. |

Specify exactly one of `smiles` and `ccd`.

### Constraints

The top-level field is `constraints`:

```yaml
constraints:
  - bond:
      atom1: [A, 20, NZ]
      atom2: [C, 1, C08]
  - distance:
      atom1: [A, 10, CA]
      atom2: [B, 3, "C1'"]
      range: [4, 8]
```

Atom references use `[chain_id, one_based_residue_index, atom_name]`. Distance
constraints are experimental; if `range` is omitted it defaults to `[2.0, 8.0]`.

## Running inference

Single GPU:

```bash
python scripts/inference.py \
  --config configs/model/kfold-ecsi.yaml \
  --weight checkpoints/model.ckpt \
  --ccd path/to/ccd.pkl \
  --input query.yaml \
  --out-dir results
```

Multiple GPUs:

```bash
python scripts/inference_multigpu.py \
  --config configs/model/kfold-ecsi.yaml \
  --weight checkpoints/model.ckpt \
  --ccd path/to/ccd.pkl \
  --input queries \
  --out-dir results \
  --num-gpus 4
```

Common options:

| Option | Default | Description |
| :--- | :--- | :--- |
| `--config` | `configs/model/kfold-ecsi.yaml` | KFold model configuration. |
| `--weight` | Required | Weight file for the selected model version. |
| `--ccd` | Required | Serialized CCD data file. |
| `--input` | Required | One YAML/JSON file or a directory. |
| `--out-dir` | `inference_results` | Output root. |
| `--seed` | `1` | One or more query seeds. |
| `--num-samples` | `5` | Diffusion samples per query and seed. |
| `--num-recycles` | `10` | Trunk recycle count. |
| `--num-steps` | `200` | Diffusion step count. |
| `--num-workers` | `8` | DataLoader worker count. GPU structure tokenization is not run in workers. |
| `--overwrite` | `False` | Allow an existing output directory. |

Single-GPU inference additionally supports `--save-trajectory`,
`--save-confidence`, and `--dry-run`. A dry run does not need `--weight`; it
parses the queries and runs the CPU data pipeline without loading a model or
writing predictions. Multi-GPU inference accepts `--num-gpus`; if omitted, all
visible GPUs are used.

## Output

Each query gets its own directory containing the stored query description and
predicted mmCIF structures:

```text
results/
└── Example_Complex/
    ├── query.yaml
    ├── Example_Complex_seed-1_sample-0.cif
    ├── Example_Complex_seed-1_sample-0_confidences.json
    └── Example_Complex_seed-1_sample-0_confidences.npz  # single GPU with --save-confidence
```

Every successful sample writes an mmCIF file and a confidence-summary JSON file.
The single-GPU script writes raw pLDDT/PAE/PDE arrays to NPZ only when
`--save-confidence` is set. The multi-GPU script currently does not write NPZ
confidence arrays or diffusion trajectories.
