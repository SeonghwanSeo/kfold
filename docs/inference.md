# K-Fold Inference Guide

This document provides a guide on how to perform structural inference (prediction) using the K-Fold model. K-Fold is designed for biomolecular co-folding, allowing you to predict the structure of complexes involving proteins, DNA, RNA, and ligands.

## Contents
- [Overview](#overview)
- [Input File Format](#input-file-format)
    - [Proteins](#proteins)
    - [DNA and RNA](#dna-and-rna)
    - [Ligands](#ligands)
    - [Covalent Bonds (Optional)](#covalent-bonds-optional)
- [Running Inference](#running-inference)
    - [Single-GPU Inference](#single-gpu-inference)
    - [Multi-GPU Inference](#multi-gpu-inference)
    - [Command Line Options](#command-line-options)
- [Output Format](#output-format)

## Overview

K-Fold takes a description of a molecular complex (sequences and optionally apo structures) and predicts its 3D coordinates. The model utilizes an **apo-to-holo** diffusion scheme, where providing an apo (unbound) structure can improve prediction accuracy for protein chains.

## Input File Format

K-Fold supports input in **YAML** or **JSON** format. An input file defines the "Query" for the model.

### Basic Example (YAML)
```yaml
name: "Example_Complex"
sequences:
  - protein:
      id: "A"
      sequence: "MKT..."
      apo: "apo/protein_a.pdb"
  - dna:
      id: "B"
      sequence: "ACGTAA.."
  - rna:
      id: "C"
      sequence: "ACGUCG.."
  - ligand:
      id: "D"
      smiles: "c1ccccc1"
```

### Basic Example (JSON)
```json
{
  "name": "Example_Complex",
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "MKT...",
        "apo": "apo/protein_a.pdb"
      }
    },
    {
      "ligand": {
        "id": "B",
        "ccd": "MOV"
      }
    }
  ]
}
```

### Proteins
Protein chains require a sequence and an **apo structure** (PDB format).

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | Chain identifier(s), e.g., `"A"` or `["A", "B"]` for homomers. |
| `sequence` | `str` | Standard amino acid sequence. |
| `apo` | `str` | Path to the apo PDB file. |
| `apo_range` | `str` | (Optional) Mapping between sequence and apo file. Format: `seq_st:seq_end->apo_st:apo_end` (1-indexed). |

**Apo Path Resolution:**
Paths to apo files are resolved in the following priority:
1. **Absolute Path**: If an absolute path is provided, it is used directly.
2. **Relative to CWD**: If the path exists relative to your **current working directory** (where you run the command), it is used.
3. **Relative to Input File**: If neither of the above works, the path is resolved relative to the **directory containing the input YAML/JSON file**.

> [!NOTE]
> **Future Improvements (TODO):**
> - **Auto-ESMFold**: We plan to support automatic generation of apo structures via ESMFold if no path is provided.
> - **Improved Folding Model**: We are developing a next-generation protein folding model intended to provide superior apo structure predictions compared to existing benchmarks.


### DNA and RNA
Nucleic acids are specified by their sequence.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | Chain identifier(s). |
| `sequence` | `str` | Nucleotide sequence (A, C, G, T for DNA; A, C, G, U for RNA). |

### Ligands
Ligands can be specified using **SMILES** or **CCD** (Chemical Component Dictionary) codes.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | Chain identifier(s). |
| `smiles` | `str` | SMILES string for the ligand. |
| `ccd` | `str` or `list[str]` | CCD code(s). Use a list for multi-residue ligands (e.g., `["GLY", "TYR"]`). |

*Note: You must provide either `smiles` or `ccd`, but not both.*

### Constraint (Optional)

```yaml
name: "Covalent_Complex"
sequences:
  - protein:
      id: "A"
      sequence: "MKT..."
  - dna:
      id: "B"
      sequence: "ACGTAA.."
  - ligand:
      id: "C"
      ccd: "WF1"
  - ligand:
      id: "D"
      smiles: "CNCBr"
constraint:
  - bond:
      atom1: ["A", 20, "NZ"]  # Chain A, residue 20, atom NZ
      atom2: ["C", 1, "C08"]  # Chain B, residue 1, atom C08
  - distance:
      atom1: ["A", 10, "CA"]  # Chain A, residue 20, atom CA
      atom2: ["B", 3, "C1'"]  # Chain B, residue 3, atom C1'
      range: [4, 8]       # Desired distance range in Angstroms
```

#### Covalent Bonds

You can specify custom covalent bonds between atoms in different chains using the `bond` entry within the `constraint` list.

Each `bond` requires two atoms, `atom1` and `atom2`, specified as `[chain_id, residue_number, atom_name]`.
- **chain_id**: The identifier assigned in the `sequences` section (e.g., `"A"`).
- **residue_number**: 1-based index of the residue in the sequence.
- **atom_name**: The atom name according to the following rules:
    - For **Proteins/DNA/RNA**, atom names are not explicitly respected in the model.
    - For ligands defined by **CCD**, names are taken from the CCD definition (e.g., [WF1](https://files.rcsb.org/ligands/view/WF1.cif)).
    - (Experimental) For ligands defined by **SMILES**, names are assigned as `<Element><Index>` based on their order in the SMILES string (e.g., the atom names in `CNCBr` would be `C1`, `N2`, `C3`, `Br4`).

#### Distance Constraints (Experimental)

Distance constraints allow you to enforce spatial relationships between atoms. This is useful for incorporating experimental data like NOEs or cross-linking constraints.

A `distance` entry includes:
- `atom1`, `atom2`: The two atoms to be constrained, using the same `[chain_id, residue_number, atom_name]` format as bonds.
    - **Note**: For proteins and nucleic acids, the model always uses the center atoms (`CA` for protein, `C1'` for DNA/RNA), regardless of the `atom_name` provided.
- `range`: A list `[lower_bound, upper_bound]` in Angstroms.
    - The valid range for constrained distances is between 2.0Å and 22.0Å.
    - Set a bound to `-1` to leave the upper bound unconstrained (e.g., `[5, -1]` for a minimum distance of 5Å)


## Running Inference

### Single-GPU Inference
Use `scripts/inference.py` for standard inference tasks. This script processes requires a single input file or a directory of input files.

```bash
python scripts/inference.py \
  --config configs/model/kfold-ecsi.yaml \
  --checkpoint checkpoints/model.ckpt \
  --input query.yaml \
  --out_dir ./results/
```

### Multi-GPU Inference
For high-throughput prediction, use `scripts/inference_multigpu.py`. This script automatically distributes queries across available GPUs.

```bash
python scripts/inference_multigpu.py \
  --config configs/model/kfold-ecsi.yaml \
  --checkpoint checkpoints/model.ckpt \
  --input ./input_directory/ \
  --out_dir ./results/ \
  --num_gpus 4
```

### Command Line Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `--config` | (Required) | Path to the model YAML configuration. |
| `--checkpoint` | (Required) | Path to the trained model checkpoint (.ckpt). |
| `--input` | (Required) | Path to a single YAML/JSON file or a directory of files. |
| `--out_dir` | `./inference_results/` | Root directory to save results. |
| `--seed` | `42` | Random seed(s). You can provide multiple seeds for ensemble prediction. |
| `--num_samples` | `5` | Number of diffusion samples to generate for each query. |
| `--num_recycles` | `10` | Number of recycling iterations. |
| `--num_steps` | `200` | Number of diffusion steps. |
| `--save_trajectory`| `False` | Save the full diffusion trajectory as a PDB file. |
| `--overwrite` | `False` | Overwrite existing results in the output directory. |

---

## Output Format

The inference script creates a subdirectory for each query name in the `--out_dir`.

```text
results/
└── Example_Complex/
    ├── query.yaml                      # Copy of the input query
    ├── Example_Complex_seed-42_sample-0.cif # Predicted structure (mmCIF)
    ├── Example_Complex_seed-42_sample-1.cif
    ...
    └── Example_Complex_seed-42_sample-0_traj.pdb # Optional trajectory
```

Predictions are saved in **mmCIF** format. If `--save_trajectory` is enabled, the diffusion path is saved as a multi-model **PDB** file.
