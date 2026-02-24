# Data

**Author:** Seonghwan Seo (Prof. Woo Youn Kim's Lab)

This document describes the data structure used in **K-Fold** for protein complex structure prediction.

## Contents

- [Data Flow](#data-flow)
- [Data Structure](#data-structure)
- [Tokenized Structure](#tokenized-structure)
- [Model Input](#model-input)

---

## Data Flow

### Training

#### Preprocessing (mmCIF -> `RefStructure`)

The preprocessing stage is performed once before training to convert raw mmCIF files into an array-based format for efficient loading during training.
This processing is done using the functions defined in [`kfold.data.pipelines.cif_factory`](../src/kfold/data/pipelines/cif_factory.py).

#### On-the-fly Data Processing (`RefStructure` -> `TokenizedStructure` -> `FoldingInput`)

The on-the-fly data processing is performed during training to convert the reference structure into model input features:
1.  **Data Loading:** Loads preprocessed `RefStructure` from disk.
2.  **Pre-Cropping:** If the structure contains more chains than `max_chains`, it extracts neighboring chains around a randomly selected interface token. (See AlphaFold3 SI Section 2.5.4)
3.  **Apo Structure Population:** Populates apo structure information into the reference structure. During training, **apo perturbation** is on-the-fly applied in this step.
4.  **Tokenization:** `RefStructure` → `TokenizedStructure` (dataclass of NumPy arrays)
5.  **Cropping:** If the structure contains more tokens than `max_tokens`, it crops a structure using three cropping strategies. (See AlphaFold3 SI Section 2.7)
6.  **Featurization:** `TokenizedStructure` → `FoldingInput` (dataclass of PyTorch tensors; model input features)

### Inference

The inference stage starts by parsing a query file (YAML or JSON) that specifies the target sequences and entities (proteins, ligands, nucleic acids). Unlike training which loads ground truth structures from mmCIF, this step extracts sequences from the query and constructs a RefStructure object with zero-initialized (masked) coordinates.

1.  **Structure Preparation:** (`YAML/JSON` → `RefStructure`) Prepares the reference structure from the query sequences.
2.  **Apo Structure Population:** Populates given apo structure information into the reference structure.
3.  **Tokenization:** `RefStructure` → `TokenizedStructure` (dataclass of NumPy arrays)
4.  **Featurization:** `TokenizedStructure` → `FoldingInput` (dataclass of PyTorch tensors; model input features)

---

## Data Structure

K-Fold provides high-level data structures for reference structures via `kfold.data.types.structure.RefStructure`. This contains chains, covalent connections, and metadata. See [here](../src/kfold/data/types/structure.py) for more details.

```python
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure, Chain, CovalentConnection

ref_struct: RefStructure = ...
chains: list[Chain] = ref_struct.chains
connections: list[CovalentConnection] = ref_struct.covalent_connections
metadata: Metadata = ref_struct.metadata
```

## Tokenized Structure

K-Fold provides high-level data structures for tokenized structures via `kfold.data.types.tokenized.TokenizedStructure`. This contains sub-layouts for chain, residue, token, atom, and bond structures. See [here](../src/kfold/data/types.tokenized.py) for more details.

```python
from kfold.data.types import tokenized

struct: tokenized.TokenizedStructure = ...
chain_arr: tokenized.ChainArray = struct.chain
token_arr: tokenized.TokenArray = struct.token
atom_arr: tokenized.AtomArray = struct.atom
bond_arr: tokenized.BondArray = struct.bond

asym_id = chain_arr.asym_id  # Shape: (Nchain,)
coords = atom_arr.coords  # Shape: (Ntoken, 24, 3)
# ...
````

### Chain-level layout

| Field         | Shape       | Description |
| :---          | :---        | :--- |
| `chain_type`  | `(Nchain,)` | Chain Type (protein, dna, rna, ligand, ion) |
| `entity_id`   | `(Nchain,)` | Entity ID (1-indexed) |
| `sym_id`      | `(Nchain,)` | Sym ID (1-indexed) |
| `asym_id`     | `(Nchain,)` | Asym ID (1-indexed) |
| `num_residues`| `(Nchain,)` | Number of residues in each chain |
| `num_tokens`  | `(Nchain,)` | Number of tokens in each chain |
| `num_atoms`   | `(Nchain,)` | Number of atoms in each chain |

### Token-level layout

| Field           | Shape         | Description |
| :---            | :---          | :--- |
| `token_index`   | `(Ntoken,)`   | Token index (0-indexed) |
| `residue_index` | `(Ntoken,)`   | Residue index (1-indexed) |
| `res_type`      | `(Ntoken,)`   | Residue type |
| `chain_type`    | `(Ntoken,)`   | Chain Type (protein, dna, rna, ligand, ion) |
| `entity_id`     | `(Ntoken,)`   | Entity ID (1-indexed) |
| `asym_id`       | `(Ntoken,)`   | Asym ID (1-indexed) |
| `sym_id`        | `(Ntoken,)`   | Sym ID (1-indexed) |
| `disto_index`   | `(Ntoken,)`   | Disto atom index (Cβ) |
| `center_index`  | `(Ntoken,)`   | Center atom index (Cα) |
| `num_atoms`     | `(Ntoken,)`   | Number of atoms in each token |
| `is_standard`   | `(Ntoken,)`   | Whether the residue is standard |

### Atom-level layout

| Field                 | Shape             | Description |
| :---                  | :---              | :--- |
| `ref_atom_name_chars` | `(Ntoken, 24, 4)` | Atom name |
| `ref_element`         | `(Ntoken, 24)`    | Atomic number |
| `ref_charge`          | `(Ntoken, 24)`    | Atom charge |
| `ref_pos`             | `(Ntoken, 24, 3)` | Reference conformer of each atom |
| `ref_mask`            | `(Ntoken, 24)`    | Whether the atom is present in the reference conformer |
| `apo_coords`          | `(Ntoken, 24, 3)` | Apo structure coordinates |
| `apo_mask`            | `(Ntoken, 24)`    | Apo structure mask |
| `pad_mask`            | `(Ntoken, 24)`    | Mask for valid atoms or padding |
| `coords`              | `(Ntoken, 24, 3)` | Target coordinates for training |
| `resolved_mask`       | `(Ntoken, 24)`    | Whether the atom is resolved |

### Bond-level layout

| Field         | Shape         | Description |
| :---          | :---          | :--- |
| `asym_id`     | `(Nbond, 2)`  | Index of connecting chains |
| `token_index` | `(Nbond, 2)`  | Index of connecting tokens |
| `atom_index`  | `(Nbond, 2)`  | Index of connecting atoms |
| `bond_type`   | `(Nbond,)`    | Bond type |

-----

## Model Input

K-Fold provides high-level data structures for model input features via `kfold.data.types.model_input.FoldingInput`. This contains sub-layouts for atom, token, and bond features. See [here](../src/kfold/data/types/model_input.py) for more details.

```python
from kfold.data.types import model_input

f_input: model_input.FoldingInput = ...
chain_layout: model_input.ChainTensor = f_input.chain
token_layout: model_input.TokenTensor = f_input.token
atom_layout: model_input.AtomTensor = f_input.atom
bond_layout: model_input.BondTensor = f_input.bond

# Get properties
all_asym_id = f_input.chain.asym_id  # Shape: (Nchain,)
res_type = f_input.token.res_type  # Shape: (Ntoken,)
ref_pos = f_input.atom.ref_pos  # Shape: (Natom, 3)
```

### Chain features

| Field         | Shape       | Description |
| :---          | :---        | :--- |
| `chain_type`  | `(Nchain,)` | Chain Type (protein, dna, rna, ligand, ion) |
| `entity_id`   | `(Nchain,)` | Entity ID (1-indexed) |
| `sym_id`      | `(Nchain,)` | Sym ID (1-indexed) |
| `asym_id`     | `(Nchain,)` | Asym ID (1-indexed) |
| `num_residues`| `(Nchain,)` | Number of residues in each chain |
| `num_tokens`  | `(Nchain,)` | Number of tokens in each chain |
| `num_atoms`   | `(Nchain,)` | Number of atoms in each chain |
| `pad_mask`    | `(Nchain,)` | Mask for valid chains or padding |

### Token features

You can get chain features from `kfold.data.types.model_input.TokenTensor`:

| Field             | Shape           | Description |
| :---              | :---            | :--- |
| `token_index`     | `(Ntoken,)`     | Token index (0-indexed) |
| `org_token_index` | `(Ntoken,)`     | Original token index before cropping |
| `residue_index`   | `(Ntoken,)`     | Residue index (1-indexed) |
| `res_type`        | `(Ntoken, 32)`  | Residue type (one-hot encoded) |
| `chain_type`      | `(Ntoken,)`     | Chain Type (protein, dna, rna, ligand, ion) |
| `entity_id`       | `(Ntoken,)`     | Entity ID (1-indexed) |
| `asym_id`         | `(Ntoken,)`     | Asym ID (1-indexed) |
| `sym_id`          | `(Ntoken,)`     | Sym ID (1-indexed) |
| `center_index`    | `(Ntoken,)`     | Center atom index (Cα) |
| `disto_index`     | `(Ntoken,)`     | Disto atom index (Cβ) |
| `frames_index`    | `(Ntoken, 3)`   | Frame defining atom index, e.g., protein: (N, Cα, C) |
| `frames_mask`     | `(Ntoken,)`     | Whether all frame atoms are resolved |
| `pad_mask`        | `(Ntoken,)`     | Mask for valid tokens or padding |
| `center_coords`   | `(Ntoken, 3)`   | Center atom coords (Cα) |
| `disto_coords`    | `(Ntoken, 3)`   | Disto atom coords (Cβ) |
| `center_mask`     | `(Ntoken,)`     | Whether center atom is present |
| `disto_mask`      | `(Ntoken,)`     | Whether disto atom is present |

### Atom features

| Field                 | Shape             | Description |
| :---                  | :---              | :--- |
| `ref_atom_name_chars` | `(Natom, 4, 64)`  | One-hot encoded atom name |
| `ref_element`         | `(Natom, 128)`    | One-hot encoded atomic number |
| `ref_charge`          | `(Natom,)`        | Atom charge |
| `ref_pos`             | `(Natom, 3)`      | Reference conformer of each atom |
| `ref_space_uid`       | `(Natom,)`        | Reference atom unique ID |
| `token_index`         | `(Natom,)`        | Token index to which the atom belongs |
| `apo_coords`          | `(Natom, 3)`      | Apo structure coordinates |
| `apo_mask`            | `(Natom,)`        | Apo structure mask |
| `pad_mask`            | `(Natom,)`        | Mask for valid atoms or padding |
| `label_coords`        | `(Natom, 3)`      | Target coordinates for training |
| `resolved_mask`       | `(Natom,)`        | Whether the atom is resolved |

### Bond features

| Field               | Shape         | Description |
| :---                | :---          | :--- |
| `asym_id`           | `(Nbond, 2)`  | Index of connecting chains |
| `token_index`       | `(Nbond, 2)`  | Index of connecting tokens |
| `atom_index`        | `(Nbond, 2)`  | Index of connecting atoms |
| `bond_type`         | `(Nbond,)`    | Bond type |
| `pad_mask`          | `(Nbond,)`    | Mask for valid bonds |
| `is_polymer_ligand` | `(Nbond,)`    | Whether the bond is between polymer and ligand |
| `is_ligand_ligand`  | `(Nbond,)`    | Whether the bond is between ligands |

### Pretrained embeddings

K-Fold uses residue-level embeddings from pre-trained language models as additional input features.
To facilitate this, we provide a separate data structure `kfold.data.types.model_input.PretrainedTensor`:

| Field                 | Shape               | Description |
| :---                  | :---                | :--- |
| `sequence_embedding`  | `(Ntoken, Cseq)`    | Residue-level embedding from pre-trained language representation model |
| `structure_embedding` | `(Ntoken, Cstruct)` | Residue-level embedding from pre-trained structure representation model |
| `pad_mask`            | `(Ntoken,)`         | Mask for valid residues or padding |
