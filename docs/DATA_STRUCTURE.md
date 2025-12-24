# Data

**Author:** Seonghwan Seo (Prof. Woo Youn Kim's Lab)

This document describes the data structure used in **K-Fold** and compares it with the **Boltz** implementation.

## Contents

- [Data Processing Pipeline](#data-processing-pipeline)
- [Data Structure](#data-structure)
    - [Chain](#chain)
    - [Residue](#residue)
    - [Token](#token)
    - [Atom](#atom)
    - [Bond](#bond)
- [Model Input](#model-input)
    - [Atom features](#atom-features)
    - [Token features](#token-features)
    - [Bond features](#bond-features)
    - [Pretrained embeddings](#pretrained-embeddings)

---

## Data Processing Pipeline

K-Fold utilizes a data processing framework and data structures inspired by the Boltz implementation. However, there are significant differences in representation and organization designed to enhance usability and clarity.

### Boltz (Reference)
* **Preprocessed data (Storage):** `boltz.data.types.Structure` (NumPy void array)
* **During data-processing** (in `Dataset.__getitem__`):
    1.  **Tokenization:** `boltz.data.types.Structure` → `boltz.data.types.Tokenized` (NumPy void array)
    2.  **Cropping:** Crops the tokenized structure to fit the maximum length (`max_tokens`).
    3.  **Featurization:** `boltz.data.types.Tokenized` → Model input (dictionary of PyTorch tensors).

### K-Fold
* **Preprocessed data**: `kfold.data.structure.TokenizedStructure` (dataclass of NumPy arrays)
    * **From Boltz data:** `boltz.data.types.Structure` → `kfold.data.structure.TokenizedStructure` (dataclass of NumPy arrays)
    * **From mmCIF:** **TODO**
* **During data-processing** (in `Dataset.__getitem__`):
    1.  **Pre-cropping:** If the structure contains more chains than `max_chains`, it extracts neighboring chains around a randomly selected interface tokens. (See AlphaFold3 SI Section 2.5.4)
    2.  **Cropping:** Crops the tokenized structure to fit the maximum length.
    3.  **Apo perturbation:** Applies random perturbation to apo structure coordinates for data augmentation.
    4.  **Featurization:** `kfold.data.structure.TokenizedStructure` → Model input (`kfold.data.model_input.FoldingInput`, dataclass of PyTorch tensors).

---

## Data Structure

The data structure used in K-Fold (`kfold.data.structure.TokenizedStructure`) is largely inspired by the Boltz implementation (`boltz.data.types.Structure`).
The main difference lies in the representation format: K-Fold uses Python dataclasses with NumPy arrays, which are easier to interpret and manage than NumPy void arrays.

> **NOTE:** In this section, we only show the common features between Boltz and K-Fold. To see the complete list of features and their shapes in K-Fold, please refer to [`src/kfold/data/structure.py`](./src/kfold/data/structure.py).

```python
from kfold.data import structure

struct: structure.TokenizedStructure = ...
chain_struct: structure.Chain = struct.chain
residue_struct: structure.Residue = struct.residue
token_struct: structure.Token = struct.token
atom_struct: structure.Atom = struct.atom
bond_struct: structure.Bond = struct.bond

asym_id = chain_struct.asym_id  # Shape: (Nchain,)
# ...
````

### Chain

You can access chain-level structures via `kfold.data.structure.Chain`:

| Boltz Field | K-Fold Field | Shape | Description |
| :--- | :--- | :--- | :--- |
| `name` | - | `(Nchain,)` | **TODO:** Add description |
| `mol_type` | `chain_type` | `(Nchain,)` | Chain Type (protein, dna, rna, ligand) |
| `entity_id` | *same* | `(Nchain,)` | Starting from 0 vs 1 |
| `sym_id` | *same* | `(Nchain,)` | Starting from 0 vs 1 |
| `asym_id` | *same* | `(Nchain,)` | Starting from 0 vs 1 |
| `res_num` | `num_residues` | `(Nchain,)` | Number of residues in each chain |
| `atom_num` | `num_atoms` | `(Nchain,)` | Number of atoms in each chain |
| - | `num_tokens` | `(Nchain,)` | Number of tokens in each chain |

### Residue

You can access residue-level structures via `kfold.data.structure.Residue`:

| Boltz Field | K-Fold Field | Shape | Description |
| :--- | :--- | :--- | :--- |
| `name` | *same* | `(Nresidue,)` | Residue name |
| `res_type` | *same* $^1$ | `(Nresidue,)` | Residue type |
| - | `chain_type` | `(Nresidue,)` | Chain Type (protein, dna, rna, ligand) |
| `atom_num` | `num_atoms` | `(Nresidue,)` | Number of atoms in each residue |
| `is_standard` | *same* | `(Nresidue,)` | Whether the residue is standard |
| `is_present` | `resolved_mask` | `(Nresidue,)` | Whether the residue is resolved |

- 1: Boltz uses 31 types + 1 gap + 1 padding. K-Fold (AlphaFold3 style) uses 31 types + 1 gap, excluding the padding token.

### Token

Since the raw data structure of Boltz does not include a token-level representation, we introduce `kfold.data.structure.Token` to represent token-level features explicitly.

### Atom

> **NOTE:** While Boltz uses a dense representation for atom features (shape: `(Natom, ...)`), K-Fold uses a token-level sparse representation (shape: `(Ntoken, 24, ...)`) to facilitate easier cropping and management.

You can access atom-level structures via `kfold.data.structure.Atom`:

| Boltz Field | K-Fold Field | Shape (Boltz vs K-Fold) | Description |
| :--- | :--- | :--- | :--- |
| `name` | `ref_atom_name_chars` | `(Natom,)` vs `(Ntoken, 24, 4)` | Atom name |
| `element` | `ref_element` | `(Natom,)` vs `(Ntoken, 24)` | Atomic number |
| `charge` | `ref_charge` | `(Natom,)` vs `(Ntoken, 24)` | Atom charge |
| `conformer` | `ref_pos` | `(Natom, 3)` vs `(Ntoken, 24, 3)` | Reference conformer of each atom |
| `coords` | *same* | `(Natom, 3)` vs `(Ntoken, 24, Nholo, 3)` | Atom coordinates |
| `is_present` | `resolved_mask` | `(Natom,)` vs `(Ntoken, 24)` | Whether the atom is resolved |
| `chirality` | - | `(Natom,)` | **TODO:** Add description |
| - | `apo_coords` | - vs `(Ntoken, 24, Napo, 3)` | Apo structure coordinates |
| - | `apo_mask` | - vs `(Ntoken, 24, Napo, 3)` | Apo structure mask |

### Bond

While Boltz separates bonds into `Bond` and `Connection`, K-Fold unifies them into a single `kfold.data.structure.Bond` structure for simplicity.

You can access bond-level structures via `kfold.data.structure.Bond`:

| Boltz Field | K-Fold Field | Shape (Boltz vs K-Fold) | Description |
| :--- | :--- | :--- | :--- |
| `chain_1`, `chain_2` | `asym_id` | `(Nbond,)` vs `(Nbond, 2)` | Index of connecting chains |
| `res_1`, `res_2` | - | `(Nbond,)` | **TODO:** Add description |
| - | `token_index` | - vs `(Nbond, 2)` | Index of connecting tokens |
| `atom_1`, `atom_2` | `atom_index` | `(Nbond,)` vs `(Nbond, 2)` | Index of connecting atoms |
| `type` | `bond_type` | - vs `(Nbond,)` | Bond type |

-----

## Model Input

Since the input features of Boltz are in a pure dictionary format, it is difficult to visualize the overall data structure and shapes. Therefore, K-Fold provides high-level data structures for model input features via `kfold.data.model_input.FoldingInput`. This contains sub-layouts for atom, token, and bond features.

### Atom features

You can get atom features from `kfold.data.model_input.AtomLayout`:

```python
from kfold.data.model_input import FoldingInput, AtomLayout
model_input: FoldingInput = ...
atom_layout: AtomLayout = model_input.atom

ref_pos = atom_layout.ref_pos  # Shape: (Natom, 3)
# ...
```

| Boltz Field | K-Fold Field | Shape (Boltz vs K-Fold) | Description |
| :--- | :--- | :--- | :--- |
| `ref_pos` | *same* $^1$ | `(Natom, 3)` | Reference conformer of each residue |
| `ref_space_uid` | *same* | `(Natom, 3)` | Index of reference conformer |
| `ref_atom_name_chars` | *same* | `(Natom, 4, 64)` | Atom name, one-hot encoding |
| `ref_element` | *same* | `(Natom, 128)` | Atomic number, one-hot encoding |
| `ref_charge` | *same* | `(Natom,)` | Atom charge (float) |
| `atom_to_token` | $^2$ | `(Natom, Ntoken)` vs `(Natom,)` | Mapping from atom to token (one-hot vs integer) |
| `atom_pad_masks` | `pad_mask` | `(Natom,)` | Mask for valid atoms or padding |
| `atom_resolved_mask` | `resolved_mask` | `(Natom,)` | Mask for resolved atoms |
| `coords` | `label_coords` | `(Nholo, Natom, 3)` vs `(Natom, 3)` | Target coordinates for training $^3$ |
| - | `apo_coords` | `(Natom, 3)` | Apo structure coordinates |
| - | `apo_mask` | `(Natom,)` | Apo structure mask |

- 1: In K-Fold, we are considering replacing `ref_pos` with `apo_coords`.
- 2: `atom_to_token` in Boltz can be accessed via `FoldingInput` instead of `AtomLayout`: `model_input.atom_to_token`.
- 3: `Nholo` represents the number of bioassemblies. This is always 1 in AlphaFold3 (Using the first bioassembly).

### Token features

You can get token features from `kfold.data.model_input.TokenLayout`:

```python
from kfold.data.model_input import FoldingInput, TokenLayout
model_input: FoldingInput = ...
token_layout: TokenLayout = model_input.token

token_index = token_layout.token_index  # Shape: (Ntoken,)
# ...
```

| Boltz Field | K-Fold Field | Shape (Boltz vs K-Fold) | Description |
| :--- | :--- | :--- | :--- |
| `token_index` | *same* | `(Ntoken,)` | Index of tokens (`=torch.arange(len(tokens))`) |
| `residue_index` | *same* | `(Ntoken,)` | Starting from 0 vs 1 |
| `asym_id` | *same* | `(Ntoken,)` | Starting from 0 vs 1 |
| `entity_id` | *same* | `(Ntoken,)` | Starting from 0 vs 1 |
| `sym_id` | *same* | `(Ntoken,)` | Starting from 0 vs 1 |
| `mol_type` | `chain_type` | `(Ntoken,)` | Chain Type (protein, dna, rna, ligand) |
| `res_type` | *same* $^1$ | `(Ntoken, 33)` vs `(Ntoken, 32)` | Residue type (one-hot encoded) |
| `token_pad_mask` | `pad_mask` | `(Ntoken,)` | Mask for valid tokens or padding |
| `token_resolved_mask` | `resolved_mask` | `(Ntoken,)` | Mask for resolved tokens |
| `pocket_feature` | `pocket_contact_type` | `(Ntoken,)` | Feature for pocket tokens (**TODO:** implement more) |
| `cyclic_period` | - | `(Ntoken,)` | Cyclic period (not used in Boltz). |

- 1: Boltz uses 31 types + 1 gap + 1 padding. K-Fold (AlphaFold3 style) uses 31 types + 1 gap, excluding the padding token.

#### Token features for distogram head

| Boltz Field | K-Fold Field | Shape | Description |
| :--- | :--- | :--- | :--- |
| `disto_center` | `disto_coords` | `(Ntoken, 3)` | Disto coords |
| `disto_target` | - | `(Ntoken, Ntoken, Nbin)` | Can be obtained from `disto_center` |
| `token_disto_mask` | `disto_mask` | `(Ntoken,)` | Mask for valid disto tokens |

#### Token features for confidence head

| Boltz Field | K-Fold Field | Shape (Boltz vs K-Fold) | Description |
| :--- | :--- | :--- | :--- |
| `token_to_rep_atom` | `disto_index` | `(Ntoken,)` | Disto atom index (Cβ) |
| `r_set_to_rep_atom` | `center_index` | `(Ntoken_valid,)` vs `(Ntoken,)` | Center atom index (Cα); ligand atom should be masked |
| `frames_idx` | `frames_index` | `(Ntoken, 3)` | Frame defining atom index, e.g., protein: (N, Cα, C) |
| `frame_resolved_mask` | `frames_mask` | `(Ntoken,)` | Whether all frame atoms are resolved |


### Bond features

You can get bond features from `kfold.data.model_input.BondLayout`:

```python
from kfold.data.model_input import FoldingInput, TokenLayout
model_input: FoldingInput = ...
bond_layout: BondLayout = model_input.bond

token_index = bond_layout.token_index  # Shape: (Ntoken,)
# ...
```

| Boltz Field | K-Fold Field | Shape (Boltz vs K-Fold) | Description |
| :--- | :--- | :--- | :--- |
| `token_bonds` | `token_index` | `(Ntoken, Ntoken, 1)` vs `(Nbond, 2)` | Index of connecting tokens |
| - | `asym_id` | `(Nbond, 2)` | Index of connecting chains |
| - | `atom_index` | `(Nbond, 2)` | Index of connecting atoms |
| - | `bond_type` | `(Nbond,)` | Bond type |
| - | `is_polymer_ligand` $^1$ | `(Nbond,)` | Whether the bond is between polymer and ligand |
| - | `is_ligand_ligand` | `(Nbond,)` | Whether the bond is between ligands |
| - | `pad_mask` | `(Nbond,)` | Mask for valid bonds |

- 1: Used to compute bond-loss in AlphaFold3, while Boltz does not use this loss.


### Pretrained embeddings

K-Fold uses residue-level embeddings from pre-trained language models as additional input features. To facilitate this, we provide a separate data structure `kfold.data.model_input.PretrainedLayout`:

```python
from kfold.data.model_input import PretrainedLayout, FoldingInput
model_input: FoldingInput = ...
pretrained: PretrainedLayout = model_input.pretrained

seq_embedding = pretrained.sequence_embedding  # Shape: (Ntoken, D_seq)
struct_embedding = pretrained.structure_embedding  # Shape: (Ntoken, D_struct)
# ...
```
