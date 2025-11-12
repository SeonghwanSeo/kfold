# Compare to Boltz

This document provides a comparison between our implementation and the Boltz.

## Feature name

### Atom features

You can get atom features from `kfold.data.model_input.AtomLayout`:

```python
from kfold.data.model_input import FoldingInput, AtomLayout
model_input: FoldingInput = ...
atom_layout: AtomLayout = model_input.atom

ref_pos = atom_layout.ref_pos  # (Natom, 3)
...
```


| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
| `ref_pos` | `ref_pos` or `apo_coords` | `(Natom, 3)` | Reference conformer of each residue, which can be replaced to `apo_coords` |
| `ref_space_uid` | same | `(Natom, 3)` | Index of reference conformer |
| `ref_atom_name_chars` | same | `(Natom, 4, 64)` | Atom name, one-hot encoding |
| `ref_element` | same | `(Natom, 128)` | Atomic number, one-hot encoding |
| `ref_charge` | same | `(Natom,)` | Atom charge (float) |
| `atom_to_token` | (*1) | `(Natom, Ntoken)` vs `(Natom,)` | Mapping from atom index to token index, one-hot vector vs integer |
| `atom_pad_masks` | `pad_mask` | `(Natom,)` | Mask for valid atoms or padding |
| `atom_resolved_mask` | `resolved_mask` | `(Natom,)` | Mask for resolved atoms |
| `coords` | `label_coords` | `(Nholo, Natom, 3)` vs `(Natom, Nholo, 3)` | Target coordinates for training, `Nholo` is the number of ensemble (Always 1). |
| - | `apo_coords` | `(Natom, Napo, 3)` | Apo structure coordinates |


* \*1: `atom_to_token` in Boltz can be accessed via `FoldingInput` instead of `AtomLayout`: `model_input.atom_to_token`.

### Token features

You can get token features from `kfold.data.model_input.TokenLayout`:

```python
from kfold.data.model_input import FoldingInput, TokenLayout
model_input: FoldingInput = ...
token_layout: TokenLayout = model_input.token

token_index = token_layout.token_index  # (Ntoken,)
...
```

| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
| `token_index` | same | `(Ntoken,)` | Index of tokens, same to `torch.arange(len(tokens))` |
| `residue_index` | same | `(Ntoken,)` | Starting from 0 vs 1 |
| `asym_id` | same | `(Ntoken,)` | Starting from 0 vs 1 |
| `entity_id` | same | `(Ntoken,)` | Starting from 0 vs 1 |
| `sym_id` | same | `(Ntoken,)` | Starting from 0 vs 1 |
| `mol_type` | `chain_type` | `(Ntoken,)` | Chain Type (protein, dna, rna, ligand) |
| `res_type` | same(\*2) | `(Ntoken, 33)` vs `(Ntoken, 32)` | Residue type (one-hot encoded) |
| `token_pad_mask` | `pad_mask` | `(Ntoken,)` | Mask for valid tokens or padding |
| `token_resolved_mask` | `resolved_mask` | `(Ntoken,)` | Mask for resolved tokens |
| `pocket_feature` | `pocket_contact_type` | `(Ntoken,)` | Feature for pocket tokens, TODO: implement more |
| `cyclic_period` | - | `(Ntoken,)` | Cyclic period, not used in Boltz1. |


* \*2: Boltz use 31 types + 1 gap + 1 padding, while AlphaFold3 uses 31 types + 1 gap without padding token.

#### Token features for distogram head

| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
| `disto_center` | `disto_coords` | `(Ntoken, 3)` | Disto coords |
| `disto_target` | - | `(Ntoken, Ntoken, Nbin)` | Can be obtained from `disto_center` |
| `token_disto_mask` | `disto_mask` | `(Ntoken,)` | Mask for valid disto tokens |


#### Token features for confidence head
| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
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

token_index = bond_layout.token_index  # (Ntoken,)
...
```

| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
| `token_bonds` | `token_index` | `(Ntoken, Ntoken, 1)` vs `(Nbond, 2)` | Index of connecting tokens |
| - | `asym_id` | `(Nbond, 2)` | Index of connecting chains |
| - | `atom_index` | `(Nbond, 2)` | Index of connecting atoms |
| - | `bond_type` | `(Nbond,)` | Bond type (single, double, triple, aromatic, covalent) |
| - | `is_polymer_ligand` (\*3) | `(Nbond,)` | Whether the bond is between polymer and ligand |
| - | `is_ligand_ligand` | `(Nbond,)` | Whether the bond is between ligands |
| - | `pad_mask` | `(Nbond,)` | Mask for valid bonds |

* \*3: Used to compute bond-loss in AlphaFold3, while Boltz does not use this loss.
