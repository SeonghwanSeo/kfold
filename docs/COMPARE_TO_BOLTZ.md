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
| `ref_pos` | `ref_pos` or `apo_coords` | `(Natom, 3)` | Local structure (RDKit Conformer), per residue, which can be replaced to `apo_coords` |
| `ref_space_uid` | `ref_space_uid` | `(Natom, 3)` | Index of reference conformer |
| `ref_atom_name_chars` | `ref_atom_name_chars` | `(Natom, 4, 64)` vs `(Natom, 4)` | One-hot encoding vs integer |
| `ref_element` | `ref_element` | `(Natom, 128)` vs `(Natom,)` | Atomic number, One-hot encoding vs integer |
| `ref_charge` | `ref_charge` | `(Natom,)` | Atom charge (float) |
| `atom_to_token` | `token_index` | `(Natom, Ntoken)` vs `(Natom,)` | Mapping from atom index to token index, one-hot vector vs integer |
| `atom_pad_masks` | `pad_mask` | `(Natom,)` | Mask for valid atoms or padding |
| `atom_resolved_mask` | `resolved_mask` | `(Natom,)` | Mask for resolved atoms |
| `coords` | `label_coords` | `(Nholo, Natom, 3)` vs `(Natom, Nholo, 3)` | Target coordinates for training, `Nholo` is the number of ensemble (Always 1). |
| - | `apo_coords` | `(Natom, Natompo, 3)` | Apo structure coordinates |


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
| `token_index` | - | `(Ntoken,)` | Index of tokens, same to `torch.arange(len(tokens))` |
| `residue_index` | `residue_index` | `(Ntoken,)` | Starting from 0 vs 1 |
| `asym_id` | `asym_id` | `(Ntoken,)` | Starting from 0 vs 1 |
| `entity_id` | `entity_id` | `(Ntoken,)` | Starting from 0 vs 1 |
| `sym_id` | `sym_id` | `(Ntoken,)` | Starting from 0 vs 1 |
| `mol_type` | `chain_type` | `(Ntoken,)` vs `(Ntoken,)` | Chain Type (protein, dna, ...) |
| `res_type` | `res_type` | `(Ntoken,)` | Residue type (one-hot encoded) |
| `token_pad_mask` | `pad_mask` | `(Ntoken,)` | Mask for valid tokens or padding |
| `token_resolved_mask` | `resolved_mask` | `(Ntoken,)` | Mask for resolved tokens |
| `pocket_feature` | `pocket_contact_type` | `(Ntoken,)` | Feature for pocket tokens, TODO: implement more |
| `cyclic_period` | `cyclic_period` | `(Ntoken,)` | Cyclic period, not used in Boltz1. |


#### Token features for distogram head

| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
| `disto_center` | `disto_index` | `(Ntoken, 3)` vs `(Ntoken,)` | Disto coords vs Disto atom index (Cβ) |
| `disto_target` | - | `(Ntoken, Ntoken, Nbondin)` | Can be obtained from `disto_center` |
| `token_disto_mask` | `disto_mask` | `(Ntoken,)` vs `(Ntoken,)` | Mask for valid disto tokens |


#### Token features for confidence head
| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
| `token_to_rep_atom` | `disto_index` | `(Ntoken,)` | Disto atom index (Cβ) |
| `r_set_to_rep_atom` | `center_index` | `(Ntoken_valid,)` vs `(Ntoken,)` | Center atom index (Cα) |
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
| - | `bond_mask` | `(Nbond,)` | Mask for valid bonds |
