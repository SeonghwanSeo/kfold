# Compare to Boltz

This document provides a comparison between our implementation and the Boltz.

## Feature name

### Atom features

You can get atom features from `kfold.data.model_input.AtomLayout`:

```python
from kfold.data.model_input import FoldingInput, AtomLayout
model_input: FoldingInput = ...
atom_layout: AtomLayout = model_input.atom

ref_pos = atom_layout.ref_pos  # (Na, 3)
...
```


| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
| `ref_pos` | `ref_pos` or `apo_coords` | `(Na, 3)` | Local structure (RDKit Conformer), per residue, which can be replaced to `apo_coords` |
| `ref_space_uid` | `ref_space_uid` | `(Na, 3)` | Index of reference conformer |
| `ref_atom_name_chars` | `ref_atom_name_chars` | `(Na, 4, 64)` vs `(Na, 4)` | One-hot encoding vs integer |
| `ref_element` | `ref_element` | `(Na, 128)` vs `(Na,)` | Atomic number, One-hot encoding vs integer |
| `ref_charge` | `ref_charge` | `(Na,)` | Atom charge (float) |
| `atom_to_token` | `token_index` | `(Na, Nt)` vs `(Na,)` | Mapping from atom index to token index, one-hot vector vs integer |
| `atom_pad_masks` | `pad_mask` | `(Na,)` | Mask for valid atoms or padding |
| `atom_resolved_mask` | `resolved_mask` | `(Na,)` | Mask for resolved atoms |
| `coords` | `label_coords` | `(Nholo, Na, 3)` vs `(Na, Nholo, 3)` | Target coordinates for training, `Nholo` is the number of ensemble (Always 1). |
| - | `apo_coords` | `(Na, Napo, 3)` | Apo structure coordinates |


### Token features

You can get token features from `kfold.data.model_input.TokenLayout`:

```python
from kfold.data.model_input import FoldingInput, TokenLayout
model_input: FoldingInput = ...
token_layout: TokenLayout = model_input.token

token_index = token_layout.token_index  # (Nt,)
...
```

| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
| `token_index` | - | `(Nt,)` | Index of tokens, same to `torch.arange(len(tokens))` |
| `residue_index` | `residue_index` | `(Nt,)` | Starting from 0 vs 1 |
| `asym_id` | `asym_id` | `(Nt,)` | Starting from 0 vs 1 |
| `entity_id` | `entity_id` | `(Nt,)` | Starting from 0 vs 1 |
| `sym_id` | `sym_id` | `(Nt,)` | Starting from 0 vs 1 |
| `mol_type` | `chain_type` | `(Nt,)` vs `(Nt,)` | Chain Type (protein, dna, ...) |
| `res_type` | `res_type` | `(Nt,)` | Residue type (one-hot encoded) |
| `token_pad_mask` | `pad_mask` | `(Nt,)` | Mask for valid tokens or padding |
| `token_resolved_mask` | `resolved_mask` | `(Nt,)` | Mask for resolved tokens |
| `pocket_feature` | `pocket_contact_type` | `(Nt,)` | Feature for pocket tokens, TODO: implement more |
| `cyclic_period` | `cyclic_period` | `(Nt,)` | Cyclic period, not used in Boltz1. |


#### Token features for distogram head

| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
| `disto_center` | `disto_index` | `(Nt, 3)` vs `(Nt,)` | Disto coords vs Disto atom index (Cβ) |
| `disto_target` | - | `(Nt, Nt, Nbin)` | Can be obtained from `disto_center` |
| `token_disto_mask` | `disto_mask` | `(Nt,)` vs `(Nt,)` | Mask for valid disto tokens |


#### Token features for confidence head
| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
| `token_to_rep_atom` | `disto_index` | `(Nt,)` | Disto atom index (Cβ) |
| `r_set_to_rep_atom` | `center_index` | `(Nt_valid,)` vs `(Nt,)` | Center atom index (Cα) |
| `frames_idx` | `frames_index` | `(Nt, 3)` | Frame defining atom index, e.g., protein: (N, Cα, C) |
| `frame_resolved_mask` | `frames_mask` | `(Nt,)` | Whether all frame atoms are resolved |


### Bond features

You can get bond features from `kfold.data.model_input.BondLayout`:

```python
from kfold.data.model_input import FoldingInput, TokenLayout
model_input: FoldingInput = ...
bond_layout: BondLayout = model_input.bond

token_index = bond_layout.token_index  # (Nt,)
...
```

| Boltz | K-Fold | Shape | Descriptions |
|---------|-----|-------|--------------|
| `token_bonds` | `token_index` | `(Nt, Nt, 1)` vs `(Nb, 2)` | Index of connecting tokens |
| - | `asym_id` | `(Nb, 2)` | Index of connecting chains |
| - | `atom_index` | `(Nb, 2)` | Index of connecting atoms |
| - | `bond_type` | `(Nb,)` | Bond type (single, double, triple, aromatic, covalent) |
| - | `bond_mask` | `(Nb,)` | Mask for valid bonds |
