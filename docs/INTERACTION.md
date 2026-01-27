# Interaction Pipeline (Training)

This document captures the current interaction-aware training pipeline and the
data processing flow as implemented after commit 738f37c. It is intended as a
step-by-step SOP for re-implementing the pipeline in a new repository. It
focuses on training only (no inference path).

## Scope

- In scope: data preprocessing, TokenizedStructure and FoldingInput shapes,
  interaction feature construction, model integration, interaction head and
  loss, and training configuration.
- Out of scope: inference-only paths, evaluation tooling, and model refactors
  unrelated to interaction features.

## Delta vs 738f37c (high-level)

- Added interaction constants and residue-to-interaction mapping.
- Added Token.interaction_type (multi-hot) in TokenizedStructure and
  TokenLayout in FoldingInput.
- Added interaction-aware input embedder and trunk configs.
- Added interaction head module and interaction loss in training.
- Updated tokenization to materialize interaction_type in preprocessing.
- Updated BaseFoldingModel to pass extra embedder outputs into trunk.
- Minor runtime change: dataloaders use persistent_workers=False.

## End-to-end pipeline overview

1) Preprocess Boltz structures into TokenizedStructure NPZ.
2) Add apo structures, save NPZ, combine to LMDB.
3) Dataset loads TokenizedStructure, crops and augments, then featurizes into
   FoldingInput.
4) Interaction_type flows from TokenizedStructure -> FoldingInput.
5) Input embedder converts interaction_type into token embeddings and pair
   interaction features.
6) Trunk processes the pair representation (fused or separate mode).
7) Interaction head predicts pair interaction logits.
8) Interaction loss uses disto coords and interaction targets.

## Data preprocessing SOP (Boltz -> TokenizedStructure)

### Inputs

- Boltz processed dataset (RCSB) NPZ files.
- Apo structures (ESMFold for proteins; holo fallback for DNA/RNA; ETKDG for
  ligands).

### Script entry points

- scripts/process/boltz_rcsb/a1_preprocess_rcsb.py
- scripts/process/boltz_rcsb/a2_combine_lmdb.py
- scripts/process/boltz_rcsb/b_get_manifest.py

### Tokenization (kfold/utils/boltz/process.py::tokenize_structure)

1) Iterate chains and residues from BoltzStructure.
2) Compute residue type and chain type.
   - Residue index uses ResidueName.idx (not .index).
3) For each token, compute interaction_type:
   - interaction_type is a multi-hot vector over NUM_INTERACTION_TYPES.
   - For standard residues, use get_residue_interaction_type(res_name, chain_type).
   - For ligand/ion atoms, compute PLIP-style atom-level features from
     SMILES/CCD (hydrophobic, HBD/HBA, charged groups, aromatic).
   - For other non-standard residues, use ResidueName.UNK with the
     chain type (protein UNK -> empty; ligand/ion placeholder if no SMILES).
4) Convert per-token interaction_type tuples into a dense multi-hot array:
   [Ntoken, NUM_INTERACTION_TYPES], dtype int8.
5) Store Token.interaction_type in TokenizedStructure.

### Apo structure injection (a1_preprocess_rcsb.py)

1) Load BoltzStructure NPZ.
2) Convert to TokenizedStructure via tokenize_structure.
3) For each chain:
   - Protein/DNA/RNA: load apo chain coordinates if available.
   - Ligand: use ref_pos as apo coords.
4) Save TokenizedStructure with apo_coords/apo_mask to NPZ.

### LMDB build (a2_combine_lmdb.py)

1) Read all NPZ files in output directory.
2) Store serialized TokenizedStructure in LMDB.

### Manifest build (b_get_manifest.py)

1) Create manifest pickles for full dataset and filtered subsets.

### Backward compatibility note

TokenizedStructure.from_npz_dict computes interaction_type on load if the field
is missing, using res_type and chain_type. If interaction_type is present but
float, it is thresholded to int8. Prefer regenerating NPZs to avoid ambiguity.

## Runtime dataset pipeline SOP

### Dataset entry (TrainingDataset.get_item)

1) Load TokenizedStructure from LMDB.
2) Pre-crop large complexes using PreCropper (max_chains).
3) Apply apo perturbation (random rotation/translation, symmetry handling).
4) Crop to max_tokens using the configured cropper.
5) Featurize into FoldingInput.
6) Pad to multiples for CUDA kernels (tokens to 16, atoms to 32).

### Pre-crop (dataset/cropper/pre_crop.py)

- Samples a chain/interface anchor token (interface if possible, fallback to
  chain-local token).
- Selects neighboring chains by center-atom distance.
- Does not count tiny ligands (< min_chain_atom_count) toward chain limit.
- Returns a sub-complex that is treated as the "original" structure.

### Featurization (data/featurize.py::featurize_structure)

1) Convert chain/token/atom/bond dicts from TokenizedStructure.
2) Token dict includes interaction_type as-is:
   - Cast to int64 and stored in TokenLayout.
3) Compute disto_coords and center_coords from label_coords.
4) Assemble FoldingInput:
   - TokenLayout.interaction_type: [Ntoken, NUM_INTERACTION_TYPES]

## Interaction feature construction SOP

### Primitive interaction types (constants/interaction.py)

InteractionType indices (NUM_INTERACTION_TYPES = 8):
0 DUMMY, 1 HI, 2 HBD, 3 HBA, 4 SBC, 5 SBA, 6 PP, 7 PC

Residue-to-interaction mapping is defined in RESIDUE_INTERACTION_FEATURES.

### Pair interaction types

PairInteractionType (NUM_PAIR_INTERACTION_TYPES = 5):
0 HYDROPHOBIC (HI-HI)
1 HYDROGEN_BOND (HBD-HBA, bidirectional)
2 SALT_BRIDGE (SBC-SBA, bidirectional)
3 PI_PI (PP-PP)
4 PI_CATION (PP-PC, bidirectional)

### Pair interaction computation

compute_pair_interactions(interaction_type, chain_type):

- Input: interaction_type [B, Lt, NUM_INTERACTION_TYPES].
- Optional chain_type masks placeholder ligand interactions:
  if chain_type == LIGAND and interaction_type has all types active,
  it is zeroed to avoid spurious pairs.
- Output: pair_interactions [B, Lt, Lt, 5], values in {0,1}.

## Model integration SOP

### Input embedder (PretrainedInputEmbedderWithInteraction)

Config: configs/model/module/input_embedder/pretrained_with_interaction.yaml

1) Token-level embedding:
   - LinearNoBias(NUM_INTERACTION_TYPES -> channel_s), zero init.
   - Added to s_inputs.
2) Pair interaction features:
   - compute_pair_interactions(interaction_type, chain_type).
3) Mode selection:
   - fused:
     - LinearNoBias(5 -> channel_z), zero init.
     - z_init = z_init + z_interaction.
   - separate:
     - LinearNoBias(5 -> channel_z_interaction).
     - z_interaction = concat(pair_interactions, projected).
     - embedder returns (s_inputs, s_init, z_init, z_interaction).
4) Debug: set debug_interaction or KFOLD_DEBUG_INTERACTION=1 for stats.

### Trunk (AF3PairformerTrunk)

Config: configs/model/module/trunk/af3.yaml with interaction options enabled

- fused:
  - standard PairformerStack on z_init.
- separate:
  - concat z_interaction_init to z_init, project to channel_z, then standard
    PairformerStack.

Note: pairformer_with_interaction.py exists but is not wired in this trunk.

### Base model changes (models/base.py)

- BaseFoldingModel accepts optional interaction_head.
- Input embedder may return extra tensors; base model forwards them into trunk.
- Interaction logits are computed only when train_interaction_head is true and
  interaction_head is configured.

## Interaction head and loss SOP

### Interaction head (modules/interaction_head)

- InteractionHead: LinearNoBias(channel_z -> num_pair_types).
- Logits are symmetrized by adding transpose over pair dims.

### Interaction loss (training/folding/loss/interaction.py)

1) Compute pairwise distances using token.disto_coords.
2) Mask pairs within distance_threshold (default 7.5 A).
3) Compute target = pair_interactions * within_threshold.
4) Mask to protein-protein pairs and valid disto tokens.
   - Optional: inter_chain_only=true masks to off-diagonal (asym_id !=) pairs.
5) BCEWithLogits over pair types, normalized by number of valid pairs.

## Training configuration SOP

1) Model config:
   - Add interaction_head:
     configs/model/module/interaction_head/linear.yaml
   - Switch input_embedder to the interaction variant and enable interaction
     options on the AF3 trunk:
     - module/input_embedder/pretrained_with_interaction.yaml
     - module/trunk/af3.yaml + use_interaction: true
2) Training config (configs/train/structure_only.yaml example):
   - training.train_interaction_head: true
   - loss.weights.interaction: > 0 to enable interaction loss
   - loss.interaction_loss.distance_threshold: 7.5
3) Data:
   - Ensure token.interaction_type exists in LMDB or rely on fallback in
     TokenizedStructure.from_npz_dict.

## Risks and edge cases

- Ligand interaction_type uses PLIP-style atom features when SMILES/CCD is
  available; placeholder (all types) is still masked to zero in
  compute_pair_interactions. Current interaction loss targets protein-protein
  only.
- Missing interaction_type in NPZ triggers a fallback computation; mixed or
  float dtypes are thresholded to int8.
- Pair interaction computation is O(L^2); memory grows quickly with Lt.
- pairformer_with_interaction.py is unused by default; separate mode is
  concat+project only.
- pocket_contact_type dtype is long; keep consistent when re-implementing.
- ResidueName.index is replaced by ResidueName.idx in newer code.

## Implementation checklist for new repo

1) Add constants/interaction.py and export in constants/__init__.py.
2) Add Token.interaction_type field and default dtype in data/structure.py.
3) Add TokenLayout.interaction_type in data/model_input.py.
4) Update Boltz tokenization to build interaction_type multi-hot.
5) Add fallback in TokenizedStructure.from_npz_dict for missing interaction_type.
6) Add PretrainedInputEmbedderWithInteraction and config.
7) Add interaction options to AF3PairformerTrunk and config.
8) Add interaction head module, registry, and config.
9) Update BaseFoldingModel forward to accept extra embedder outputs.
10) Add interaction loss and training flags in Lightning module.

## Minimal validation checklist

- Load one LMDB sample and confirm:
  - f_input.token.interaction_type shape is [Lt, 8] and dtype long.
  - interaction_type is multi-hot (values 0/1).
- Forward pass with fused mode:
  - z_init changes (non-zero z_interaction contribution).
  - interaction head outputs [B, Lt, Lt, 5].
- Interaction loss:
  - Runs without NaN, uses non-empty pair mask for protein-protein samples.
