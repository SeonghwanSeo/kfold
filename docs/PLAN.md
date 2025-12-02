# PLAN

Author: Seonghwan Seo (Prof. Woo Youn Kim's Lab)

This document outlines the development plan for implementing the K-Fold framework and training.

If the features listed below are completed, I will include the corresponding PR links.

### Data preparation

- [x] Prepare own data processing pipeline to **LMDB format** (`TokenizedStructure`).
    - 251116 - [#23](https://github.com/SeonghwanSeo/kfold/pull/23)
- [ ] **Apo structure mapping**:
    - [x] Protein (ESMFold): 251116 - [#23](https://github.com/SeonghwanSeo/kfold/pull/23)
    - [ ] DNA
    - [ ] RNA
    - [x] Ligand (ETKDG; single conformer): 251116 - [#23](https://github.com/SeonghwanSeo/kfold/pull/23)
- [ ] **Multiple apo structures** if possible (in particular, ligand)
    - AlphaFold uses different conformers for each seed. (Boltz2: uses 10 different conformers as `ref_pos` for ligands)
- [ ] Parse **multiple bioassembly structures** from RCSB (optional, less-priority).
    - Some structures have multiple valid biological assemblies.
    - Need to select one assembly for training. (AF3/Boltz1/Boltz2/Protenix/OpenFold-3: first assembly is used)

### Data featurization

- [x] Add PDBBind split introduced by EquiBind for Proof of Concept.
- [ ] Better cropping algorithm for **apo-to-holo** diffusion bridge training.
- [ ] Implement apo perturbation module.
- [x] Include symmetry information in the input features for validation.
- [ ] Implement contact conditioning features as in Boltz1.

### Model implementation

- [x] Add AlphaFold3 layers: 251124 - [#53](https://github.com/SeonghwanSeo/kfold/pull/53)
- [x] Add Boltz1 layers: 251118 - [#29](https://github.com/SeonghwanSeo/kfold/pull/29)
    - [x] Load pre-trained weights from Boltz1.
    - [x] Introduce Cu-equivariance kernels for acceleration.
- [x] Implement initial diffusion bridge framework
- [x] Integrate to pre-trained sequence/structure embeddings and train.

### Training

- [x] Test **multi-node training**.
- [x] Add **validation pipeline** including symmetry-aware LDDT calculation.
    - [x] Add **validation metrics**.
    - [x] Add chain-permutation and atom-swapping for symmetry correction.
- [ ] Add geometric OT computation for better training stability.
    - [ ] Add **symmetry in cropped structure**.
- [ ] Correct **cropping algorithm** to match training losses and validation metrics to Boltz1.

### Benchmark

- [x] Implement PDB/mmCIF writer
    - [x] PDB writer
    - [x] mmCIF writer
- [ ] Implement benchmark pipeline for K-Fold.
