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
- [ ] Parse **multiple bioassembly structures** from RCSB (optional).
    - Some structures have multiple valid biological assemblies.
    - Need to select one assembly for training. (AF3: first assembly is used)

### Data featurization

- [ ] Better cropping algorithm for **apo-to-holo** diffusion bridge training.
- [ ] Implement apo perturbation module.
- [ ] Include symmetry information in the input features for validation.
- [ ] Implement pocket conditioning features as in Boltz1.

### Model implementation

- [ ] Add Boltz1 layers

### Training

- [x] Test **multi-node training**.
- [ ] Add **validation pipeline** including symmetry-aware LDDT calculation.

### Benchmark

- [ ] Implement benchmark pipeline for K-Fold.
