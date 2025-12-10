# PLAN

Author: Seonghwan Seo (Prof. Woo Youn Kim's Lab)

This document outlines the development plan for implementing the K-Fold framework and training.

If the features listed below are completed, I will include the corresponding PR links.

### Data preparation

- [x] Prepare own data processing pipeline to **LMDB format** (`TokenizedStructure`) - [#23](https://github.com/SeonghwanSeo/kfold/pull/23)
- [ ] **Apo structure mapping**:
    - [x] Protein (ESMFold) - [#23](https://github.com/SeonghwanSeo/kfold/pull/23)
    - [ ] DNA
    - [ ] RNA
    - [x] Ligand (ETKDG; single conformer) - [#23](https://github.com/SeonghwanSeo/kfold/pull/23)
- [ ] **Multiple apo structures** if possible (in particular, ligand)
    - AlphaFold uses different conformers for each seed. (Boltz2: uses 10 different conformers as `ref_pos` for ligands)

### Data featurization

- [x] Add PDBBind split introduced by EquiBind for Proof of Concept. - [#57](https://github.com/SeonghwanSeo/kfold/pull/57)
- [x] Include symmetry information in the input features for validation - [#67](https://github.com/SeonghwanSeo/kfold/pull/67)
- [x] Implement cropping algorithm used in AF-M/AF3. - [#83](https://github.com/SeonghwanSeo/kfold/pull/83), [#86](https://github.com/SeonghwanSeo/kfold/pull/86)
- [x] Implement Better cropping algorithm for **apo-to-holo** scheme. - [#85](https://github.com/SeonghwanSeo/kfold/pull/85)
- [x] Implement symmetry alignment between apo and holo structures. - [#92](https://github.com/SeonghwanSeo/kfold/pull/92)
- [ ] Implement apo perturbation module.
- [ ] Implement contact conditioning features as in Boltz1.

### Model implementation

- [x] Add AlphaFold3 layers - [#53](https://github.com/SeonghwanSeo/kfold/pull/53)
- [x] Add Boltz1 layers - [#29](https://github.com/SeonghwanSeo/kfold/pull/29)
    - [x] Load pre-trained weights from Boltz1.
    - [x] Introduce Cu-equivariance kernels for acceleration.
    - [x] Check the validation results are consistent with Boltz1 official repository. 
- [x] Implement initial diffusion bridge framework - [#48]
- [x] Integrate to pre-trained sequence/structure embeddings and train. - [#62](https://github.com/SeonghwanSeo/kfold/pull/62)
- [x] Add apo information (e.g., distance map) before Pairformer trunk - [#80](https://github.com/SeonghwanSeo/kfold/pull/80)
- [ ] Add apo information (e.g., distance map) in Structure module.

### Training

- [x] Test **multi-node training**.
- [x] Add **validation pipeline** including symmetry-aware LDDT calculation.
    - [x] Add **validation metrics**. - [#33](https://github.com/SeonghwanSeo/kfold/pull/3)
    - [x] Add chain-permutation and atom-swapping for symmetry correction. - [#67](https://github.com/SeonghwanSeo/kfold/pull/67)
- [x] Correct **cropping algorithm** to match training losses and validation metrics to Boltz1. - [#71](https://github.com/SeonghwanSeo/kfold/pull/71)

### Benchmark

- [x] Implement PDB/mmCIF writer
- [ ] Implement benchmark pipeline for K-Fold.
