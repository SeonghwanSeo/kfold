# PLAN

Author: Seonghwan Seo (Prof. Woo Youn Kim's Lab)

This document outlines the development plan for implementing the K-Fold framework and training.

If the features listed below are completed, I will include the corresponding PR links.

### Data preparation

- [x] Prepare **own data pre-processing pipeline** from mmCIF to `RefStructure` - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
    - [x] Add CCD database according to AlphaFold3 description - [#113](https://github.com/SeonghwanSeo/kfold/pull/113)
    - [x] Preserving original residue information before modification (PTM), instead of UNK. - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
    - [x] Handling ambiguous residues (e.g., GLX, ASX) - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
- [ ] **Apo structure mapping**:
    - [x] Protein (ESMFold/AFDB) - [#23](https://github.com/SeonghwanSeo/kfold/pull/23)
    - [ ] DNA (from Langevin dynamics)
    - [ ] RNA (from Langevin dynamics)
    - [x] Ligand (ETKDG; single conformer) - [#23](https://github.com/SeonghwanSeo/kfold/pull/23), [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
- [ ] **Multiple apo structures** if possible (in particular, ligand)
    - [ ] Protein: Consider AlphaFold2 / OpenFold predicted structures and Holo structures as well.
    - [x] Ligand: AlphaFold uses different conformers for each seed. - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)

### Data featurization

- [x] Include symmetry information in the input features for validation - [#67](https://github.com/SeonghwanSeo/kfold/pull/67)
- [x] Implement cropping algorithm used in AF-M/AF3. - [#83](https://github.com/SeonghwanSeo/kfold/pull/83), [#86](https://github.com/SeonghwanSeo/kfold/pull/86)
- [x] Implement Better cropping algorithm for **apo-to-holo** scheme. - [#85](https://github.com/SeonghwanSeo/kfold/pull/85)
- [x] Implement symmetry alignment between apo and holo structures. - [#92](https://github.com/SeonghwanSeo/kfold/pull/92)
- [x] Implement apo perturbation module. - [#115](https://github.com/SeonghwanSeo/kfold/pull/115)
- [x] Modularize apo perturbation module (RiePrody) - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
- [ ] Separate apo perturbation and prior sampling (langevin dynamics).
- [ ] Implement symmetry alignment between ref conformers and holo structures during training.
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
    - [x] Add one-hot distogram option (contact map) - [#140](https://github.com/SeonghwanSeo/kfold/pull/140)
- [x] Add apo-conditioned diffusion score model in Structure module - [#96](https://github.com/SeonghwanSeo/kfold/pull/96)
- [x] Add `EnsembleModule` to feed multiple pre-trained structure embeddings from structure ensemble - [#120](https://github.com/SeonghwanSeo/kfold/pull/120)
- [x] Add `Interformer` to allow bi-directional information flow between single (`s`) and pair (`z`) representations - [#124](https://github.com/SeonghwanSeo/kfold/pull/124)
- [ ] Add `MultiStateModule` to allow multiple apo states from different sources (e.g., ESMFold, AlphaFold2, Holo structures, etc.)

### Training

- [x] Test **multi-node training**.
- [x] Add **validation pipeline** including symmetry-aware LDDT calculation.
    - [x] Add **validation metrics**. - [#33](https://github.com/SeonghwanSeo/kfold/pull/3)
    - [x] Add chain-permutation and atom-swapping for symmetry correction. - [#67](https://github.com/SeonghwanSeo/kfold/pull/67)
- [x] Correct **cropping algorithm** to match training losses and validation metrics to Boltz1. - [#71](https://github.com/SeonghwanSeo/kfold/pull/71)
- [x] Add **compile** option for training - [#138](https://github.com/SeonghwanSeo/kfold/pull/138)
    - [ ] Fix the issue related to model save/checkpointing after compilation (`_orig_mod`)
- [x] Implement multi-dataset training pipeline (e.g., RCSB + AFDB + ...) - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
- [ ] Prepare our own validation set.

### Inference
- [x] Prepare data preparation pipeline for inference (from YAML config to `TokenizedStructure`) - [#113](https://github.com/SeonghwanSeo/kfold/pull/113)

### Benchmark

- [x] Implement PDB/mmCIF writer
- [x] Implement benchmark pipeline for K-Fold - [#113](https://github.com/SeonghwanSeo/kfold/pull/113)
