# PLAN

Author: Seonghwan Seo (Prof. Woo Youn Kim's Lab)

This document outlines the development plan for implementing the K-Fold framework and training.

If the features listed below are completed, I will include the corresponding PR links.

### Data preparation

- [x] Prepare **own data pre-processing pipeline** from mmCIF to `RefStructure` - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
    - [x] Add CCD database according to AlphaFold3 description - [#113](https://github.com/SeonghwanSeo/kfold/pull/113)
    - [x] Preserving original residue information before modification (PTM), instead of UNK. - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
    - [x] Handling ambiguous residues (e.g., GLX, ASX) - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
- [x] **Apo structure mapping**:
    - [x] Protein (ESMFold/AFDB) - [#23](https://github.com/SeonghwanSeo/kfold/pull/23)
    - [x] DNA (from Langevin dynamics)
    - [x] RNA (from Langevin dynamics)
    - [x] Ligand (ETKDG; single conformer) - [#23](https://github.com/SeonghwanSeo/kfold/pull/23), [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
- [x] **Multiple apo structures** if possible (in particular, ligand)
    - [x] Protein: Consider AlphaFold2 / OpenFold predicted structures and Holo structures as well. - [#158](https://github.com/SeonghwanSeo/kfold/pull/158)
    - [x] Ligand: AlphaFold uses different conformers for each seed. - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
- [x] **Change time split**:
    - [x] Change the time split date to more recent one. - [#171](https://github.com/SeonghwanSeo/kfold/pull/171)
- [x] **Fallback for missing apo**: Use holo protein structure w/o perturbation if no apo structure is found.
- [ ] **On-the-fly apo prior sampling on Riemannian manifold**:
    - [ ] Peptide (<16 residues)
    - [ ] DNA
    - [ ] RNA
- [ ] **Synthetic dataset preparation**: Prepare synthetic dataset for better training.
    - [ ] Disordered protein PDB distillation (used in AlphaFold3)
    - [x] AlphaFold2 protein monomer distillation.
        - [x] Incorporate AFDB structures into training dataset. - [#229](https://github.com/SeonghwanSeo/kfold/pull/229)
        - [x] Run ESMFold as an initial structure of AlphaFold2 structures, i.e., `ESMFold-to-AFDB` scheme. - [#243](https://github.com/SeonghwanSeo/kfold/pull/243)
        - [ ] Replace AFDB to MGNify dataset used in OpenFold-3.
    - [ ] RNA monomer distillation.
        - [ ] Rfam
    - [ ] Complex structures.
        - Protein-protein complexes)
            - [x] Human Protein (huMAP) with Boltz-2 - [#148](https://github.com/SeonghwanSeo/kfold/pull/148)
            - [x] Antibody-antigen complexes (NaturalAb) with Boltz-2 - [#148](https://github.com/SeonghwanSeo/kfold/pull/148)
        - Protein-ligand complexes
            - [x] SAIR(ChEMBL) with Boltz-1x - [#236](https://github.com/SeonghwanSeo/kfold/pull/236)
        - Protein-RNA complexes
            - [x] ENCORE with Boltz-2 - [#243](https://github.com/SeonghwanSeo/kfold/pull/223)
        - Protein-DNA complexes
            - [ ] JASPAR

### Data featurization

- [x] Include symmetry information in the input features for validation - [#67](https://github.com/SeonghwanSeo/kfold/pull/67)
- [x] Implement cropping algorithm used in AF-M/AF3. - [#83](https://github.com/SeonghwanSeo/kfold/pull/83), [#86](https://github.com/SeonghwanSeo/kfold/pull/86)
- [x] Implement Better cropping algorithm for **apo-to-holo** scheme. - [#85](https://github.com/SeonghwanSeo/kfold/pull/85)
- [x] Implement symmetry alignment between apo and holo structures. - [#92](https://github.com/SeonghwanSeo/kfold/pull/92)
- [x] Implement apo perturbation module. - [#115](https://github.com/SeonghwanSeo/kfold/pull/115)
- [x] Modularize apo perturbation module (RiePrody) - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
- [x] Separate apo perturbation and prior sampling (langevin dynamics). - [#154](https://github.com/SeonghwanSeo/kfold/pull/154)
- [x] Implement on-the-fly apo perturbation during training (fallback). - [#181](https://github.com/SeonghwanSeo/kfold/pull/181)
- [x] Implement symmetry alignment between ref conformers and holo structures during training. - [#209](https://github.com/SeonghwanSeo/kfold/pull/201819)
- [ ] Implement contact conditioning features.

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
- [x] Add `PLMModule` to incorporate pre-trained models' embeddings - [#204](https://github.com/SeonghwanSeo/kfold/pull/204)
- [x] Implement on-the-fly sequence/structure encoder running in model forward - [#209](https://github.com/SeonghwanSeo/kfold/pull/209)
- [ ] Extract all layer sequence embeddings instead of just the last layer for better performance, as in ESMFold.

### Training

- [x] Test **multi-node training**.
- [x] Add **validation pipeline** including symmetry-aware LDDT calculation.
    - [x] Add **validation metrics**. - [#33](https://github.com/SeonghwanSeo/kfold/pull/3)
    - [x] Add chain-permutation and atom-swapping for symmetry correction. - [#67](https://github.com/SeonghwanSeo/kfold/pull/67)
    - [x] Group symmetry for covalent ligands and glycans (chain-permutation) - [#184](https://github.com/SeonghwanSeo/kfold/pull/184)
- [x] Correct **cropping algorithm** to match training losses and validation metrics to Boltz1. - [#71](https://github.com/SeonghwanSeo/kfold/pull/71)
- [x] Add **compile** option for training - [#138](https://github.com/SeonghwanSeo/kfold/pull/138)
    - [x] Fix the issue related to model save/checkpointing after compilation (`_orig_mod`)
- [x] Implement multi-dataset training pipeline (e.g., RCSB + AFDB + ...) - [#149](https://github.com/SeonghwanSeo/kfold/pull/149)
- [x] Prepare our own validation set. - [#171](https://github.com/SeonghwanSeo/kfold/pull/171)
- [x] Compute validation metrics with low-homology chains and interfaces only. - [#184](https://github.com/SeonghwanSeo/kfold/pull/184)
- [x] Update model selection criteria as average of top5 and top1 LDDT. - [#184](https://github.com/SeonghwanSeo/kfold/pull/184)
- [x] Implement **WeightedMSELoss**, **BondLoss**, and **SmoothLDDTLoss** for training.

### Inference
- [x] Prepare data preparation pipeline for inference (from YAML config to `TokenizedStructure`) - [#113](https://github.com/SeonghwanSeo/kfold/pull/113)
- [x] Implement smiles input option for ligand - [#250](https://github.com/SeonghwanSeo/kfold/pull/250)
- [ ] Implement covalent bond constraint option
- [x] Implement multi-seed inference option - [#256](https://github.com/SeonghwanSeo/kfold/pull/256)

### Benchmark

- [x] Implement PDB/mmCIF writer
- [x] Implement benchmark pipeline for K-Fold - [#113](https://github.com/SeonghwanSeo/kfold/pull/113)
- [ ] Prepare our own RecentPDB test set.
