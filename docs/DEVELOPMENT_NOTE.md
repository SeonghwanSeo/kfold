# Development Note

Author: Seonghwan Seo (Prof. Woo Youn Kim's Lab)

This document provides all detailed notes on the development of the K-Fold project.

## Contents:
- [Reference](#reference)
- [Reproduction of AlphaFold3 Algorithms](#reproduction-of-alphafold3-algorithms)
- [Implementation of K-Fold](#implementation-of-k-fold)
- [Data construction](#data-construction)

---

## Reference

In developing this codebase, we referred to the following repositories:
- [AlphaFold3](https://github.com/google-deepmind/alphafold3)
- [Boltz](https://github.com/jwohlwend/boltz)
- [Protenix](https://github.com/bytedance/Protenix)
- [OpenFold3](https://github.com/aqlaboratory/openfold-3)

In particular, we started from Boltz's implementation.
- NOTE: Boltz1's processed data and training code are publicly available, but Boltz2's training code is not, so implementation is based on Boltz1.

---

## Reproduction of AlphaFold3 Algorithms

This section explains how we implement AlphaFold3 algorithm and compares it to Boltz1 or other models, highlighting the differences.

### Data Processing

1. We started from Boltz1's data pre-processed dataset. We re-featurized the data to match our model's input format. (`TokenizedStructure`)
2. Boltz1's Cropping algorithm is implemented differently from AlphaFold3. For faster implementation, we use the Boltz1 implementation.

### Architecture

Compared to the AlphaFold3 article, Boltz1 modified some layers and architectures. In this repository, we revert these modifications to match the original algorithm as described in the AlphaFold3 paper. However, if the official AlphaFold3 implementation differs from the main article (e.g., due to typos), we prioritize the official repository's implementation.

NOTE: If you want to use Boltz1's original implementation, please refer to [`src/kfold/model/models/boltz.py`](src/kfold/model/models/boltz.py), which contains Boltz1's original architecture and pre-trained weights.

1. **InputFeatureEmbedder (Algorithm 2)**: This module is **implemented slightly differently** from the official algorithm. The official implementation simply has an `s_input` dimension of `c_s + 32 (residue) + 32 (profile) + 1`, but we add a Linear layer to project the feature dimension to `s_token`. This may cause minor differences in the model size of modules that take `s_input` as input. (Dimensions of `s_input`, `s_init`, `s_trunk`: `c_s=384`)
- AF3/Protenix/OpenFold3: `c_s + 32 + 32 + 1`
- Boltz: `2 * c_s + 32 + 32 + 1` (See point 2)
- Ours: `c_s` (with projection)

2. **RelativePositionEncoding (Algorithm 3)**:
- There is a typo in Algorithm. In line 8, $b_{ij}^\text{same-chain}$ should be corrected to $b_{ij}^\text{diff-entity}$, according to AlphaFold3's official implementation. We note that Boltz does not fix this typo. We use the correct version.
- There is a linear layer after the concatenation of different chain/entity information in the algorithm, it is missing in the official implementation. We follow the official implementation.

3. **AtomAttentionEncoder (Algorithm 3)**: Boltz's output dimension is calculated differently from the actual algorithm (always calculated as `c_token = 2 * c_s`). We explicitly introduce the `c_token` parameter in accordance with the official algorithm's notation.

4. **SampleDiffusion (Algorithm 18)**: This part was additionally introduced in Boltz, featuring a function to minimize the drift term during the coordinate update in the sampling process (Line 11). It also implements features like FK steering. However, for now, we only implement the basic AlphaFold3 algorithm as the simplest implementation.

5. **DiffusionModule (Algorithm 20)**: Due to point 2, the internal dimensions are slightly different.

6. **DiffusionTransformer (Algorithm 23)**: Although the algorithm specification does not include a skip connection, it exists in the actual official implementation.

### Loss Function

1. **MSE Loss (Equation 3)**: We follow the official AlphaFold3 implementation exactly:
    - AF3(paper), Protenix, OpenFold3: $\mathcal{L}_{\text{MSE}} = \frac{1}{3}\underset{l}{\text{mean}}\left(w_l||\vec{\mathbf{x}}_l - \vec{\mathbf{x}}_l^{\text{GT-aligned}}||^2\right)$
    - Boltz1: $\mathcal{L}_{\text{MSE}} = \frac{1}{3}\underset{l}{\text{sum}}\left(w_l||\vec{\mathbf{x}}_l - \vec{\mathbf{x}}_l^{\text{GT-aligned}}||^2\right) / \underset{l}{\text{sum}}{\left(w_l\right)}$

2. **Bond Loss (Equation 5)**: Unlike Boltz1, this has been implemented to be usable in final training (Boltz1 did not use it).

3. **Diffusion Loss (Equation 6)**: Unlike AlphaFold3, all other reference codes calculate the weight differently based on the loss scale. Using the official AlphaFold3 implementation prevents the model from training effectively.
    - AF3 (Paper): $w_{\text{diffusion}} = \left(\hat{t}^2 + \sigma_\text{data}^2\right) / \left(\hat{t} + \sigma_\text{data}\right)^2$
    - Boltz1, Protenix, OpenFold3: $w_{\text{diffusion}} = \left(\hat{t}^2 + \sigma_\text{data}^2\right) / \left(\hat{t} \times \sigma_\text{data}\right)^2$

### Validation Metrics

1. **LDDT Calculation**: We follow the official AlphaFold3 explanation for LDDT calculation during validation. However, **symmetry correction is not yet implemented**.

---

## Implementation of K-Fold

This section describes the additional implementations which are not part of the original AlphaFold3 but are necessary for training the K-Fold Foundation Model.

### Data Processing

1. **Apo Structure Construction**: To train the model to learn the dynamics between **apo** and **holo** states, we need to generate **apo** structures to pair with the existing **holo** structures in the dataset. We use multiple methods to generate these **apo** structures based on the type of biomolecule:
    - Protein: Using ESMFold to predict the **apo** structure.
    - DNA: **Not implemented yet (to be added later).**
    - RNA: **Not implemented yet (to be added later).**
    - Ligand: Using ETKDG to generate free conformers for small molecule ligands.

2. **Multi-Chain Handling**: Since our model is designed to model dynamics between **apo** and **holo** states, our training pipeline cannot defined on single-chain structures. Therefore, we modified the data processing pipeline to handle **multi-chain complex structures** only:
    - New data splits:
        - While preserving AF3's original time split, our training set includes only **complex** structures (more than one chains).
    - Implementing a cropping algorithm that ensures the cropped structure contains multiple chains.
        - Not implemented yet (to be added later).

3. **Multi-Anchor Cropping**: Since our model utilizes apo structures as input, we modified the cropping algorithm to prioritize inter-chain co-folding over local folding.
    - Rationale: The apo input provides a strong structural prior for intra-chain geometry and relative positioning, reducing the need to learn these features from scratch.
    - Method: We extend standard spatial cropping and spatial interface cropping to multi-anchor cropping, which selects multiple spatial centers to form a single input. This allows the model to simultaneously capture disparate regions of the complex, focusing training on interface regions and global chain arrangement.

4. **Optimal Transport Permutation**: To effectively learn the mapping between **apo** and **holo** structures, we implemented a chain permutation algorithm and residue atom swapping algorithm to match the symmetry between the two states.

### Pre-trained Representation Model

1. **Pre-trained Language Model Integration**: We integrated pre-trained language models to enhance the sequence representation of each chain in the complex structure. This replaces the needs of MSA-based representation.
  ```python
  two_linear_mlp = nn.Sequential(
      nn.Linear(esm_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
  )
  s_inputs = s_inputs + two_linear_mlp(lm_embedding)
  ```

2. **Pre-trained Structure Representation Model Integration**: We integrated pre-trained structure representation models to introduce structural priors from the **apo** structure. To feed the multiple structure embeddings (from structure ensemble), we introduce `EnsembleModule`, which is the modified version of `MSAModule`.

### Apo Feature Embedding

1. **Modified AtomAttentionEncoder**: We modified the input feature embedding architecture (`AtomAttentionEncoder`) to incorporate features derived from the **apo** structure:
  - Local structure: Similar to the **ref_pos** embedding in AF3, pairwise offset vectors between atoms in the **apo** structure are computed and embedded to provide local context.
  - Global structure: Pairwise distance maps (token-level) are computed and embedded with RBF/Distogram to provide spatial context.
  - TODO: Currently, only the `InputFeatureEmbedder` uses this modified module. In future, we may want to explore using this module in score model as well.

### Trunk
1. **RBF Embedding for Apo Features**: In the trunk module, we added RBF embedding of pairwise distance maps from the **apo** structure to the pair representation update module. This allows the trunk to effectively utilize global structural information from the **apo** state.

2. **Interformer**: We modified the `Pairformer` module to `Interformer`, which allows bi-directional information flow between single (`s`) and pair (`z`) representations. This is crucial to enrich the evolutionary pre-trained sequence features with interaction context from the pair representation.

3. **EnsembleModule**: We modified the `MSAModule` to `EnsembleModule`, which allows the integration of multiple structure embeddings (from structure ensemble). This is essential for modeling the **flexibility** of the **apo** state, based on multiple pre-trained structure representations from the ensemble of **apo** structures.

4. **MultiStateModule**: TODO: To be added once the AlphaFold2 predicted structures are integrated as additional inputs.


### Structure Module
1. **Diffusion Bridge**: We implemented a diffusion bridge module that learns the dynamics between **apo** and **holo** states. This module is designed to take both **apo** and **holo** structures as input during training, allowing the model to learn the transition dynamics effectively.

2. **Apo-conditioned Diffusion Score Model**: We implemented the diffusion score model which conditions on the **apo** structure features. This allows the model to generate **holo** structures that are consistent with the provided **apo** context.
  - Local structure: Same procedure as in Apo Feature Embedding.
  - Global structure: Inverse distance maps (token-level) are computed to provide attention bias during the diffusion process.

---

## Data construction

See [`scripts/process/rcsb/README.md`](../scripts/process/rcsb/README.md) for instructions on downloading and preparing the RCSB PDB dataset.

### RCSB Training set

Our training dataset contains all PDB entries released before 2022-12-31 (inclusive). The filtering criteria follow those of AlphaFold3 (see SI 2.5.4 of the AlphaFold3 paper) with the following modifications:
- For bioassemblies with more than 20 chains, we save the entire bioassembly and apply on-the-fly pre-cropping during training.

### RCSB Validation set

Our validation set construction started by taking all PDB entries released between 2023-01-01 and 2023-12-31 (inclusive).
We note that this time split is identical to Boltz2's validation set (2024-01-01), as there were no entries released on 2024-01-01.

Closely following AlphaFold3's validation set construction methodology (see SI 5.8 of the AlphaFold3 paper), we implemented the following steps:

1. Take all PDB targets released between 2023-01-01 and 2023-12-31 (inclusive) with a token count <= 2560, chain count <= 1000, and resolution <= 4.5 Å.
2. Remove entries where any chain was filtered out during the PDB data filtering step (see SI 2.5.4 of the AlphaFold3 paper).
3. Select low-homology interfaces using the following criteria:
    1. Collect all interface chain pairs. Interfaces with multi-residue ligands are excluded.
    2. Filter for low-homology interfaces only:
        - Remove the interface if any training target has two chains with sequence identity >= 40% (polymer) or Tanimoto similarity >= 0.85 (ligand) to the involved chains.
        - Remove polymer-ion interfaces if any training target has one polymer chain with sequence identity >= 40% to the involved polymer chain.
        - Remove ligand-ligand interfaces.
    3. Assign interfaces to clusters `(cluster_id1, cluster_id2`) based on the following homology criteria:
        - 40% sequence identity for protein chains.
        - 100% sequence identity for DNA/RNA chains.
        - CCD identity for ligands.
    4. Sample one interface from each cluster.
    5. Retain up to the following limits per interface type:
        - Protein-Protein: 600
        - Protein-DNA: 200
        - Protein-RNA: All
        - Protein-Ligand: 500
        - DNA-DNA: 100
        - DNA-RNA: All
        - DNA-Ligand: 50
        - RNA-RNA: All
        - RNA-Ligand: All
        - Ligand-Ligand: 0
4. Select low-homology monomers using the following criteria:
    1. Take all targets with a single nucleic acid chain. (single polymer chains with multiple ligands are permitted)
    2. Filter out polymers with >= 40% sequence identity to any training target.
5. Take all PDB entries containing the remaining interfaces and monomers from steps 3 and 4.
6. Remove entries with more than 2048 tokens.
7. Sample 1280 PDB entries from the remaining set to construct the validation set.
