# Development Note

Author: Seonghwan Seo (Prof. Woo Youn Kim's Lab)

## Sections:
- [Reference](#reference)
- [Implementation of AlphaFold3 Algorithms](#implementation-of-alphafold3-algorithms)
- [K-Fold Implementation](#implementation-for-k-fold-foundation-model)
- [Not Yet Implemented](#not-yet-implemented)
- [To Be Changed](#to-be-changed)

## Reference

In developing this codebase, we referred to the following repositories:
- [AlphaFold3](https://github.com/google-deepmind/alphafold3)
- [Boltz](https://github.com/jwohlwend/boltz)
- [Protenix](https://github.com/bytedance/Protenix)
- [OpenFold3](https://github.com/aqlaboratory/openfold-3)

In particular, we started from Boltz's implementation.
- NOTE: Boltz1's processed data and training code are publicly available, but Boltz2's training code is not, so implementation is based on Boltz1.

## Implementation of AlphaFold3 Algorithms

This section explains how we implement AlphaFold3 algorithm and compares it to Boltz1 or other models, highlighting the differences.

### Data Processing

1. We started from Boltz1's data pre-processed dataset. We re-featurized the data to match our model's input format. (`TokenizedStructure`)
2. Boltz1's Cropping algorithm is implemented differently from AlphaFold3. For faster implementation, we use the Boltz1 implementation.

### Architecture

Compared to the AlphaFold3 article, Boltz1 modified some layers and architectures. In this repository, we revert these modifications to match the original algorithm as described in the AlphaFold3 paper. However, if the official AlphaFold3 implementation differs from the main article (e.g., due to typos), we prioritize the official repository's implementation.

1. **InputFeatureEmbedder (Algorithm 2)**: This module is **implemented slightly differently** from the official algorithm. The official implementation simply has an `s_input` dimension of `c_s + 32 (residue) + 32 (profile) + 1`, but we add a Linear layer to project the feature dimension to `s_token`. This may cause minor differences in the model size of modules that take `s_input` as input. (Dimensions of `s_input`, `s_init`, `s_trunk`: `c_s=384`)
- AF3/Protenix/OpenFold3: `c_s + 32 + 32 + 1`
- Boltz: `2 * c_s + 32 + 32 + 1` (See point 2)
- Ours: `c_s` (with projection)

2. **RelativePositionEncoding (Algorithm 3)**: There is a typo in Algorithm. In line 8, $b_{ij}^\text{same\_chain}$ should be corrected to $b_{ij}^\text{diff\_entity}$, according to AlphaFold3's official implementation. We note that Boltz does not fix this typo. We use the correct version.

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

## Implementation for K-Fold Foundation Model.

This section describes the additional implementations which are not part of the original AlphaFold3 but are necessary for training the K-Fold Foundation Model.

### Data Processing

1. **Apo Structure Construction**: To train the model to learn the dynamics between **apo** and **holo** states, we need to generate **apo** structures to pair with the existing **holo** structures in the dataset. We use multiple methods to generate these **apo** structures based on the type of biomolecule:
    - Protein: Using ESMFold to predict the **apo** structure.
    - DNA: Not implemented yet (to be added later).
    - RNA: Not implemented yet (to be added later).
    - Ligand: Using ETKDG to generate free conformers for small molecule ligands.

2. **Multi-Chain Handling**: Since our model is designed to model dynamics between **apo** and **holo** states, our training pipeline cannot defined on single-chain structures. Therefore, we modified the data processing pipeline to handle **multi-chain complex structures** only:
    - New data splits:
        - While preserving AF3's original time split, our training set includes only **complex** structures (more than one chains).
    - Implementing a cropping algorithm that ensures the cropped structure contains multiple chains.
        - Not implemented yet (to be added later).

### Representation Model
1. **Pre-trained Language Model Integration**: We integrated pre-trained language models to enhance the sequence representation of each chain in the complex structure. This replaces the needs of MSA-based representation.
2. **Pair-wise Representation**: Not yet implemented (to be added later).

### Diffusion Module
1. **Diffusion Bridge**: We implemented a diffusion bridge module that learns the dynamics between **apo** and **holo** states. This module is designed to take both **apo** and **holo** structures as input during training, allowing the model to learn the transition dynamics effectively.

## Not Yet Implemented

The following items require future implementation.

### Data Processing

1. **Symmetry**: Addition of symmetry information is required for **LDDT** calculation during the validation process.
2. **Pocket Conditioning**: Boltz1 utilizes Pocket conditioning during training (Implementation required).
3. **Apo Perturbation**: To be added once the Apo perturbation module is complete.

### Model Training

1. **Validation**: Implementation of symmetry calculation and **LDDT** metric is required.

### Benchmark

1. **mmCIF Writer**: **PDB** writing code is currently available, but **mmCIF** implementation is needed.
2. **Evaluation Metrics**: Installation of evaluation tools and script writing.

## To Be Updated

The following items are scheduled for future modification.

1. **Cropping Algorithm Planning**: **A new cropping algorithm** must be implemented when training the Diffusion bridge model for `apo to holo` (e.g., ensuring the cropped structure always contains two or more chains).

2. **Atom Layout Change**: Boltz represents atom features as a dense feature (`f_atom: [N_allatom, ...]]`). However, this requires a `scatter` operation in the mapping between tokens and atoms, which Boltz handles by performing a `matmul` operation using an `[N_allatom, Ntoken]` one-hot vector. We plan to represent it as a **sparse** matrix (`f_atom: [Ntoken, 24]`), consistent with AlphaFold3, Protenix, and OpenFold3, to increase computational efficiency.
