# AlphaFold3 Implementation

This directory contains the implementation of the layers described in the AlphaFold3 paper.
Below is a mapping of the algorithms presented in the paper to their corresponding classes and files in the codebase.
Most implementations are adapted from the [Boltz](https://github.com/jwohlwend/boltz), MIT licensed.


| Section | Algorithm | Class | File |
|---------|-----------|------| -----|
| **3.1** Input Embeddings  | Algorithm 3  | RelativePositionEncoding | `embeddings.py` |
| **3.2** Atom attention    | Algorithm 5  | AtomAttentionEncoder | `transformers.py` |
|                           | Algorithm 6  | AtomAttentionDecoder | `transformers.py` |
|                           | Algorithm 7  | AtomTransformer | `transformers.py` |
| **3.3** MSA Module        | Algorithm 8  | MsaModule | `msa_module.py` |
|                           | Algorithm 9  | OuterProductMean | `msa_module.py` |
|                           | Algorithm 10 | MSAPairWeightedAveraging | `msa_module.py` |
|                           | Algorithm 11 | Transition | `primitives.py` |
| **3.4** Triangle updates  | Algorithm 12 | TriangleMultiplicationOutgoing | `triangular_update/` |
|                           | Algorithm 13 | TriangleMultiplicationIncoming | `triangular_update/` |
|                           | Algorithm 14 | TriangleAttentionStartingNode | `triangular_update/` |
|                           | Algorithm 15 | TriangleAttentionEndingNode | `triangular_update/` |
| **3.5** Template module   | Algorithm 16 | TemplateEmbedder | N/A |
| **3.6** Pairformer stack  | Algorithm 18| PairformerStack | `pairformer.py` |
| **3.7** Diffusion Module  | Algorithm 19 | CenterRandomAugmentation | `primitives.py` |
|                           | Algorithm 20 | DiffusionModule | `diffusion.py` |
|                           | Algorithm 21 | DiffusionConditioning | `diffusion.py` |
|                           | Algorithm 22 | FourierEmbedding | `diffusion.py` |
|                           | Algorithm 23 | DiffusionTransformer | `transformers.py` |
|                           | Algorithm 24 | AttentionPairBias | `transformers.py` |
|                           | Algorithm 25 | ConditionedTransitionBlock | `transformers.py` |
|                           | Algorithm 26 | AdaLN | `primitives.py` |
