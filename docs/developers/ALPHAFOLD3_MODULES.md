# AlphaFold3 Implementation

Author: Seonghwan Seo (Prof. Woo Youn Kim's Lab)

## Mapping of Algorithms to Code

### Layer Implementations
The directory [`src/kfold/model/layers/alphafold3`](../../src/kfold/model/layers/alphafold3) contains the implementation of the modules described in the AlphaFold3 paper.

Below is a mapping of the algorithms presented in the paper to their corresponding classes and files in the codebase.
Most implementations are adapted from the [Boltz](https://github.com/jwohlwend/boltz), MIT licensed.


| Section | Algorithm | Class | File |
|---------|-----------|------| -----|
| **3.1** Input Embeddings  | Algorithm 2  | InputFeatureEmbedder | `input_encoder.py` |
|                           | Algorithm 3  | RelativePositionEncoding | `embeddings.py` |
| **3.2** Atom attention    | Algorithm 5  | AtomAttentionEncoder | `atom_transformer.py` |
|                           | Algorithm 6  | AtomAttentionDecoder | `atom_transformer.py` |
|                           | Algorithm 7  | AtomTransformer | `atom_transformer.py` |
| **3.3** MSA Module        | Algorithm 8  | PLMModule (K-Fold mod) | `src/kfold/model/layers/kfold/plm_module.py` |
|                           | Algorithm 9  | OuterProductMean | N/A (Replaced by PLM) |
|                           | Algorithm 10 | MSAPairWeightedAveraging | N/A (Replaced by PLM) |
|                           | Algorithm 11 | Transition | `transition.py` |
| **3.4** Triangle updates  | Algorithm 12 | TriangleMultiplicationOutgoing | `src/kfold/model/layers/primitives/triangle_multiplication.py` |
|                           | Algorithm 13 | TriangleMultiplicationIncoming | `src/kfold/model/layers/primitives/triangle_multiplication.py` |
|                           | Algorithm 14 | TriangleAttentionStartingNode | `src/kfold/model/layers/primitives/triangle_attention.py` |
|                           | Algorithm 15 | TriangleAttentionEndingNode | `src/kfold/model/layers/primitives/triangle_attention.py` |
| **3.5** Template module   | Algorithm 16 | TemplateEmbedder | N/A |
| **3.6** Pairformer stack  | Algorithm 17 | PairformerStack | `pairformer.py` |
| **3.7** Diffusion Module  | Algorithm 19 | CenterRandomAugmentation | `utils.py` |
|                           | Algorithm 20 | DiffusionStack | `diffusion.py` |
|                           | Algorithm 21 | DiffusionConditioning | `diffusion.py` |
|                           | Algorithm 22 | FourierEmbedding | `embeddings.py` |
|                           | Algorithm 23 | CachedGlobalTransformerStack | `diffusion_transformer.py` |
|                           | Algorithm 24 | AttentionPairBias | `attention_pair_bias.py` |
|                           | Algorithm 25 | ConditionedTransitionBlock | `diffusion_transformer.py` |
|                           | Algorithm 26 | AdaLN | `src/kfold/model/layers/primitives/normalization.py` |
| **3.7** Confidence Head   | Algorithm 31 | ConfidenceHead | `confidence.py` (TODO) |


### Loss Implementations

The directory [`src/kfold/training/loss`](../../src/kfold/training/loss) contains the implementation of the loss functions.

Below is a mapping of the algorithms of loss functions.
| Section | Algorithm | Class | File |
|---------|-----------|------| -----|
| **3.7** Diffusion Module            | Equation 2-4  | WeightedMSELoss | `diffusion.py` |
|                                     | Equation 5  | BondLoss | `diffusion.py` |
|                                     | Algorithm 27  | SmoothLDDTLoss | `diffusion.py` |
|                                     | Algorithm 28  | weighted_rigid_align | `src/kfold/utils/geometry/rigid_align.py` |
| **4.3** Model confidence prediction | Equation 8-9  | PLDDTLoss | `confidence.py` (TODO)|
|                                     | Algorithm 29  | expressCoordinatesInFrame | `confidence.py` (TODO) |
|                                     | Algorithm 30  | computeAlignmentError | `confidence.py` (TODO)|
|                                     | Equation 10-11  | PAELoss | `confidence.py` (TODO)|
|                                     | Equation 12-13  | PDELoss | `confidence.py` (TODO)|
|                                     | Equation 14  | ResolvedLoss | `confidence.py` (TODO)|


## Modifications from the Original Paper

### Out dimension of InputFeatureEmbedder (Algorithm 4)

Modified by Seonghwan Seo.

In the original paper, the hidden dimension of `s_inputs`, which is from the `InputFeatureEmbedder` (Algorithm 2), is `C_s + C_res + C_profile + 1`:

```python
# C_s: 384 (single representation channel)
# C_res: 32 (residue type one-hot)
# C_profile: 128 (MSA profile)

class InputFeatureEmbedder(nn.Module):
  def forward(self, feat):
    a, _, _, _ = AtomAttentionEncoder(feat, None, None, None)  # [..., Ntoken, C_s]
    s_inputs = torch.cat(
      [a, feat.restype, feat.profile, feat.deletion_mean], dim=-1,
    )  # [..., Ntoken, C_s + C_res + C_profile + 1]
    return s_inputs
```

However, in our implementation, we assert that `s_inputs` is already projected to the hidden dimension `C_s` (384) in `InputFeatureEmbedder`:

```python
class InputFeatureEmbedder(nn.Module):
  def forward(self, feat):
    a, _, _, _ = AtomAttentionEncoder(feat, None, None, None)  # [..., Ntoken, C_s]
    s_inputs = torch.cat(
      [a, feat.restype, feat.profile, feat.deletion_mean], dim=-1,
    )  # [..., Ntoken, C_s + C_res + C_profile + 1]
    s_inputs = self.s_projection(s_inputs)  # [..., Ntoken, C_s]
    return s_inputs
```

AlphaFold3 uses JAX, which does not require input dimensions for model initialization, so the original paper may have omitted this projection step.
However, in PyTorch, this projection can help unify hidden dimensions between different tensors such as `s_inputs`, `s_init`, and `s_trunk`.
