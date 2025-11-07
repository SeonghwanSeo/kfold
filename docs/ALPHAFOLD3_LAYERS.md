# AlphaFold3 Implementation

The directory [`src/kfold/model/layers/alphafold3`](src/kfold/model/layers/alphafold3) contains the implementation of the modules described in the AlphaFold3 paper.

## Mapping of Algorithms to Code
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
