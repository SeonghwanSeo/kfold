# K-Fold Python API

Inference requires a CUDA GPU.
See the [inference guide](inference.md) for installation requirements, input formats, and output descriptions.

## Running predictions

Save the [input example](inference.md#input-format) as `query.yaml`, then load a model and reuse the runner across queries:

```python
from kfold.inference.query import Query
from kfold.inference.runner import KFoldRunner
from kfold.model import KFold

model = KFold.from_pretrained(
    device="cuda",
    use_struct_encoder=True,  # Disable to reduce memory use
    use_rna_encoder=True,  # Disable only if queries contain no RNA.
)
runner = KFoldRunner(
    model,
    lazy_load=True,  # Defer loading AtlasFold until needed.
)
query = Query.load("query.yaml")

result = runner.fold(
    query,
    seed=1,  # Seed for generation.
    num_apos=1,  # Apo structures per protein entry without supplied structures.
    num_samples=5,  # Predictions per seed.
    num_recycles=10,  # Number of recycling iterations.
    num_steps=100,  # Number of diffusion steps.
    return_embeddings=False,  # Return single and pair embeddings.
    return_trajectory=False,  # Return diffusion trajectories.
    return_distogram=False,  # Return distogram logits.
)
result.save("predictions/test/test_seed-1/")
```

`fold()` prepares apo and prior structures and returns a `FoldingResult` with CPU NumPy arrays.
It leaves the input query unchanged.

`result.save(directory)` writes directly into the supplied directory and saves raw confidence NPZ files by default; pass `save_confidence=False` to omit them.
Completion markers, ranking CSVs, and top-level best-prediction copies are managed by the CLI.

## Constructing queries

Use `Query.from_dict(data, base_dir=...)` for the same schema as YAML/JSON files.
`base_dir` resolves relative structure paths; `Query.load(path)` uses the query file's parent directory.

Typed objects are also available:

```python
from kfold.inference.query import LigandSequence, ProteinSequence, Query

query = Query(
    name="test",
    sequences=[
        ProteinSequence(id=["A"], sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL"),
        LigandSequence(id=["B"], ccd=["ATP"]),
    ],
)
```

Direct constructors require lists for `id` and ligand `ccd`, and lists of ID pairs for `ProteinPair.id`, such as `[["H", "L"]]`.
Supplied `apo` and `prior` values must be lists of absolute `pathlib.Path` objects pointing to existing files.
Modifications use `Modification(residue_index=4, ccd="SEP")`; bonds use `Bond(atom1=("A", 20, "NZ"), atom2=("C", 1, "C08"))`.

## Options

### Model loading

`KFold.from_pretrained()` returns a frozen model in evaluation mode.

| Argument | Default | Description |
| --- | --- | --- |
| `pretrained_model_name_or_path` | `"SeonghwanSeo/kfold"` | Hugging Face model repository or local directory containing `config.yaml` and `weights/kfold.pth`. |
| `device` | `"cuda"` | Device for the model, such as `"cuda:1"`; the inference runner requires CUDA. |
| `cache_dir` | `None` | Download cache for model and encoder weights; uses the Hugging Face default when omitted. |
| `use_struct_encoder` | `True` | Load the pretrained protein structure encoder. Set to `False` to reduce memory use; apo coordinates are still used. |
| `use_rna_encoder` | `True` | Load the pretrained RNA sequence encoder. Set to `False` only when queries contain no RNA. |

### Runner initialization

`KFoldRunner()` can be reused across queries and seeds.

| Argument | Default | Description |
| --- | --- | --- |
| `model` | Required | A loaded `KFold` model on a CUDA device. |
| `ccd` | `None` | Custom `CCD` object; loads the release CCD when omitted. |
| `lazy_load` | `False` | Defer loading AtlasFold prediction models onto the GPU until needed. The example above enables this. |
| `cache_dir` | `None` | Download cache for CCD and AtlasFold assets. |
| `share_atlaslm` | `True` | Reuse K-Fold's AtlasLM for apo generation. If `False`, the apo samplers share a separately loaded AtlasLM. |
| `verbose` | `True` | Emit INFO logs for initialization and prediction stages. |

Pass `cache_dir` to both `KFold.from_pretrained()` and `KFoldRunner()` to use the same custom cache throughout.
Release assets are downloaded during runner initialization even with `lazy_load=True`.

### Prediction

`runner.fold()` predicts one query with one seed.
Sampling defaults match the CLI.

| Argument | Default | Description |
| --- | --- | --- |
| `query` | Required | A `Query` object. |
| `seed` | Required | Positive integer seed for input preparation and prediction. |
| `num_apos` | `1` | Generated apo structures per protein entry without supplied structures; 1–5. |
| `num_samples` | `5` | Predictions per seed. |
| `num_recycles` | `10` | Number of model recycling iterations. |
| `num_steps` | `100` | Number of diffusion steps. |
| `return_embeddings` | `False` | Include single and pair embeddings in `result.embeddings`, shared across samples. |
| `return_trajectory` | `False` | Include per-sample diffusion trajectories in `result.trajectory`. |
| `return_distogram` | `False` | Include shared distance logits and metadata in `result.distogram`. |

Sampling counts must be positive.
See [apo structures and priors](inference.md#apo-structures-and-priors) for how supplied and generated structures are used.

### Saving results

`result.save()` always writes prediction structures and confidence-summary JSON files.
Matching files are overwritten; other files are retained.

| Argument | Default | Description |
| --- | --- | --- |
| `out_dir` | Required | Destination directory, created if needed. |
| `save_query` | `True` | Save `query.json` and prepared apo and prior PDB files. |
| `save_confidence` | `True` | Save per-sample pLDDT, PAE, and PDE arrays as NPZ. |
| `save_embeddings` | `False` | Save embeddings when returned by `fold()`. |
| `save_distogram` | `False` | Save distogram arrays when returned by `fold()`. |
| `save_trajectory` | `False` | Save per-sample trajectories as mmCIF when returned by `fold()`. |

## Accessing optional arrays

Enable optional outputs when folding and again when saving:

```python
result = runner.fold(
    query,
    seed=1,
    return_embeddings=True,
    return_distogram=True,
)
print(result.embeddings["z"].shape)
result.save("predictions/test/", save_embeddings=True, save_distogram=True)
```

Single embeddings describe individual tokens; pair embeddings describe token pairs.
These arrays are shared across prediction samples and have padding removed:

| Key | Shape | Dtype |
| --- | --- | --- |
| `s_inputs` | `(num_tokens, single_channels)` | FP16 |
| `s_lm` | `(num_tokens, lm_channels)` | FP16 |
| `z` | `(num_tokens, num_tokens, pair_channels)` | FP16 |

Use `return_trajectory=True` with `save_trajectory=True` for trajectories.
