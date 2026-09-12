# K-Fold Inference Guide

K-Fold accepts YAML or JSON queries describing proteins, DNA, RNA and ligands.
Run `kfold predict` on a CUDA GPU. It generates missing apo structures and predicts
each requested query/seed through `KFoldRunner.fold()`. Use `--input` for a query
file or a directory of query files, and `--out-dir` for results.

AtlasFold is the default tool for generating missing protein apo structures
(AtlasFold-M for protein pairs). You can also use supplied apo structures
predicted by AlphaFold2 or determined experimentally by setting the `apo` field.
Supplied apo structures are used without generating replacements.

## Input format

Save a query as `query.yaml`:

```yaml
name: example
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL
```

See [examples](../examples) for antibody–antigen, protein–RNA and
protein–ligand inputs. YAML and JSON use the same fields. Give each query a unique
`name`; if omitted, the filename stem is used. Chain IDs must be unique across
the query.

### Proteins

Use `protein` entries under `sequences`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | Chain ID. A list such as `[A, B]` creates copies. |
| `sequence` | `str` | Amino-acid sequence. |
| `modifications` | `dict[int, str]` | Optional 1-based residue index to CCD code mapping. |
| `apo` | `str` or `list[str]` | Optional apo structure paths; missing apos are generated. |
| `prior` | `str` or `list[str]` | Optional separate prior ensemble; defaults to the supplied apo structures. |

To use your own apo structures, add `apo: structures/protein.pdb` or a list of
paths to the protein entry. Paths may be absolute or relative to the working
directory or query file.

### Protein pairs

Use `multimer_sequences` for two proteins whose apo structure contains both
chains, such as a Fab heavy/light pair:

```yaml
multimer_sequences:
  - protein:
      id: H:L
      sequence: EVQLVESGG:DIQMTQSP
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | Pair IDs such as `H:L`; `[H:L, M:N]` creates two copies. |
| `sequence` | `str` | Two sequences separated by `:`. |
| `modifications1`, `modifications2` | `dict[int, str]` | Optional residue modifications for each component. |
| `apo` | `str` or `list[str]` | Optional two-chain structure paths; missing apos are generated. |
| `prior` | `str` or `list[str]` | Optional separate two-chain prior ensemble; defaults to the supplied apo structures. |

Each supplied structure must contain exactly two protein chains in the same
order as the input sequences. Ordinary proteins and pairs can appear together
in `sequences` and `multimer_sequences`.

### DNA and RNA

Use `dna` or `rna` entries under `sequences`. Neither accepts `apo` or `prior`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | One or more chain IDs. |
| `sequence` | `str` | DNA uses A/C/G/T; RNA uses A/C/G/U. |
| `modifications` | `dict[int, str]` | Optional 1-based residue index to CCD code mapping. |

### Ligands

Use `ligand` entries under `sequences`, specifying exactly one of `smiles` or `ccd`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` or `list[str]` | One or more chain IDs. |
| `smiles` | `str` | A SMILES representation. |
| `ccd` | `str` or `list[str]` | One CCD code or a multi-residue CCD sequence. |

### Covalent bonds

The optional top-level `bonds` field connects atoms using
`[chain_id, one_based_residue_index, atom_name]`:

```yaml
bonds:
  - [[A, 20, NZ], [C, 1, C08]]
```

## Prediction workflow

From the repository root, run the same query and inference settings as
[`test.py`](../test.py):

```bash
kfold predict --input examples/8and.yaml --out-dir tmp/tests/ \
  --seed 42 --num-apos 3 --save-confidence
```

This predicts five samples for the two identical protein chains in
`examples/8and.yaml`, using seed 42 and three apo generation groups. AtlasFold
uses seeds 421, 422 and 423, yielding three apo candidates and fifteen prior
candidates for the entry shared by chains A and B. `--save-confidence` matches
the default `result.save()` behavior in `test.py`.

The CLI saves to `tmp/tests/8and/8and_seed-42/`; `test.py` saves directly to
`tmp/tests/8and/`. To use the script wrapper, replace `kfold` with
`python run_kfold.py` in the command above.

`predict` is the only command. Each GPU process loads KFold and CCD once, then
calls `fold(query, seed=...)` for each assigned job. AtlasFold heads remain loaded
for subsequent calls and share AtlasLM with KFold by default. Multi-query inputs
run sequentially within each GPU process. Ordinary proteins use AtlasFold;
protein pairs use AtlasFold-M.

For every protein or multimer entry without supplied apos, `--num-apos N` generates
N groups of five AtlasFold candidates. The generation seeds are
`kfold_seed * 10 + 1` through `kfold_seed * 10 + N`. The highest-ranked candidate
from each group becomes apo; all candidates become priors. This budget applies
only to generation: all supplied apo models are used.

If apo is supplied and prior is omitted, apo also serves as prior. An explicit
prior requires a supplied apo. The query's source data is preserved across calls,
so each seed gets independent preparation.

### Structure paths

Apo and prior fields accept a path or list of paths. Every model in each file is
used in file order, followed by subsequent files in list order. PDB and mmCIF
ensembles are supported. For example:

```json
{
  "apo": ["structures/protein.pdb"],
  "prior": ["structures/prior.cif"]
}
```

Paths are tried as supplied first (relative to the current working directory),
then relative to the query file. Supplied structure sequences are aligned to the
query using an ungapped overlap; query positions outside the overlap have no
supplied coordinates. For ordinary proteins, the first subchain of each model
is used; protein pairs use both chains in input order.

Preparation keeps coordinates and encoded structure tokens separate from the
query's source paths. Saved apo and prior files are PDB ensembles with relative
references in `query.json`.

### Output directories

```text
tmp/tests/8and/8and_seed-42/
  query.json
  apo/
    seq-0-apo.pdb
    seq-0-prior.pdb
  8and_seed-42_sample-0_model.cif
  8and_seed-42_sample-0_confidence.json
  8and_seed-42_sample-0_confidence.npz
  ...
```

Sample indices run from 0 to 4 with the default `--num-samples 5`. The raw
confidence NPZ files contain `plddt`, `pae`, and `pde` and are written only when
`--save-confidence` is set; structure mmCIF and confidence-summary JSON files
are always written.

`seq-{i}` uses the zero-based position in `sequences`; multimer entries use
`multimer-{i}` with their position in `multimer_sequences`. Both supplied and
generated structures are saved from prepared PDB text. Move the entire seed
directory to preserve the relative references in `query.json`.

Existing nonempty target/seed directories cause an error. `--overwrite` replaces
those directories after inference succeeds, before saving the new result. It
removes old outputs, including files from previous sample counts or save options.
Directories containing source query/structure files cannot be overwritten.
To extend a run, request only the additional seeds; existing seed directories
are left alone. There is no automatic resume or preparation cache in this CLI.

### Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `-i`, `--input` | Required | Query JSON/YAML file or directory of query files. |
| `-o`, `--out-dir` | Required | Root directory for target/seed outputs. |
| `--seed` | `1` | One or more unique positive KFold seeds. |
| `--num-apos` | `1` | AtlasFold generation groups per entry without supplied apos. |
| `--num-samples` | `5` | Diffusion samples per query/seed. |
| `--num-recycles` | `10` | Trunk recycle count. |
| `--num-steps` | `100` | Diffusion step count. |
| `--num-gpus` | `1` | Independent GPU processes. |
| `--cache-dir` | HF default | Download cache for models and CCD. |
| `--dry-run` | `False` | Check query schema, paths, and output conflicts; list jobs without loading models. |
| `--overwrite` | `False` | Replace requested target/seed directories. |
| `--save-confidence` | `False` | Save confidence scores as NPZ. Summary JSON is always saved. |
| `--save-embeddings` | `False` | Return and save unpadded single/pair embeddings as FP16 NPZ. |
| `--save-distogram` | `False` | Compute and save distogram arrays as NPZ. |
| `--save-trajectory` | `False` | Compute and save per-sample trajectories as mmCIF. |

`--dry-run` does not download models or CCD, parse structure contents, or run
learned encoding. Those checks happen during inference. It does not create or
modify output directories.

The CLI loads the default pretrained KFold release. For multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 kfold predict --input examples/ --out-dir results/ \
  --seed 1 2 3 4 --num-gpus 2 --num-apos 3
```

The directory scan includes immediate `.json`, `.yaml`, and `.yml` files. Query
names must be unique and usable as directory names. GPU workers receive independent
query/seed jobs, and a worker failure stops the command with a nonzero exit status.

## Python API

```python
from kfold.model import KFold
from kfold.inference.query import Query
from kfold.inference.runner import KFoldRunner

model = KFold.from_pretrained("SeonghwanSeo/kfold")
runner = KFoldRunner(model)
query = Query.load("examples/8and.yaml")
result = runner.fold(query, seed=42, num_apos=3)
result.save("tmp/tests/8and/")
```

`fold()` performs apo generation, learned structure encoding, CPU input processing,
and KFold inference. It returns a CPU `FoldingResult`. Use `fold_input(item, seed)`
when you already have an unbatched `InferenceInput`.

`runner.prepare_query(query, seed, num_apos=...)` returns `(apos, priors)` as
candidate lists for each protein entry, followed by each multimer entry, without
changing the query. To construct an `InferenceInput`, pass these lists to
`runner.input_pipeline.build_input(query, seed, num_samples, apos=apos, priors=priors)`.
Use the same seed and sample count when calling `fold_input()`.

`FoldingResult` holds the source query, prepared `apos` and `priors`, coordinates,
confidence summaries and scores, and any requested embeddings, distogram or
trajectory. `result.save(directory)` writes directly into that directory and
includes raw confidence NPZ files by default. Pass `save_confidence=False` to
omit them. For optional arrays, enable the corresponding `return_*` flag in
`fold()` and `save_*` flag in `save()`.

`KFold.from_pretrained(source, device=..., cache_dir=...)` accepts a Hugging Face
repository or a local model directory containing `config.yaml` and
`weights/kfold.pth`. It returns a frozen model in evaluation mode. CCD loading is
available as `kfold.inference.runner.load_ccd(ccd_path=..., cache_dir=...)`.

### Returned array dtypes

`runner.fold(query, seed=42, num_apos=3, return_embeddings=True)` includes
`result.embeddings["single"]` with shape `(num_tokens, single_channels)` and
`result.embeddings["pair"]` with shape `(num_tokens, num_tokens, pair_channels)`.
These are the trunk's `s_inputs` and final `z`, respectively, with padding removed;
they are shared across diffusion samples. Both are CPU NumPy FP16 arrays.

Save them with `result.save(directory, save_embeddings=True)` or CLI
`--save-embeddings`. The file `<name>_seed-<seed>_embeddings.npz` contains `single`
and `pair`. Embeddings are omitted by default.

Distogram logits are also FP16. Coordinates, trajectories, pLDDT, PAE, PDE, and
distogram bin boundaries are FP32; distogram chain/residue indices are INT32.
These casts apply to returned arrays, after model and confidence calculations.
