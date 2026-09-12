# K-Fold

K-Fold predicts biomolecular complex structures without multiple sequence alignments
(MSAs), using pretrained encoders and an apo-to-holo diffusion model.

The preprint will be available soon.

## Installation

Python 3.11+ and a CUDA GPU are required for prediction.

```bash
git clone https://github.com/SeonghwanSeo/kfold.git
cd kfold
pip install '.[cuequiv]'
```

For development and training:

```bash
pip install -e '.[train,cuequiv,dev]'
pre-commit install
```

## Inference

Describe proteins, DNA, RNA and ligands in YAML or JSON. From the repository root,
run the CLI equivalent of `test.py` with seed 42 and three apo generation groups:

```bash
kfold predict --input examples/8and.yaml --out-dir tmp/tests/ \
  --seed 42 --num-apos 3 --save-confidence
```

This produces five KFold samples in `tmp/tests/8and/8and_seed-42/`.
`--save-confidence` includes the raw confidence arrays saved by default in
`test.py`. AtlasFold uses seeds 421, 422 and 423, producing three apo candidates
and fifteen prior candidates for the protein entry shared by chains A and B.

Each target/seed gets an independent directory under
`<out-dir>/<target>/<target>_seed-<seed>/` containing `query.json`, `apo/`, and
prediction/confidence files. Missing protein apo structures are generated with
AtlasFold. Use `--num-apos` (default: 1) for the number of generation groups per
entry; each group contributes its best candidate as apo and all five candidates
as priors. Supplied apos serve as priors unless a separate prior is supplied.

AtlasFold is the default apo generation tool. You can also supply apo structures
predicted by AlphaFold2 or determined experimentally through the query's `apo`
field. See the [Inference Guide](docs/inference.md#proteins) for the input format.

Each GPU process calls `fold()` sequentially for its assigned target/seed jobs,
retaining the loaded models between calls. Use `--num-gpus` for multiple GPUs and
`--num-samples` for the number of KFold predictions per seed. Pass a directory to
`--input` to process multiple queries, and use `--seed 1 2 3` for multiple seeds.

Nonempty run directories require `--overwrite`, which replaces their contents.
To add predictions, request additional seeds. `--dry-run` checks query files,
paths, and output conflicts without loading models or writing outputs.
Models and CCD are downloaded automatically; `--cache-dir` selects their cache.

You can also run the CLI with `python run_kfold.py` in an environment where
K-Fold is installed.

See the [Inference Guide](docs/inference.md) for input examples, options and the
Python `KFoldRunner` API.

## Training and development

- [Training and dataset preparation](docs/developers/TRAINING_GUIDE.md)
- [Custom distillation datasets](docs/developers/CUSTOM_DATASET.md)
- [Data structures](docs/developers/DATA_STRUCTURE.md)

## License

K-Fold is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
