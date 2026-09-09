# K-Fold

K-Fold predicts biomolecular complex structures without multiple sequence alignments
(MSAs), using pretrained encoders and an apo-to-holo diffusion model.

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

Describe proteins, DNA, RNA and ligands in YAML or JSON. Prepare protein apo
structures, then run K-Fold. Preparation preserves existing apo inputs and uses
AtlasFold for proteins without supplied apo structures:

```bash
kfold prepare --input examples/ --out-dir prepared/ --seed 1 2 3 4 5
kfold predict --input prepared/ --out-dir results/ --seed 1 2 3 4 5
```

Run both stages with one command:

```bash
kfold pipeline --input examples/ --out-dir results/ --seed 1 2 3 4 5
```

`pipeline` creates an independent input for each target and prediction seed in
`results/<target>/<target>_seed-<seed>/`, containing `query.yaml` and `apo/`.
`--num-apo` (default: 1) controls how many apo candidates are selected per protein
entry. For example, `--seed 5 --num-apo 3` generates five structures each with seeds
51, 52 and 53. The best structure from each seed becomes an apo candidate (3 total),
and all generated structures become prior candidates (15 total), unless a prior
was supplied. Prediction uses seed 5 and saves its samples in the same directory.
`--num-samples` controls only the number of prediction samples in the pipeline.

For generated ensembles, each preparation seed produces five PDB structures by
default. The best structure from each seed becomes an apo candidate, and all 25
become prior candidates.
Prediction uses one apo candidate by default and produces five samples per seed.
Queries with existing apo inputs can go directly to `predict`.

Models and CCD are downloaded automatically. Use `--cache-dir` to select the
Hugging Face cache and `--num-gpus` for multiple GPUs.

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
