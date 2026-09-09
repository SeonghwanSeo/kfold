# KFold

KFold predicts biomolecular complex structures without multiple sequence alignments
(MSAs), using pretrained encoders and an apo-to-holo diffusion model.

## Installation

Python 3.11+ and a CUDA GPU are required for prediction.

```bash
git clone https://github.com/wykim-lab/kfold.git
cd kfold
pip install '.[cuequiv]'
```

For development and training:

```bash
pip install -e '.[train,cuequiv,dev]'
pre-commit install
```

## Inference

Describe proteins, DNA, RNA and ligands in YAML or JSON. Generate missing protein
apo structures with AtlasFold, then run KFold:

```bash
kfold prepare --input queries/ --out-dir prepared/ --seed 1 2 3 4 5
kfold predict --input prepared/ --out-dir results/ --seed 1 2 3 4 5
```

Each preparation seed generates five PDB structures by default. The best structure
from each seed becomes an apo candidate, and all 25 become prior candidates.
Prediction uses up to three apo candidates and produces five samples per seed.
Queries with existing apo inputs can go directly to `predict`.

Models and CCD are downloaded automatically. Use `--cache-dir` to select the
Hugging Face cache and `--num-gpus` for multiple GPUs. Completed seeds are skipped;
use `--overwrite` to regenerate them.

You can also run the CLI with `python run_kfold.py` in an environment where
KFold is installed.

See the [Inference Guide](docs/inference.md) for input examples, options and the
Python `KFoldRunner` API.

## Training and development

- [Training and dataset preparation](docs/developers/TRAINING_GUIDE.md)
- [Custom distillation datasets](docs/developers/CUSTOM_DATASET.md)
- [Data structures](docs/developers/DATA_STRUCTURE.md)

## License

KFold is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
