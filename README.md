# K-Fold

K-Fold is currently in the **preview stage**. Preprint will be released soon.

K-Fold predicts biomolecular complex structures and binding-induced conformational changes.
Through an apo-to-holo diffusion bridge, K-Fold aims to capture conformational changes in systems such as G protein-coupled receptors (GPCRs).

K-Fold supports proteins, DNA, RNA, small molecules, and chemical modifications without multiple sequence alignments (MSAs).
Protein component structures are generated with [AtlasFold](https://github.com/SeonghwanSeo/atlasfold) by default or supplied from other predictors or experiments.
This repository provides pretrained models, inference and training code, and data preprocessing workflows.

## Installation

K-Fold requires Python 3.11 or later.

Install from PyPI:

```bash
pip install 'kfold[cuequiv]'
```

Or install from source:

```bash
git clone https://github.com/SeonghwanSeo/kfold.git
cd kfold
pip install '.[cuequiv]'
```

The `cuequiv` extra installs cuEquivariance kernels for faster inference on NVIDIA GPUs.
Model weights and the chemical component dictionary (CCD) are downloaded automatically on first use.

## Inference

Run prediction on a YAML or JSON query file from the repository root:

```bash
kfold predict --input examples/8and.yaml --out-dir predictions/ --seed 42
```

`--input` accepts a YAML or JSON file, or a directory of query files.
The default run generates missing protein apo structures and predicts five samples, saving structures and confidence scores under `--out-dir`.

Run `kfold predict --help` for all options.
See the [inference guide](docs/inference.md) for input formats, sampling, multiple GPUs, and outputs, or the [Python API guide](docs/python_api.md) for use in Python.

## Training

See the [training guide](docs/training.md) for data preparation, training commands, and configuration.

## License

Copyright © 2026 Korea Advanced Institute of Science and Technology (KAIST).

K-Fold is licensed under the [Apache License 2.0](LICENSE).
