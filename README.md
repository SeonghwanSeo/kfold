# K-Fold

K-Fold is currently in the **preview stage** and under active development.
The first official release is expected around September 20.
Preprint will be released soon.

K-Fold predicts biomolecular complex structures and binding-induced conformational changes.
Through an apo-to-holo diffusion bridge, K-Fold aims to capture conformational changes in systems such as G protein-coupled receptors (GPCRs).

K-Fold supports proteins, DNA, RNA, small molecules, and chemical modifications without multiple sequence alignments (MSAs).
This repository provides pretrained models, inference and training code, and data preprocessing workflows.

## Model parameters

K-Fold uses pretrained [AtlasLM](https://github.com/SeonghwanSeo/atlasfold) for protein sequence representations and [TriProRep](https://github.com/hsjang0/TriProRep) for protein structure representations.
The parameters for these models and K-Fold are downloaded automatically on first use from [Hugging Face](https://huggingface.co/collections/SeonghwanSeo/k-fold).

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

## Inference

Run predictions from a YAML or JSON query file, or a directory of query files for bulk runs:

```bash
kfold --input examples/8and.yaml --out-dir predictions/ --seed 42
```

By default, K-Fold prepares apo structures with [AtlasFold](https://github.com/SeonghwanSeo/atlasfold), then runs K-Fold predictions.
You can also [provide apo structures](docs/inference.md#providing-apo-structures) from experiments or other prediction tools (e.g., AlphaFold2).

Use `--stage apo` to prepare apo structures only, or `--stage complex` to predict complexes from prepared apos:

```bash
kfold --stage apo --input examples/ --out-dir predictions/ --seed 42
kfold --stage complex --input examples/ --out-dir predictions/ --seed 42
```

See `kfold --help`, the [inference guide](docs/inference.md), or the [Python API guide](docs/python_api.md) for details.

## Training

See the [training guide](docs/training.md) for data preparation, training commands, and configuration.

## Acknowledgements

K-Fold was developed at KAIST as part of the K-Fold initiative supported by the Ministry of Science and ICT (MSIT), Republic of Korea.

Members of Team KAIST are listed below (alphabetical order):

- **Project management:** Hyeongwoo Kim<sup>3,†</sup>
- **K-Fold architecture:** Seokhyun Moon<sup>3,†</sup>, Jun Hyeong Kim<sup>3</sup>, Shinwoo Kim<sup>3</sup>, Minha Park<sup>3</sup>, Jisu Seo<sup>3</sup>, Mingyeong Shin<sup>3</sup>, Wonho Zhung<sup>3</sup>
- **Protein structure encoder:** Hyosoon Jang<sup>1,†</sup>, Taewon Kim<sup>1,†</sup>, Hyunjin Seo<sup>1,†</sup>
- **RNA sequence encoder:** Dongki Kim<sup>1,†</sup>, Jun Hyeong Kim<sup>1,†</sup>, Jinheon Baek<sup>1</sup>, Jaehyeong Jo<sup>1</sup>
- **Training data preparation:** Yeongnam Bae<sup>2,†</sup>, Woosung Jeon<sup>2,†</sup>, Joongwon Lee<sup>3,†</sup>, Junyup Lee<sup>2,†</sup>, Yunsu Shin<sup>2,†</sup>, Eugene Choi<sup>2</sup>, Jeong Hun Choi<sup>2</sup>, Hyeongyu Han<sup>2</sup>, Calvin Samuel<sup>2</sup>
- **Kernel optimization:** Youngchan Kim<sup>4</sup>
- **Engineering lead:** Seonghwan Seo<sup>3,†</sup>
- **Supervision:** Sungsoo Ahn<sup>1</sup>, Dongsu Han<sup>4,1</sup>, Sung Ju Hwang<sup>1</sup>, Ho Min Kim<sup>2</sup>, Woo Youn Kim<sup>3</sup>, Gyu Rie Lee<sup>2</sup>, Byung-Ha Oh<sup>2</sup>

<sup>†</sup> Core contributor; <sup>1</sup> KAIST AI; <sup>2</sup> KAIST Biological Sciences; <sup>3</sup> KAIST Chemistry; <sup>4</sup> KAIST Electrical Engineering.

We thank our collaborators at [HITS](https://hits.ai) for their contributions to K-Fold.

## License

Copyright © 2026 Korea Advanced Institute of Science and Technology (KAIST).

K-Fold source code and model weights are licensed under the [Apache License 2.0](LICENSE).
