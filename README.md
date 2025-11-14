# K-Fold Bio Foundation Co-Folding Model


## \[IMPORTANT\] Notice for Everyone: Development Guidelines

### Project sturcture

Please read [`DEVELOPER_GUIDE.md`](./docs/DEVELOPER_GUIDE.md).

### Formatting and Linting

To ensure code quality and consistency, please run the following commands after [Installation](#installation):

```bash
pre-commit install
```


### Quick Start

#### Installation

```bash
pip install -e '.[train,dev]'
pre-commit install
```

#### Inference

**TODO**

#### Training

```bash
python ./scripts/train.py ./configs/train-af3.yaml --wandb
```

#### Evaluation

**TODO**
