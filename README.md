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

See [`docs/TRAINING_GUIDE.md`](./docs/TRAINING_GUIDE.md) for detailed training instructions.

```bash
python ./scripts/train.py -h

# Run first with deug mode
python ./scripts/train.py --config ./configs/train-af3-mini.yaml --debug

# If everything works well, run full training
python ./scripts/train.py --config ./configs/train-af3.yaml --wandb --num_gpus ...
```

#### Evaluation

**TODO**
