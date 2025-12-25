# K-Fold Bio Foundation Co-Folding Model


## \[IMPORTANT\] Notice for Everyone: Development Guidelines

### Project sturcture

Please read [`DEVELOPER_GUIDE.md`](./docs/DEVELOPER_GUIDE.md).

### Formatting and Linting

To ensure code quality and consistency, please run the following commands after [Installation](#installation):

```bash
pre-commit install
```

## Quick Start

### Installation

```bash
pip install -e '.[train,dev]'
pre-commit install
```

### Inference

```bash
# Single-GPU Inference
python scripts/inference.py \
  --config {CONFIG_PATH} \
  --checkpoint {CKPT_PATH} \
  --input {INPUT_YAML_PATH_OR_DIR} \
  --out_dir ./inference_results/ \
  --num_recycles 10 \
  --num_steps 200 \
  --num_samples 5

# Multi-GPU Inference
python scripts/inference_multigpu.py \
  --config {CONFIG_PATH} \
  --checkpoint {CKPT_PATH} \
  --input {INPUT_YAML_PATH_OR_DIR} \
  --out_dir ./inference_results/ \
  --num_recycles 10 \
  --num_steps 200 \
  --num_samples 5 \
  --num_gpus 8
```

- `CONFIG_PATH`: Path to the model configuration file (YAML format).
- `CKPT_PATH`: Path to the trained model checkpoint file.
- `INPUT_YAML_PATH_OR_DIR`: Path to the input single YAML file or directory containing multiple YAML files.
- `OUTPUT_DIR`: Directory where the inference results will be saved (default: `./inference_results/`).
- `num_recycles`: Number of recycling iterations during inference (default: 10).
- `num_steps`: Number of optimization steps during inference (default: 200).
- `num_samples`: Number of samples to generate for each input (default: 5).

#### Single-GPU Inference Example

```bash
python scripts/inference.py \
  --config {CONFIG_PATH} \
  --checkpoint {CKPT_PATH} \
  --input ./examples/queries/casp15_h1106.yaml \
  --out_dir ./out/
```

#### Multi-GPU Inference Example

```bash
python scripts/inference_multigpu.py \
  --config {CONFIG_PATH} \
  --checkpoint {CKPT_PATH} \
  --input ./examples/queries/ \
  --out_dir ./out/
```

**TODO**

### Training

See [`docs/TRAINING_GUIDE.md`](./docs/TRAINING_GUIDE.md) for detailed training instructions.

```bash
python ./scripts/train.py -h

# Run first with debug mode
python ./scripts/train.py --config ./configs/train-af3-tiny.yaml --debug

# If you want to skip validation, use --skip_val
python ./scripts/train.py --config ./configs/train-af3-tiny.yaml --debug --skip_val

# If everything works well, run full training
python ./scripts/train.py --config ./configs/train-af3.yaml --wandb --num_gpus ...
```

### Evaluation

See [`docs/EVALUATION.md`](./docs/EVALUATION.md) for detailed evaluation instructions.

