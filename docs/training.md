# Training K-Fold

Run the following commands from the repository root.

## Installation

Install the training dependencies with Python 3.11 or later:

```bash
pip install -e '.[train,cuequiv]'
```

## Training data

The following will be released soon:

- Preprocessing code for custom distillation datasets.
- The RCSB training dataset, including AtlasFold predictions from a single seed; the paper uses 40 seeds.

To conduct experimental RCSB training dataset yourself, follow the [RCSB preprocessing guide](../scripts/process/rcsb/README.md).

Set `train.data.data_root` and `train.data.ccd_path` in your training configuration to the prepared dataset directory and CCD file.
The supplied configurations use `rcsb-train` and `rcsb-val` under the data root.

## Running training

Start with a configuration from [configs/train](../configs/train):

```bash
python scripts/train.py --config configs/train/stage_1.yaml
```

Use `--num_gpus` and `--num_nodes` to select training resources, `--out_dir` for logs and checkpoints, and `--wandb` to enable Weights & Biases logging.
Resume a run with `--resume_from_checkpoint /path/to/checkpoint.ckpt`.

For a short debug run on one GPU with no data-loader workers or Weights & Biases logging:

```bash
python scripts/train.py --config configs/train/stage_1.yaml --debug
```

## Configuration

Use `--override` to change configuration values without editing the YAML:

```bash
python scripts/train.py --config configs/train/stage_1.yaml \
  --override train.data.data_root=/path/to/data train.data.ccd_path=/path/to/ccd-train.pkl
```

`--batch_size` sets the batch size per GPU.
`--global_batch_size` sets gradient accumulation to reach the requested effective batch size and must be divisible by `batch_size × num_gpus × num_nodes`.
Without it, the configured accumulation is used.
`--num_batches_per_epoch` sets training batches per GPU per epoch.

Run `python scripts/train.py --help` for all options.
