# Affinity training

Affinity training consumes frozen K-Fold representations from an LMDB cache and
optimizes only the affinity Pairformer readout. The training process does not run
the trunk, diffusion, confidence, or coordinate-prediction modules.

## Configuration contract

Use one of the checked-in presets and override deployment-specific paths on the
command line:

```bash
python scripts/affinity/train_affinity.py \
  --config configs/train-affinity-pocket80k-target.yaml \
  --override \
    train.data.manifest_path=/absolute/path/to/training-manifest.parquet \
    train.data.cache_root=/absolute/path/to/cache-shards \
    train.out_dir=/absolute/path/to/output
```

The resolved config is saved in the run directory before training starts. It is
the source of truth for all experiment behavior, including:

- cache schema, encoding, crop contract, token limits, and entropy policy;
- activity-cliff batch composition and validation batch size;
- affinity architecture, kernels, activation checkpointing, and compilation;
- loss weights and optimizer values;
- devices, precision, step count, and distributed-sampler policy;
- Torch sharing strategy and float32 matrix-multiplication precision;
- W&B routing, checkpoint selection, milestone steps, and performance gates;
- immutable input snapshot lineage.

Custom `AFFINITY_*` environment overrides are intentionally unsupported. Scheduler
rank variables such as `RANK` and `SLURM_PROCID` remain environment-owned because
they identify the launched process rather than change the experiment. Credentials
such as `WANDB_API_KEY` also remain secret-owned and must not be stored in YAML.

Module constants that identify a cache encoding, serialized schema, or crop
contract are compatibility identifiers, not tunable experiment settings. The
corresponding selected identifier is still recorded in the resolved config and
run metadata, while the implementation registry remains in code so unsupported
formats cannot be enabled by editing YAML alone.

## Presets

- `train-affinity-ranking.yaml` reads the full-cross cache and performs a
  distogram-derived runtime crop.
- `train-affinity-pocket80k-target.yaml` requires the direct-load raw cache with
  the target-consensus crop already materialized. Training performs one LMDB
  lookup per system and does not select a pocket or decode Zstd values.

Both presets default to a portable `outputs/affinity` directory. Production jobs
should override input and output paths explicitly and retain the resolved config
with their artifacts.
