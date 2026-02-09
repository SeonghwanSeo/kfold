# Recycling Mechanism Analysis

Author: Auto-generated analysis of the K-Fold model recycling flow

## Summary

During **training**, the recycle count is **dynamic** — it changes every step.
During **inference**, the recycle count is **static** — it is fixed at a configured value.

---

## How Recycling Works

Recycling is an iterative refinement mechanism in the trunk module. The trunk's
output representations (`s_hat`, `z_hat`) are fed back as input for the next
iteration, allowing the model to progressively refine its predictions.

All trunk implementations (AF3, Boltz1, KFold, Pairmixer, PairmixerFormer,
PairformerV2) share the same recycling pattern:

```
s_hat, z_hat = 0, 0

for i in range(0, num_recycles + 1):        # total iterations = num_recycles + 1
    s = s_init + Linear(LayerNorm(s_hat))    # add recycled single repr
    z = z_init + Linear(LayerNorm(z_hat))    # add recycled pair repr
    s, z = PairformerStack(s, z)             # run transformer blocks
    s_hat, z_hat = s, z                      # store for next recycle

return s_hat, z_hat
```

**Gradient control**: During training, gradients are enabled **only on the last
iteration** (`i == num_recycles`). All previous iterations run under
`torch.no_grad()` to save memory.

---

## Training: Dynamic Recycle Count

The recycle count **changes every training step**. It is randomly sampled from a
discrete uniform distribution `Uniform{0, 1, ..., num_recycles}` (inclusive on
both ends).

### Mechanism (in `KFoldTrainingModule.__init__`)

```python
# Pre-sample 100,000 recycle counts with a fixed seed
rng = np.random.default_rng(seed=42)
self.recycles_per_step = rng.integers(
    0,
    self.training_config.num_recycles + 1,   # e.g., num_recycles=3 → sample from {0,1,2,3}
    size=100_000,
)
```

### Usage (in `training_step`)

```python
idx = self.global_step % len(self.recycles_per_step)   # cycles through 100k values
num_recycles = int(self.recycles_per_step[idx])         # dynamic per step
out = self(f_input=f_input, num_recycles=num_recycles, ...)
```

### Key Design Points
- **Pre-sampled with fixed seed (42)**: Ensures all GPUs in distributed training
  use the same recycle schedule, preventing deadlocks or gradient mismatches.
- **Wraps around**: After 100,000 steps, the schedule repeats.
- **Range**: If `num_recycles=3`, each step samples from `{0, 1, 2, 3}`, so
  the trunk runs between 1 and 4 total iterations.

---

## Inference: Static Recycle Count

During inference, the recycle count is **fixed** at the configured value and
does not change between samples.

```python
# InferenceConfig default
num_recycles: int = 10

# BaseFoldingModel.sample() default
def sample(self, f_input, num_recycles=10, ...):
    trunk_out = self.trunk(s_inputs, s_init, z_init, f_input, num_recycles)
```

The inference client (`KFoldInferenceClient`) passes the same `num_recycles`
for every input.

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Model Forward Pass                             │
│                                                                       │
│  ┌─────────────────┐                                                  │
│  │  Input Embedder  │──▶ s_inputs, s_init, z_init                     │
│  └─────────────────┘                                                  │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        TRUNK MODULE                            │   │
│  │                                                                │   │
│  │  num_recycles ◄── ┌──────────────────────────────────────────┐ │   │
│  │                   │  TRAINING: Dynamic (changes every step)  │ │   │
│  │                   │    Sampled from Uniform{0..N}            │ │   │
│  │                   │    Pre-sampled array, seed=42            │ │   │
│  │                   ├──────────────────────────────────────────┤ │   │
│  │                   │  INFERENCE: Static (fixed)               │ │   │
│  │                   │    Default = 10, user-configurable       │ │   │
│  │                   └──────────────────────────────────────────┘ │   │
│  │                                                                │   │
│  │  ┌──────────────────────────────────────────────────────────┐  │   │
│  │  │  s_hat = 0, z_hat = 0                                   │  │   │
│  │  │                                                          │  │   │
│  │  │  for i in range(num_recycles + 1):                       │  │   │
│  │  │    ┌──────────────────────────────────────────────────┐  │  │   │
│  │  │    │  Recycle Iteration i                             │  │  │   │
│  │  │    │                                                  │  │  │   │
│  │  │    │  grad = ON  only if (training AND i == last)     │  │  │   │
│  │  │    │  grad = OFF for all earlier iterations           │  │  │   │
│  │  │    │                                                  │  │  │   │
│  │  │    │  s = s_init + Linear(LayerNorm(s_hat)) ◄─┐      │  │  │   │
│  │  │    │  z = z_init + Linear(LayerNorm(z_hat)) ◄─┤      │  │  │   │
│  │  │    │         │                          (recycled)    │  │  │   │
│  │  │    │         ▼                                │       │  │  │   │
│  │  │    │  ┌─────────────────────┐                 │       │  │  │   │
│  │  │    │  │  Pairformer Stack   │                 │       │  │  │   │
│  │  │    │  │  (N transformer     │                 │       │  │  │   │
│  │  │    │  │   blocks)           │                 │       │  │  │   │
│  │  │    │  └────────┬────────────┘                 │       │  │  │   │
│  │  │    │           │                              │       │  │  │   │
│  │  │    │           ▼                              │       │  │  │   │
│  │  │    │  s_hat, z_hat = s, z  ───────────────────┘       │  │  │   │
│  │  │    └──────────────────────────────────────────────────┘  │  │   │
│  │  │                                                          │  │   │
│  │  │  return s_hat, z_hat                                     │  │   │
│  │  └──────────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────┘   │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────┐        │
│  │ Distogram   │  │ Structure Module │  │ Interaction Head  │        │
│  │ Head        │  │ (Diffusion)      │  │ (optional)        │        │
│  └─────────────┘  └──────────────────┘  └───────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Training Step Recycle Selection

```
 global_step:    0     1     2     3     4     5     ...
                 │     │     │     │     │     │
                 ▼     ▼     ▼     ▼     ▼     ▼
 recycles_per_step (pre-sampled with seed=42, size=100,000):
                [2,    0,    3,    1,    3,    2,    ...]
                 │     │     │     │     │     │
                 ▼     ▼     ▼     ▼     ▼     ▼
 trunk runs:    3x    1x    4x    2x    4x    3x    iterations
                (2+1) (0+1) (3+1) (1+1) (3+1) (2+1)

 Gradient:      last  last  last  last  last  last  iteration only
```

---

## Code References

| Component | File | Key Lines |
|-----------|------|-----------|
| Recycle pre-sampling | `src/kfold/training/training_module.py` | `__init__`: `rng.integers(0, num_recycles+1, size=100_000)` |
| Recycle selection per step | `src/kfold/training/training_module.py` | `training_step`: `self.recycles_per_step[self.global_step % len(...)]` |
| Training config | `src/kfold/training/training_module.py` | `TrainingConfig.num_recycles = 3` |
| Inference config | `src/kfold/inference/pl_client.py` | `InferenceConfig.num_recycles = 10` |
| Model forward | `src/kfold/model/models/base.py` | `forward(num_recycles=3)`, `sample(num_recycles=10)` |
| Trunk base class | `src/kfold/model/modules/trunk/base.py` | `forward(num_recycles)` abstract method |
| AF3 Trunk | `src/kfold/model/modules/trunk/af3_trunk.py` | `for i in range(0, num_recycles + 1)` |
| Boltz1 Trunk | `src/kfold/model/modules/trunk/boltz1_trunk.py` | `for i in range(0, num_recycles + 1)` |
| KFold Trunk | `src/kfold/model/modules/trunk/kfold_trunk.py` | `for i in range(0, num_recycles + 1)` |
| Pairmixer Trunk | `src/kfold/model/modules/trunk/pairmixer_trunk.py` | `for i in range(0, num_recycles + 1)` |
| PairformerV2 Trunk | `src/kfold/model/modules/trunk/trunk_with_registry.py` | `for i in range(0, num_recycles + 1)` |
