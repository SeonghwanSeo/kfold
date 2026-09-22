# Ab–Ag sequential prediction: installation and execution

This branch contains the inference code used for the September 22, 2026
Ab–Ag experiment. Query files are distributed separately as
`abag_sequential_inputs_20260922.tar.gz`. They are not required for installing
the package. Full algorithm details are in [sequential_assembly.md](sequential_assembly.md).

## Obtain and install the source

```bash
git clone --branch codex/abag-sequential-apo-ensemble-20260922 --single-branch \
  https://github.com/SeonghwanSeo/kfold.git kfold-abag-sequential
cd kfold-abag-sequential
conda activate kfold
python -m pip install --no-deps -e .
python -m pip check
python -c 'import kfold; print(kfold.__file__)'
```

The `--no-deps` command assumes an existing compatible KFold environment.
For a new environment, create Python 3.11, install a CUDA-enabled PyTorch
compatible with the machine's driver, then `python -m pip install -e '.[cuequiv]'
'atlasfold==1.0.2'` (on one line). The source experiment uses PyTorch
2.10.0+cu128, AtlasFold 1.0.2 and cuEquivariance 0.10.0. The input kit records
the observed package versions; it is not a portable environment lockfile.

The input kit contains `SOURCE_COMMIT.txt`; checkout that commit to reproduce
the released handoff if the branch subsequently advances. The launcher sets
`PYTHONPATH` to this checkout so inference uses its source.

Model weights and CCD are downloaded automatically. They, existing prepared
apos, ground truths, and predictions are not included in the input kit.
Use `CACHE_DIR` to choose the download cache. Strict numerical reproduction
also needs matching weights, environment and the original prepared monomer
apo files; regenerated apos are not guaranteed to be identical.

## Input and modes

Each protein is a separate `protein` entry without `apo` or `prior` paths.
AtlasFold prepares monomers automatically. The `assembly` block specifies
which physical chains KFold assembles before predicting the full complex.
These are not `protein_pair` inputs for AtlasFold-Multimer preparation.

For example, 8CHE first assembles A/B and then predicts E/A/B:

```yaml
assembly:
  stages:
    - id: assemble_A_B
      chains: [A, B]
  selection: confidence_top1
```

| `--conditioning` | ECSI initialization | Trunk apo | Protein structure representation |
|---|---|---|---|
| `prior_only` | Global intermediate top-1 | Original monomers and grouping | Original inputs |
| `prior_and_trunk` | Global intermediate top-1 | Per-final-seed ensemble, shared H/L apo UID | Re-encoded chainwise |
| `prior_and_trunk_multichain` | Global intermediate top-1 | Same ensemble policy and shared UID | Chainwise tokens, distinct chain embeddings, joint structure attention |

Both trunk modes use `conditioning_schema=3` and
`per_final_seed_top1_per_generation_seed`. Each final seed S gets five
intermediate generation seeds `10*S+1` through `10*S+5`. Five predictions per
generation seed yield one confidence top-1 apo for that slot. The H/L chains
share coordinates from the same candidate in each slot. The ECSI prior still
uses the global confidence top-1 across all intermediate candidates.

This aligns the trunk apo selection/grouping policy with AtlasFold-M; it does
not make all conditioning or sampling identical to the AtlasFold-M baseline.
Intermediate predictions run separately for each conditioning mode. The
trunk is recomputed at each stage, and denoising may change H/L geometry.

The multichain mode now follows TriProRep's native complex input convention:
per-chain tokenization, chain embedding IDs 0/1 in input-chain order, positions
restarting at 0 per chain, and joint attention within the selected object. It
uses the existing multi-chain-trained weights. Attention groups remain separate
from chain embedding IDs. The old 512-gap tokenization is no longer used.

`structure_representation_policy` is recorded as
`triprorep_chainwise_tokens_chain_ids_reset_positions_v1`. Use a new multichain
output directory; old gap-policy intermediates/final results cannot resume.
Prepared AtlasFold monomer apos remain reusable. The other two modes retain
their existing input and resume policies.

## Run

Extract the input kit, activate the environment, and obtain GPU allocation.
Preserve the scheduler's `CUDA_VISIBLE_DEVICES`; GPU IDs below are logical
indices within the visible allocation.

```bash
tar -xzf /path/to/abag_sequential_inputs_20260922.tar.gz -C /path/to/data
INPUT=/path/to/data/abag_sequential_inputs_20260922/inputs/sequential_17 \
OUT=/scratch/your_user/abag_sequential GPU_IDS='0 1' \
bash scripts/abag_sequential/run.sh
```

Use `GPU_IDS='0'` for one GPU. Defaults are ten final seeds (1–10), five apos,
five samples per seed, ten recycles, 100 diffusion steps, CPU offload disabled,
and `prior_and_trunk` followed by `prior_and_trunk_multichain`.

The launcher prepares monomer apos once in `OUT/prepared_apo` and reuses
them in both modes. Rerun the unchanged command to resume completed work.
Use a new output directory for changed settings or old schema 1/2 results.

```bash
# Select modes; prior_only is optional.
INPUT=/path/to/inputs/sequential_17 OUT=/scratch/your_user/abag_all \
GPU_IDS='0 1' MODES='prior_only prior_and_trunk prior_and_trunk_multichain' \
bash scripts/abag_sequential/run.sh

# The three large excluded systems are a separate experiment.
INPUT=/path/to/inputs/excluded_large_3 OUT=/scratch/your_user/abag_large \
CPU_OFFLOAD=1 GPU_IDS='0 1' bash scripts/abag_sequential/run.sh
```

The default 17 systems each have one intermediate stage: each trunk mode
generates 250 intermediate plus 50 final structures per system. Both modes
across 17 systems generate 10,200 structures, excluding AtlasFold preparation.
`prior_only` generates 50 intermediate plus 50 final structures per system.
The three large systems have two independent antibody assembly stages each,
so each trunk mode generates 500 intermediate plus 50 final structures per
large system. More GPUs distribute systems; they do not shard one system.

## Inspect outputs

```text
OUT/<mode>/<query>/<query>_model.cif             final confidence top-1
OUT/<mode>/<query>/<query>_confidence.json
OUT/<mode>/<query>/sequential/completed.json
OUT/<mode>/<query>/sequential/apo_policy.json
OUT/<mode>/<query>/sequential/<stage>/selection.json
OUT/<mode>/<query>/sequential/<stage>/parent-seed-S/apo_selection.json
OUT/<mode>/<query>/sequential/final/selection.json
```

`prior_only` has no per-parent trunk apo ensemble layout. Use the final
complex, rather than an intermediate antibody, for Ab–Ag DockQ evaluation.
This handoff does not bundle evaluation data or scripts.

## Validation

This revision replaces the earlier experiment's 512-gap representation with
TriProRep's native chain embedding and per-chain position convention. Existing
running jobs use their frozen old source snapshots; this change takes effect
only in new runs.
The updated policy passed 34 CPU tests, including learned chain-embedding
use, cross-chain attention within an object, separate-object isolation,
per-chain positions, one/five-apo axis routing, and old-policy resume rejection. All 20 delivered queries passed the native parser, and the 17 default
queries passed the ordinary CLI dry-run. The launcher was syntax-checked;
full GPU execution on a recipient's machine has not been validated.

```bash
python -m kfold.cli.main --input /path/to/inputs/sequential_17 \
  --out-dir /scratch/your_user/abag_check --seeds 1 2 3 4 5 6 7 8 9 10 \
  --num-apos 5 --conditioning prior_and_trunk --dry-run
```
