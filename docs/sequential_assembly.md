# Sequential complex priors

Experimental, inference-only feature based on main commit
`5705156d34e87bdf302ee3aa9aaf4d0a385ebaad`, branch
`dev_sequential_prediction` (original worktree: `codex/sequential-complex-prior`).

Predict a subset, select its confidence top-1, then carry its coordinates as
one rigid prior object into the next prediction. For P1–L–P2, the first call
contains only P1 and L; the final call contains all three molecules. Each call
runs structure/sequence embedding, the complete trunk, ECSI and confidence.
The model is loaded once and reused. Intermediate pair/trunk tensors are not
reused, and no training parameters or checkpoint shapes are added.

## Input

Add this block to an existing native JSON/YAML query with A=P1, B=P2, C=ligand:

```yaml
assembly:
  stages:
    - id: bind_p1_ligand
      chains: [A, C]
  selection: confidence_top1
```

The original `sequences`, chemistry, modifications, bonds and protein `apo`
paths remain required and unchanged. A complete final-system stage is appended
automatically. Reverse the binding order by replacing `[A, C]` with `[B, C]`.

Multiple declared stages execute in order. Already constructed objects may be
merged, but may not be partially consumed. Disjoint objects can be created in
separate stages and assembled in the final call. A chain, ligand or existing
protein multimer cannot be duplicated or split across objects. Stages that cut
an explicit covalent bond are rejected. Stage IDs must be unique simple names;
`final` is reserved. Subset references use physical chain IDs, including when
two proteins have identical sequences or share a sequence entry.

## Precisely what is carried forward

By default (`--conditioning prior_only`), only the ECSI prior is updated.
The optional native-path re-encoding mode is described below.
A selected P1–L prediction receives one shared
rotation/translation per prior sample, preserving all internal distances.
Protein and ligand atom identities remain separate: topology, atom order,
chain IDs and model interface masks are not collapsed into a new molecule.
Unaffected chains keep the usual prior generation path. Generated group atoms
are not independently relaxed or replaced with Gaussian coordinates.

The original full-query apo/source choices are resolved once per seed and
mapped into every stage. In the default mode, `apo_coords` and raw structure-token records
continue to come from these original sources. Prior augmentation has a separate
stage-dependent RNG. Torch sampling is reseeded with the declared seed for each
model call, as in ordinary inference.

The grouped coordinates become `f_input.atom.prior_coords`, and hence ECSI
`x_T`. This is an initial/conditioning object, **not a rigid constraint during
denoising**: all atoms may subsequently move, including the P1–L interface.
The implementation first generates normal priors and then overlays group
coordinates before tokenization; this preserves the normal RNG consumption
for unaffected chains. It does not modify the training prior sampler.

Intermediate coordinates are stored as float32 NPZ plus stable
`(original chain ID, 1-based residue index, atom name)` keys. The next stage
uses those keys, not stage-local asym IDs or CIF re-parsing. Missing, duplicate,
extra or nonfinite atoms are errors.

## Execution

Use `scripts/inference_sequential.py`. The ordinary data pipeline rejects an
assembly-bearing query and points to this runner instead of ignoring it.
Existing queries without `assembly` retain their original code path.

```bash
export PYTHONPATH="$SEQUENTIAL_SOURCE/src"
"$KFOLD_PYTHON" "$SEQUENTIAL_SOURCE/scripts/inference_sequential.py" \
  --input "$INPUT" --weight "$WEIGHT" --config "$CONFIG" --ccd "$CCD" \
  --out-dir "$OUTPUT" --dry-run
```

Dry-run validates the plan and builds every stage's ordinary features. It hashes
the supplied checkpoint/config/CCD/apo files but does not load a neural model;
it is **not** a checkpoint-load or GPU forward smoke test. For a real GPU run,
export the same variables and submit `scripts/inference_sequential.sbatch`.
No Slurm job is submitted automatically. Supply a config compatible with the
chosen checkpoint and the pinned main code; strict loading is retained.

Defaults: seeds 1–5, 5 samples per seed, 10 recycles, 100 diffusion steps,
all supplied apo sources. `--num-apo N` caps the apo ensemble. The Slurm example
reserves one GPU, eight CPUs and limits library threads to one. CLI arguments
can be appended to the sbatch invocation, for example `--seed 1 --num-samples 1`
for a small forward test. Full GPU execution requires an ECSI checkpoint and
the normal pretrained-encoder resources required by its model config.

## Ranking and restart

Every seed must produce its complete sample count before it is published as
complete. Select the global maximum of native `complex.ranking_score` across
all seeds/samples, breaking ties by ascending seed then sample. No ground truth
is available to selection. The next stage receives that same selected object
for all of its seeds; each seed creates independently augmented priors.

`--resume` requires the identical source code, configuration, weight, CCD,
apo/prior file hashes, inputs and sampling settings recorded in `manifest.json`.
A lock prevents concurrent writers. Completed seed artifacts are checksum
verified and reused. Failed attempts remain in hidden attempt directories for
diagnosis; only that incomplete seed is recomputed. Corrupted completed results
are rejected. Published stage selections must match on restart.

```text
OUTPUT/
  manifest.json
  QUERY/
    source_choices_seed-N.npz
    bind_p1_ligand/
      seed-N/
        input.json, prior.npz, ecsi_init.npz, timing.json
        QUERY_seed-N_sample-M.cif
        QUERY_seed-N_sample-M_confidences.json
        QUERY_seed-N_sample-M_confidences.npz
        QUERY_seed-N_sample-M_atoms.npz
        complete.json
      selection.json
    final/
      seed-N/...
      selection.json
    candidates.csv
    completed.json
```

Use only `QUERY/final` as the prediction root for downstream accuracy
evaluation; the intermediate systems deliberately have fewer molecules.
Existing CIF and confidence formats are retained. Evaluators with fixed
directory-depth globs may need their input-root adapter adjusted. No OST
evaluation or benchmark-specific contact metric computation is embedded here.

## Experiment and validation

### Optional native-path structure re-encoding

`--conditioning prior_only` remains the default. With
`--conditioning prior_and_trunk`, selected intermediate structures also update
the existing per-molecule structural inputs before the next full trunk call:

- Protein P1′: replace apo coordinates and regenerate atom37 BB/FA token records.
  The learned structure tokenizer is called again on the model device.
- Ligand L′: replace residue-local `ref_pos` using independently augmented
  predicted coordinates and recompute native ligand frames. Atom names,
  chemical bonds, charge, and reference-space IDs are unchanged.
- No P–L shared apo UID, no ligand apo coordinates, and no new P–L pair pathway.
  The P1′–L′ relative pose is retained only in the ECSI prior object.
- Unassembled P2 keeps the original apo ensemble. One selected P1′ is broadcast
  over the existing apo axis; these are not independent predicted apo samples.
- No weights/architecture changes. The full trunk and structural encoders rerun;
  old trunk states are not reused. Nucleic-acid re-encoding is not supported.

The runner records `trunk_conditioning.npz` (apo geometry, reference positions,
and IDs) and `structure_token_ids.npz` alongside each seed's existing artifacts.
Conditioning mode is part of the resume manifest; modes cannot share outputs.
The 21-system 0720-68K prepared experiment is documented at
`/home/icl_hwkim/kfold/cofolding_jobs/20260909_mgbench_sequential_trunk_0720_68k/README.md`.
That frozen runtime retains the original control's checkpoint-compatible legacy
training-only `patch_geometry.py` from `7e75336b93e4cb332da5efdc90d429ea228d2851`.
This branch does not revert the main model's patch head: use a checkpoint-compatible
model configuration/source snapshot, as recorded in the experiment manifest.

Prepare direct, P1-first and P2-first inputs using the same full query,
checkpoint, model config and final 5×5 sampling. Direct uses the ordinary runner
with no assembly block. Compare final confidence-top-1 DockQ, lDDT-PLI, ligand
RMSD and joint success using the same ground truth and evaluator. Report the
extra partial-system model calls and time separately; equal final candidate
count is not equal total compute. Binary molecular-glue predictions can be
incorrect even when a ternary complex is stable, so accuracy improvement is a
hypothesis rather than an assumption.

Run CPU checks with:

```bash
NUMBA_CACHE_DIR=/tmp/kfold-sequential-numba OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src "$KFOLD_PYTHON" -m pytest \
  tests/unit_tests/test_sequential_assembly.py -q
```

The tests exercise real SMILES and protein–ligand–protein feature pipelines,
atom remapping, rigid-object distance preservation, unchanged apo conditioning,
ECSI endpoint consumption, subset validation, complete-candidate selection and
restart/corruption behavior. The orchestration test uses a fake model backend;
it verifies stage composition and call counts but is not evidence of learned
model accuracy or GPU memory viability.
