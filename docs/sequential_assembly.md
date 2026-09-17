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

The original `sequences`, chemistry, modifications and bonds remain unchanged.
Protein `apo` paths are optional. The standard apo stage automatically runs
AtlasFold for `protein` entries and AtlasFold-Multimer for `protein_pair`
entries when paths are absent. A complete final-system stage is appended
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
The optional chainwise and multi-chain native-path re-encoding modes are
described below.
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

Use the ordinary `kfold` CLI. It preserves `assembly` while preparing apos,
then routes assembly-bearing queries to the sequential runner. Queries without
`assembly` retain the ordinary direct-prediction path.

```bash
kfold --input "$INPUT" --out-dir "$OUTPUT" \
  --seeds 1 2 3 4 5 --num-apos 5 --dry-run
kfold --input "$INPUT" --out-dir "$OUTPUT" \
  --seeds 1 2 3 4 5 --num-apos 5 \
  --conditioning prior_and_trunk_multichain
```

Dry-run validates the native query, assembly plan, CCD codes and any completed
apo preparation without loading K-Fold. Defaults match ordinary inference: seed
1, 5 samples, 10 recycles and 100 diffusion steps. `--num-apos N` generates N
apo candidates per inference seed; `--share-apo-seeds` prepares one ensemble
shared by all complex seeds. All requested seeds for one assembly query stay on
one GPU because a stage chooses one global confidence top-1 before continuing.

## Ranking and restart

Every seed must produce its complete sample count before it is published as
complete. Select the global maximum of native `complex.ranking_score` across
all seeds/samples, breaking ties by ascending seed then sample. No ground truth
is available to selection. The next stage receives that same selected object
for all of its seeds; each seed creates independently augmented priors.

Rerunning the same command resumes completed stage/seed artifacts after checksum
verification. Sequential settings are recorded in `settings.json`; changed
settings require `--overwrite`. Failed attempts remain in hidden attempt
directories for diagnosis, and only incomplete seeds are recomputed. Published
stage selections must match on restart.

```text
OUTPUT/
  QUERY/
    QUERY_model.cif
    QUERY_confidence.json
    QUERY_summary.csv
    kfold_settings.json
    QUERY_seed-N/query.json, apo/...
    sequential/
      settings.json
      source_choices_seed-N.npz
      bind_p1_ligand/
        seed-N/
          input.json, prior.npz, ecsi_init.npz, timing.json
          trunk_conditioning.npz, structure_token_ids.npz
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

Use `QUERY/QUERY_model.cif` or `QUERY/sequential/final` for downstream accuracy
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

`--conditioning prior_and_trunk_multichain` retains all behavior above and
changes only the protein structure-representation path for a selected object
containing two or more proteins:

- Its protein chains are converted to atom37 in one shared coordinate frame and
  passed to the backbone/full-atom structure tokenizers in one call. A residue
  index gap marks each chain boundary; the backbone tokenizer's spatial KNN can
  therefore observe cross-chain neighbors without creating a peptide bond.
- Those protein chains receive one structure-encoder-only sequence group ID, so
  the pretrained structure encoder can attend across their chain boundary.
  Its rotary position IDs use the same gap-separated residue indices as the
  joint tokenizers, avoiding position collisions between chains.
  Physical `asym_id` values remain unchanged, and the protein sequence encoder
  remains chainwise.
- Ligands never enter the protein structure encoder. Unassembled proteins and
  separate selected objects keep distinct structure-encoder groups.
- Apo geometry, `apo_uid`, ligand reference conformers and the ECSI prior are
  identical to `prior_and_trunk`; this mode does not create a new shared apo
  object. It is an experimental inference path and does not imply that the
  frozen structure encoder was trained on this grouping policy.

The runner records `trunk_conditioning.npz` (apo geometry, reference positions,
and IDs) and `structure_token_ids.npz` alongside each seed's existing artifacts.
The latter includes physical/effective sequence and position IDs (`asym_id`,
`pos_id`, `structure_seq_id`, and `structure_pos_id`), making joint attention
grouping and chain-boundary positions auditable.
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
joint multi-protein tokenization and attention grouping, ECSI endpoint
consumption, subset validation, complete-candidate selection and
restart/corruption behavior. The orchestration test uses a fake model backend;
it verifies stage composition and call counts but is not evidence of learned
model accuracy or GPU memory viability.

## Main integration (2026-09-17)

The release `query.py` and `data_pipeline.py` follow main. The experimental
source-selection/RNG behavior lives in `sequential_query.py`,
`sequential_pipeline.py`, `sequential_dataset.py` and `sequential_tokenization.py`.
The sequential runner accepts both legacy `multimer_sequences` and the release
`sequences: [{protein_pair: ...}]` spelling. The release query parser now
preserves and validates `assembly`, and `prepare_apo.py` serializes it into each
prepared `query.json`. The standard CLI invokes the sequential runner after apo
preparation.

`examples/8jeo_sequential.yaml` is a template using the exact 8JEO sequences.
As written, it generates A with AtlasFold and B/C jointly with
AtlasFold-Multimer. Provided apo fields are still accepted. Prepared
multi-model PDB ensembles are expanded so every model remains an apo/prior
candidate.

```yaml
# Append to the prepared 8jeo input, retaining its A and protein_pair B/C entries:
assembly:
  stages:
    - id: assemble_BC
      chains: [B, C]
  selection: confidence_top1
```

This invokes KFold on B/C, selects confidence top-1 across all requested
seeds/samples, and invokes KFold on A/B/C with the selected B/C coordinates.
AtlasFold-Multimer's B/C apo preparation precedes these calls and is separate
from the KFold intermediate prediction. An existing protein pair cannot be
split by an assembly stage, so `[A, B]` is invalid while B/C remains paired.

```bash
cd /home/hwkim/kfold/_worktrees/sequential-complex-prior
pip install -e .
kfold --input examples/8jeo_sequential.yaml \
  --out-dir /path/to/8jeo_sequential_output \
  --seeds 1 2 3 4 5 --num-apos 5 \
  --conditioning prior_and_trunk_multichain
```

`prior_only` updates ECSI initialization only. `prior_and_trunk` also replaces
protein apo coordinates and recomputes structure tokens chainwise.
`prior_and_trunk_multichain` jointly tokenizes and encodes the selected B/C
protein complex while retaining their physical chain IDs. Existing B/C shared
apo UID is retained; a newly assembled pair does not receive a new shared apo
UID. Final denoising can change the B/C arrangement.

Main changed the model/configuration API. The backend now strictly loads a
state dictionary with the current KFold configuration and uses the current
inference/writer APIs. Historical 0720 checkpoints/configs are not automatically
converted. Keep a compatible historical runtime for reproducing old numerical
results; merging main alone does not establish checkpoint compatibility.
