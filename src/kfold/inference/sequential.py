"""Sequential inference orchestration with durable, verified seed results."""

import csv
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from .assembly import (
    PriorObject,
    atom_keys,
    chain_names,
    select_seed_apo_ensemble,
    select_top1,
    subset_query,
    subset_sources,
    validate_plan,
)

TRUNK_CONDITIONING_MODES = {
    "prior_and_trunk",
    "prior_and_trunk_multichain",
}
CONDITIONING_MODES = {"prior_only", *TRUNK_CONDITIONING_MODES}
TRUNK_CONDITIONING_SCHEMA = 3
TRUNK_APO_POLICY = "per_final_seed_top1_per_generation_seed"
MULTICHAIN_STRUCTURE_POLICY = "triprorep_chainwise_tokens_chain_ids_reset_positions_v1"


def apo_generation_seeds(seeds, counts):
    """Match native seed*10+slot for <=10 apos; avoid collisions for larger N."""
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Expected unique nonnegative inference seeds")
    if set(counts) != set(seeds) or any(n < 1 for n in counts.values()):
        raise ValueError("Expected a positive apo count for every inference seed")
    stride = max(10, max(counts.values()))
    result = {s: [s * stride + i for i in range(1, counts[s] + 1)] for s in seeds}
    if any(g >= 2**64 for values in result.values() for g in values):
        raise ValueError("Apo generation seed exceeds the torch seed range")
    return result


def execution_plan(query, direct=False):
    """Use the identical backend for a final-only, no-assembly control."""
    if direct:
        if query.assembly is not None:
            raise ValueError("Direct control requires an input without assembly")
        return [{"id": "final", "chains": chain_names(query)}]
    return validate_plan(query)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def config_files(config):
    import yaml

    paths = set()

    def visit(path):
        path = Path(path).resolve()
        if path in paths:
            return
        paths.add(path)
        data = yaml.safe_load(path.read_text())

        def walk(obj):
            if isinstance(obj, dict):
                if "_yaml_" in obj:
                    visit(path.parent / obj["_yaml_"])
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    walk(value)

        walk(data)

    visit(config)
    return paths


def run_manifest(args, queries):
    paths = {args.weight.resolve(), args.ccd.resolve()} | config_files(args.config)
    provided_root = getattr(args, "provided_intermediates", None)
    if provided_root is not None:
        for q in queries:
            for path in provided_stage_paths(q, provided_root).values():
                paths.add(path.resolve())
    for q in queries:
        for sequence in q.sequences + q.multimer_sequences:
            for attr in ("apo", "prior"):
                paths.update(
                    Path(p).resolve() for p in (getattr(sequence, attr, None) or [])
                )
    root = Path(__file__).resolve().parents[3]
    source = sorted((root / "src/kfold").rglob("*.py")) + [
        root / "scripts/inference_sequential.py"
    ]
    import torch

    return {
        "format": 1,
        "source_commit": (
            (root / "SOURCE_COMMIT").read_text().strip()
            if (root / "SOURCE_COMMIT").is_file()
            else subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
        ),
        "source_hashes": {str(p.relative_to(root)): digest(p) for p in source},
        "files": {str(p): digest(p) for p in sorted(paths)},
        "config": str(args.config.resolve()),
        "weight": str(args.weight.resolve()),
        "queries": {q.name: q.yaml for q in queries},
        "seeds": args.seed,
        "samples": args.num_samples,
        "recycles": args.num_recycles,
        "steps": args.num_steps,
        "num_apo": args.num_apo,
        "torch": torch.__version__,
        "conditioning": getattr(args, "conditioning", "prior_only"),
        "direct_control": getattr(args, "direct", False),
        **(
            {"provided_intermediates": str(provided_root.resolve())}
            if provided_root is not None
            else {}
        ),
    }


def provided_stage_paths(query, root):
    """Explicit complete intermediates bypass their prediction, never selection."""
    stages = execution_plan(query)[:-1]
    paths = {s["id"]: Path(root) / query.name / (s["id"] + ".npz") for s in stages}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing provided intermediate: {path}")
    return paths


class ModelBackend:
    """One model load; a fresh full model call for every stage and seed."""

    def __init__(self, args, model=None):
        self.args = args
        self.model = model
        self._validated = False

    def predict(self, query, struct, features, records, out):
        import torch

        from kfold.data.utils.writer import KFoldWriter
        from kfold.model import KFold
        from kfold.utils import confidence_metrics

        from .sequential_tokenization import apply_apo_structure_tokens

        if self.model is None:
            from omegaconf import OmegaConf

            self.model = KFold(OmegaConf.load(self.args.config))
            state = torch.load(self.args.weight, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
            del state
            self.model.requires_grad_(False).eval().cuda()
        if not self._validated:
            from kfold.model.modules.ecsi import KFoldECSI

            if not isinstance(self.model.diffusion_head, KFoldECSI):
                raise ValueError(
                    "Sequential prior experiment requires an ECSI diffusion head"
                )
            if (
                getattr(self.args, "conditioning", "prior_only")
                == "prior_and_trunk_multichain"
                and self.model.prot_struct_encoder is None
            ):
                raise ValueError(
                    "prior_and_trunk_multichain requires the protein structure encoder"
                )
            self._validated = True
        device = self.model.device
        torch.manual_seed(query.seed)
        torch.cuda.manual_seed_all(query.seed)
        features = features.to(device)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, dtype=torch.bfloat16),
        ):
            structure_seq_id = None
            structure_pos_id = None
            structure_chain_id = None
            if self.model.prot_struct_encoder is not None:
                structure_seq_id, structure_pos_id, structure_chain_id = (
                    apply_apo_structure_tokens(
                        features, records, self.model.prot_struct_encoder
                    )
                )
            if getattr(self.args, "conditioning", "prior_only") in (
                TRUNK_CONDITIONING_MODES
            ):
                effective_structure_seq_id = (
                    features.sequence.asym_id
                    if structure_seq_id is None
                    else structure_seq_id
                )
                effective_structure_pos_id = (
                    features.sequence.pos_id
                    if structure_pos_id is None
                    else structure_pos_id
                )
                np.savez_compressed(
                    out / "structure_token_ids.npz",
                    bb=features.sequence.bb_struct_token_id.cpu().numpy(),
                    fa=features.sequence.fa_struct_token_id.cpu().numpy(),
                    asym_id=features.sequence.asym_id.cpu().numpy(),
                    pos_id=features.sequence.pos_id.cpu().numpy(),
                    structure_seq_id=effective_structure_seq_id.cpu().numpy(),
                    structure_pos_id=effective_structure_pos_id.cpu().numpy(),
                    structure_chain_id=(
                        torch.zeros_like(features.sequence.asym_id)
                        if structure_chain_id is None
                        else structure_chain_id
                    )
                    .cpu()
                    .numpy(),
                )
            started = time.perf_counter()
            output = self.model.inference(
                features,
                num_recycles=self.args.num_recycles,
                num_steps=self.args.num_steps,
                num_samples=self.args.num_samples,
                structure_seq_id=structure_seq_id,
                structure_pos_id=structure_pos_id,
                structure_chain_id=structure_chain_id,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing = {"inference_seconds": time.perf_counter() - started}
        coords = (
            output["diffusion"]["coordinates"][:, : struct.num_atoms]
            .float()
            .cpu()
            .numpy()
        )
        summary, scores = confidence_metrics.summarize_confidence_metrics(
            features, struct, output
        )
        writer = KFoldWriter()
        candidates = []
        for sample, xyz in enumerate(coords):
            stem = f"{query.name}_seed-{query.seed}_sample-{sample}"
            obj = PriorObject(atom_keys(struct), xyz)
            obj.save(out / f"{stem}_atoms.npz")
            cif_path = out / f"{stem}.cif"
            writer.write_mmcif(
                struct.copy_with_new_coords(xyz, b_factors=scores[sample]["plddt"]),
                cif_path,
            )
            if not cif_path.is_file():
                raise RuntimeError(f"Failed to write prediction: {cif_path}")
            write_json(out / f"{stem}_confidences.json", summary[sample])
            if getattr(self.args, "save_confidence", True):
                np.savez_compressed(out / f"{stem}_confidences.npz", **scores[sample])
            candidates.append(
                dict(
                    seed=query.seed,
                    sample=sample,
                    **summary[sample]["complex"],
                    stem=stem,
                )
            )
        # This is the sampler-reported x_T, after global rigid augmentation.
        np.savez_compressed(
            out / "ecsi_init.npz",
            coordinates=output["diffusion"]["init_coordinates"][:, : struct.num_atoms]
            .cpu()
            .numpy(),
        )
        write_json(out / "timing.json", timing)
        return candidates


def run_query(
    query,
    pipeline,
    seeds,
    samples,
    out,
    backend,
    direct=False,
    conditioning="prior_only",
    provided_intermediates=None,
    source_queries=None,
):
    from .sequential_dataset import InferenceDataset

    if conditioning not in CONDITIONING_MODES:
        raise ValueError("Unknown sequential conditioning")
    if direct and conditioning != "prior_only":
        raise ValueError("Direct control has no predicted intermediate to re-encode")
    stages = execution_plan(query, direct)
    provided_intermediates = provided_intermediates or {}
    if direct and provided_intermediates:
        raise ValueError("Direct control cannot use provided intermediates")
    if set(provided_intermediates) - {s["id"] for s in stages[:-1]}:
        raise ValueError("Provided intermediates must reference a non-final stage")
    plain = query.copy(assembly=None)
    full = pipeline.read_query(plain)
    if source_queries is None:
        source_queries = {seed: plain.copy(seed=seed) for seed in seeds}
    if set(source_queries) != set(seeds):
        raise ValueError("Prepared source queries must match all inference seeds")
    source_full = {}
    for seed, source_query in source_queries.items():
        source_query = source_query.copy(seed=seed, assembly=None)
        source_struct = pipeline.read_query(source_query)
        if atom_keys(source_struct) != atom_keys(full):
            raise ValueError(f"Prepared query atom mapping changed for seed {seed}")
        source_queries[seed] = source_query
        source_full[seed] = source_struct
    # Resolve against the full original query, so earlier stages cannot change
    # apo selection or per-chain prior choices of later stages.
    sources = {
        s: pipeline.resolve_structure_sources(
            source_full[s],
            source_queries[s],
            np.random.default_rng(np.random.SeedSequence([s, 0])),
        )
        for s in seeds
    }
    trunk_mode = conditioning in TRUNK_CONDITIONING_MODES
    generation_seeds = (
        apo_generation_seeds(seeds, {s: sources[s].num_apo for s in seeds})
        if trunk_mode
        else {}
    )
    out.mkdir(parents=True, exist_ok=True)
    if trunk_mode:
        policy = {
            "conditioning_schema": TRUNK_CONDITIONING_SCHEMA,
            "apo_policy": TRUNK_APO_POLICY,
            "conditioning": conditioning,
            "seeds": list(seeds),
            "samples_per_generation_seed": samples,
            "generation_seeds": {str(s): v for s, v in generation_seeds.items()},
            "stages": stages,
        }
        if conditioning == "prior_and_trunk_multichain":
            policy["structure_representation_policy"] = MULTICHAIN_STRUCTURE_POLICY
        policy_path = out / "apo_policy.json"
        if policy_path.exists():
            if json.loads(policy_path.read_text()) != policy:
                raise ValueError(
                    "Sequential apo policy changed; use a new output directory"
                )
        elif any((out / stage["id"]).exists() for stage in stages):
            raise ValueError(
                "Missing per-seed apo policy for existing results; "
                "use a new output directory"
            )
        else:
            write_json(policy_path, policy)
    for seed, source in sources.items():
        # Retain exact selected apo/prior arrays for auditing embedding inputs.
        values = {f"apo_{a}": xyz for a, xyz in source.apo_coords.items()}
        values.update(
            {
                f"prior_{i}_{a}": xyz
                for i, p in enumerate(source.prior_sources)
                for a, xyz in p.items()
            }
        )
        source_path = out / f"source_choices_seed-{seed}.npz"
        if source_path.exists():
            with np.load(source_path, allow_pickle=False) as saved:
                if set(saved.files) != set(values) or any(
                    not np.array_equal(saved[k], v, equal_nan=True)
                    for k, v in values.items()
                ):
                    raise ValueError(f"Resolved sources changed: {source_path}")
        else:
            np.savez_compressed(source_path, **values)
    groups = {s: [] for s in seeds}
    rows = []
    padder = InferenceDataset([], pipeline.ccd, samples, pipeline.num_apo)
    for stage_index, stage in enumerate(stages):
        stage_out = out / stage["id"]
        stage_out.mkdir(parents=True, exist_ok=True)
        chosen = set(stage["chains"])
        sub = subset_query(plain, chosen)
        sub_struct = pipeline.read_query(sub)
        if stage["id"] in provided_intermediates:
            path = Path(provided_intermediates[stage["id"]]).resolve()
            obj = PriorObject.load(path)
            if set(obj.keys) != set(atom_keys(sub_struct)):
                raise ValueError(
                    "Provided intermediate atom mapping is incomplete or mismatched"
                )
            receipt = {
                "kind": "provided_structure",
                "path": str(path),
                "sha256": digest(path),
                "chains": sorted(obj.chains),
                "inference_skipped": True,
                "observed_atoms": int(obj.observed_mask.sum()),
                "missing_atoms": int((~obj.observed_mask).sum()),
                "missing_protein_policy": "masked_trunk_native_langevin_prior_only",
            }
            receipt_path = stage_out / "provided_structure.json"
            if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
                raise ValueError("Provided intermediate changed during resume")
            write_json(receipt_path, receipt)
            obj.save(stage_out / "provided_atoms.npz")
            for seed in seeds:
                groups[seed] = [g for g in groups[seed] if not g.chains <= chosen] + [obj]
            continue
        all_candidates = []
        per_seed_apo = trunk_mode and stage["id"] != "final"
        jobs = [
            (source_seed, seed)
            for source_seed in seeds
            for seed in (generation_seeds[source_seed] if per_seed_apo else [source_seed])
        ]
        for source_seed, seed in jobs:
            active = [g for g in groups[source_seed] if g.chains <= chosen]
            job_root = (
                stage_out / f"parent-seed-{source_seed}" if per_seed_apo else stage_out
            )
            job_root.mkdir(parents=True, exist_ok=True)
            target = job_root / f"seed-{seed}"
            job_policy = (
                {"source_seed": source_seed, "apo_policy": TRUNK_APO_POLICY}
                if trunk_mode
                else {}
            )
            if conditioning == "prior_and_trunk_multichain":
                job_policy["structure_representation_policy"] = (
                    MULTICHAIN_STRUCTURE_POLICY
                )
            if target.exists():
                previous = json.loads((target / "input.json").read_text())
                if previous.get("conditioning", "prior_only") != conditioning:
                    raise ValueError(f"Conditioning changed: {target}")
                if (
                    conditioning in TRUNK_CONDITIONING_MODES
                    and previous.get("conditioning_schema") != TRUNK_CONDITIONING_SCHEMA
                ):
                    raise ValueError(
                        f"Trunk conditioning schema changed: {target}; "
                        "use a new output directory"
                    )
                if any(previous.get(k) != v for k, v in job_policy.items()):
                    raise ValueError(f"Apo seed routing changed: {target}")
                complete = json.loads((target / "complete.json").read_text())
                for filename, expected in complete["hashes"].items():
                    if digest(target / filename) != expected:
                        raise ValueError(f"Changed result: {target / filename}")
                candidates = complete["candidates"]
            else:
                # An incomplete attempt is never read as a completed seed.
                attempt = job_root / f".seed-{seed}-{uuid.uuid4().hex}"
                attempt.mkdir()
                start = time.monotonic()
                try:
                    seeded = sub.copy(seed=seed)
                    source = subset_sources(
                        source_full[source_seed], sub_struct, sources[source_seed]
                    )
                    extra = {}
                    if conditioning in TRUNK_CONDITIONING_MODES:
                        extra = {
                            "trunk_groups": active,
                            "multichain_structure": conditioning
                            == "prior_and_trunk_multichain",
                        }
                    struct, _, features, records = pipeline.run(
                        seeded,
                        sources=source,
                        prior_groups=active,
                        stage_index=stage_index,
                        **extra,
                    )
                    if conditioning in TRUNK_CONDITIONING_MODES:
                        np.savez_compressed(
                            attempt / "trunk_conditioning.npz",
                            apo_coords=features.atom.apo_coords.numpy(),
                            apo_mask=features.atom.apo_mask.numpy(),
                            apo_uid=features.token.apo_uid.numpy(),
                            ref_pos=features.atom.ref_pos.numpy(),
                            ref_space_uid=features.atom.ref_space_uid.numpy(),
                            apo_repr_coords=features.token.apo_repr_coords.numpy(),
                            apo_frame_coords=features.token.apo_frame_coords.numpy(),
                        )
                    structure_repr_objects = []
                    if conditioning == "prior_and_trunk_multichain":
                        protein_names = {
                            metadata.name
                            for metadata, chain in zip(
                                struct.metadata.chains, struct.chains, strict=True
                            )
                            if chain.is_protein
                        }
                        structure_repr_objects = [
                            sorted(group.chains & protein_names)
                            for group in active
                            if len(group.chains & protein_names) >= 2
                        ]
                    np.savez_compressed(
                        attempt / "prior.npz",
                        coordinates=features.atom.prior_coords.numpy(),
                        keys=json.dumps(atom_keys(struct)),
                    )
                    write_json(
                        attempt / "input.json",
                        {
                            "original_query": query.yaml,
                            "chains": stage["chains"],
                            "seed": seed,
                            **job_policy,
                            "prior_objects": [sorted(g.chains) for g in active],
                            "conditioning": conditioning,
                            **(
                                {"conditioning_schema": TRUNK_CONDITIONING_SCHEMA}
                                if conditioning in TRUNK_CONDITIONING_MODES
                                else {}
                            ),
                            "trunk_apo_samples": [
                                len(g.apo_coordinates)
                                if g.apo_coordinates is not None
                                else 1
                                for g in active
                            ]
                            if extra
                            else [],
                            "trunk_objects": [sorted(g.chains) for g in active]
                            if extra
                            else [],
                            "structure_repr_objects": structure_repr_objects,
                            "atom_keys": atom_keys(struct),
                            "num_tokens": features.num_tokens,
                        },
                    )
                    features = padder.pad_input(features)
                    candidates = backend.predict(
                        seeded, struct, features, records, attempt
                    )
                    if len(candidates) != samples:
                        raise ValueError(
                            f"Incomplete seed {seed}: {len(candidates)}/{samples}"
                        )
                    select_top1(candidates)  # validate all scores before publishing
                    hashes = {p.name: digest(p) for p in attempt.iterdir() if p.is_file()}
                    write_json(
                        attempt / "complete.json",
                        dict(
                            candidates=candidates,
                            hashes=hashes,
                            seconds=time.monotonic() - start,
                        ),
                    )
                    attempt.rename(target)
                except Exception as exc:
                    write_json(attempt / "failure.json", {"error": repr(exc)})
                    raise
            if (
                len(candidates) != samples
                or {c["sample"] for c in candidates} != set(range(samples))
                or any(c["seed"] != seed for c in candidates)
            ):
                raise ValueError(f"Invalid completed candidate set: {target}")
            for c in candidates:
                all_candidates.append(
                    {
                        **c,
                        **({"source_seed": source_seed} if trunk_mode else {}),
                        "stage": stage["id"],
                        "cif": str(target / f"{c['stem']}.cif"),
                        "atoms": str(target / f"{c['stem']}_atoms.npz"),
                    }
                )
        best = select_top1(all_candidates)
        selection_file = stage_out / "selection.json"
        if selection_file.exists() and json.loads(selection_file.read_text()) != best:
            raise ValueError(f"Selection changed: {selection_file}")
        write_json(selection_file, best)
        obj = PriorObject.load(best["atoms"])
        for source_seed in seeds:
            next_obj = obj
            if per_seed_apo:
                candidates = [
                    c for c in all_candidates if c["source_seed"] == source_seed
                ]
                next_obj, apo_selection = select_seed_apo_ensemble(
                    candidates, generation_seeds[source_seed], prior=obj
                )
                apo_root = stage_out / f"parent-seed-{source_seed}"
                apo_selection_file = apo_root / "apo_selection.json"
                if (
                    apo_selection_file.exists()
                    and json.loads(apo_selection_file.read_text()) != apo_selection
                ):
                    raise ValueError(f"Apo selection changed: {apo_selection_file}")
                write_json(apo_selection_file, apo_selection)
                next_obj.save(apo_root / "selected_ensemble.npz")
            groups[source_seed] = [
                g for g in groups[source_seed] if not g.chains <= chosen
            ] + [next_obj]
        rows.extend(all_candidates)
    with (out / "candidates.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "completed.json", {"final_top1": best, "stages": len(stages)})
    return best
