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
    select_top1,
    subset_query,
    subset_sources,
    validate_plan,
)


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
            self._validated = True
        device = self.model.device
        torch.manual_seed(query.seed)
        torch.cuda.manual_seed_all(query.seed)
        features = features.to(device)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, dtype=torch.bfloat16),
        ):
            if self.model.prot_struct_encoder is not None:
                apply_apo_structure_tokens(
                    features, records, self.model.prot_struct_encoder
                )
            if getattr(self.args, "conditioning", "prior_only") == "prior_and_trunk":
                np.savez_compressed(
                    out / "structure_token_ids.npz",
                    bb=features.sequence.bb_struct_token_id.cpu().numpy(),
                    fa=features.sequence.fa_struct_token_id.cpu().numpy(),
                )
            started = time.perf_counter()
            output = self.model.inference(
                features,
                num_recycles=self.args.num_recycles,
                num_steps=self.args.num_steps,
                num_samples=self.args.num_samples,
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

    if conditioning not in {"prior_only", "prior_and_trunk"}:
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
    out.mkdir(parents=True, exist_ok=True)
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
    groups = []
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
            groups = [g for g in groups if not g.chains <= chosen] + [obj]
            continue
        active = [g for g in groups if g.chains <= chosen]
        all_candidates = []
        for seed in seeds:
            target = stage_out / f"seed-{seed}"
            if target.exists():
                previous = json.loads((target / "input.json").read_text())
                if previous.get("conditioning", "prior_only") != conditioning:
                    raise ValueError(f"Conditioning changed: {target}")
                complete = json.loads((target / "complete.json").read_text())
                for filename, expected in complete["hashes"].items():
                    if digest(target / filename) != expected:
                        raise ValueError(f"Changed result: {target / filename}")
                candidates = complete["candidates"]
            else:
                # An incomplete attempt is never read as a completed seed.
                attempt = stage_out / f".seed-{seed}-{uuid.uuid4().hex}"
                attempt.mkdir()
                start = time.monotonic()
                try:
                    seeded = sub.copy(seed=seed)
                    source = subset_sources(source_full[seed], sub_struct, sources[seed])
                    extra = (
                        {"trunk_groups": active}
                        if conditioning == "prior_and_trunk"
                        else {}
                    )
                    struct, _, features, records = pipeline.run(
                        seeded,
                        sources=source,
                        prior_groups=active,
                        stage_index=stage_index,
                        **extra,
                    )
                    if conditioning == "prior_and_trunk":
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
                            "prior_objects": [sorted(g.chains) for g in active],
                            "conditioning": conditioning,
                            "trunk_objects": [sorted(g.chains) for g in active]
                            if extra
                            else [],
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
        groups = [g for g in groups if not g.chains <= chosen] + [obj]
        rows.extend(all_candidates)
    with (out / "candidates.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "completed.json", {"final_top1": best, "stages": len(stages)})
    return best
