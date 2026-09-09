"""Prepare KFold query documents with AtlasFold apo and prior structures."""

import copy
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import yaml
from tqdm import tqdm


def input_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            raise ValueError(f"Expected a YAML/JSON input: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        p
        for p in path.iterdir()
        if p.suffix.lower() in {".yaml", ".yml", ".json"} and p.is_file()
    )
    if not files:
        raise ValueError(f"No query files found in {path}")
    return files


def _name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"Invalid query name: {value!r}")
    return value


def needs_apo(document: dict) -> bool:
    return any(
        "protein" in wrapper and not wrapper["protein"].get("apo")
        for section in ("sequences", "multimer_sequences")
        for wrapper in document.get(section, [])
    )


def copy_document(document: dict, source_dir: str | Path) -> dict:
    """Copy a document, resolving existing structure paths before relocating it."""
    from kfold.inference.query import resolve_structure_path

    data = copy.deepcopy(document)
    for section in ("sequences", "multimer_sequences"):
        for wrapper in data.get(section, []):
            for entity in wrapper.values():
                for field in ("apo", "prior"):
                    if entity.get(field):
                        paths = entity[field]
                        if field == "apo" and isinstance(paths, str):
                            paths = [paths]
                        entity[field] = [
                            resolve_structure_path(p, source_dir) for p in paths
                        ]
    return data


def prepare_job_inputs(
    input_path: str | Path,
    out_dir: str | Path,
    *,
    seeds: Sequence[int],
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[tuple[Path, int]]:
    """Create a query and local copies of supplied structures for each seed."""
    files = input_files(input_path)
    documents = [yaml.safe_load(path.read_text()) for path in files]
    names = [
        _name(data.get("name", path.stem))
        for path, data in zip(files, documents, strict=True)
    ]
    if len(set(names)) != len(names):
        raise ValueError("Query names must be unique across input files.")
    root = Path(out_dir).resolve()
    targets = {
        root / name / f"{name}_seed-{seed}" / "query.yaml"
        for name in names
        for seed in seeds
    }
    if {path.resolve() for path in files} & targets:
        raise ValueError("Output queries must not overwrite source queries.")
    jobs = []
    for source, document, name in zip(files, documents, names, strict=True):
        for seed in seeds:
            directory = root / name / f"{name}_seed-{seed}"
            done = directory / "done.txt"
            if done.exists() and not overwrite and not dry_run:
                continue
            if overwrite:
                done.unlink(missing_ok=True)
            (directory / "apo").mkdir(parents=True, exist_ok=True)
            data = copy_document(document, source.parent)
            data["name"] = name
            data["seed"] = seed
            for section in ("sequences", "multimer_sequences"):
                for index, wrapper in enumerate(data.get(section, [])):
                    for entity in wrapper.values():
                        for field in ("apo", "prior"):
                            if not entity.get(field):
                                continue
                            paths = []
                            for number, supplied in enumerate(entity[field], 1):
                                supplied = Path(supplied)
                                destination = (
                                    directory
                                    / "apo"
                                    / f"{section}-{index}"
                                    / f"{field}-{number}{supplied.suffix}"
                                )
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                if supplied.resolve() != destination.resolve():
                                    shutil.copy2(supplied, destination)
                                paths.append(
                                    destination.relative_to(directory).as_posix()
                                )
                            entity[field] = paths
            path = directory / "query.yaml"
            temporary = path.with_suffix(".yaml.tmp")
            temporary.write_text(yaml.safe_dump(data, sort_keys=False))
            temporary.replace(path)
            jobs.append((path, seed))
    return jobs


def prepare_document(
    runner,
    document: dict,
    out_dir: str | Path,
    *,
    source_dir: str | Path = ".",
    apo_dir: str | Path = "apo",
    seeds: Sequence[int] = (1,),
    num_samples: int = 5,
    overwrite: bool = False,
) -> dict:
    """Return a copied query dictionary with paths to saved apo/prior files.

    Existing apo/prior inputs are preserved and made absolute. Generated
    structures are stored in ``apo_dir`` relative to the output query directory.
    """
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique, nonnegative integers.")
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    data = copy_document(document, source_dir)
    target = _name(data["name"])
    root = Path(out_dir).resolve()
    for section, multimer in (("sequences", False), ("multimer_sequences", True)):
        for index, wrapper in enumerate(data.get(section, [])):
            if "protein" not in wrapper:
                continue
            entity = wrapper["protein"]
            if entity.get("apo"):
                continue
            sequence = entity["sequence"]
            sequences = sequence.split(":") if multimer else [sequence]
            if not all(sequences) or (multimer and len(sequences) != 2):
                raise ValueError(
                    f"Invalid protein sequence in {target}:{section}[{index}]"
                )
            task_name = f"{section}-{index}"
            apo, prior = [], []
            for seed in seeds:
                directory = root / apo_dir / task_name / f"seed-{seed}"
                done = directory / "done.txt"
                complete = done.is_file() and all(
                    (directory / f"rank_{rank}.{suffix}").is_file()
                    for rank in range(1, num_samples + 1)
                    for suffix in ("pdb", "json")
                )
                if overwrite or not complete:
                    done.unlink(missing_ok=True)
                    if multimer:
                        from atlasfold.model import SamplingConfig

                        result = runner.atlasfold_multimer.fold(
                            task_name,
                            sequences,
                            seeds=[seed],
                            num_samples=num_samples,
                            num_recycles=4,
                            mlm_prob=0.20,
                            sampling_config=SamplingConfig(num_steps=100),
                        )
                    else:
                        result = runner.atlasfold.fold(
                            task_name,
                            sequence,
                            seeds=[seed],
                            num_samples=num_samples,
                            num_recycles=4,
                            mlm_prob=0.15,
                        )
                    ranked = sorted(
                        (
                            (i, sample)
                            for (s, i), sample in result.outputs.items()
                            if s == seed
                        ),
                        key=lambda pair: (-float(pair[1].ranking_score), pair[0]),
                    )
                    if len(ranked) != num_samples:
                        raise ValueError(
                            f"Incomplete AtlasFold output: {target}:{task_name}, "
                            f"seed={seed}"
                        )
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    temporary = Path(
                        tempfile.mkdtemp(prefix=".prediction-", dir=directory.parent)
                    )
                    try:
                        for rank, (sample_index, sample) in enumerate(ranked, 1):
                            pdb = sample.to_pdb(
                                model="multimer" if multimer else "monomer"
                            )
                            if not pdb.strip():
                                raise ValueError("AtlasFold returned an empty structure.")
                            (temporary / f"rank_{rank}.pdb").write_text(pdb)
                            (temporary / f"rank_{rank}.json").write_text(
                                json.dumps(
                                    dict(
                                        seed=seed,
                                        rank=rank,
                                        sample_index=sample_index,
                                        ranking_score=float(sample.ranking_score),
                                    )
                                )
                            )
                        if directory.exists():
                            shutil.rmtree(directory)
                        temporary.rename(directory)
                        done.touch()
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
                apo.append((directory / "rank_1.pdb").relative_to(root).as_posix())
                prior.extend(
                    (directory / f"rank_{rank}.pdb").relative_to(root).as_posix()
                    for rank in range(1, num_samples + 1)
                )
            entity["apo"] = apo
            if not entity.get("prior"):
                entity["prior"] = prior
    return data


def prepare_files(runner, input_path, out_dir, **kwargs) -> list[Path]:
    files = input_files(input_path)
    root = Path(out_dir).resolve()
    documents = [yaml.safe_load(path.read_text()) for path in files]
    for path, data in zip(files, documents, strict=True):
        data.setdefault("name", path.stem)
    names = [_name(data["name"]) for data in documents]
    if len(set(names)) != len(names):
        raise ValueError("Query names must be unique.")
    targets = [root / f"{name}.yaml" for name in names]
    if {p.resolve() for p in files} & set(targets):
        raise ValueError("Output queries must not overwrite source queries.")
    root.mkdir(parents=True, exist_ok=True)
    for path, data, target in tqdm(
        list(zip(files, documents, targets, strict=True)), desc="Prepare", unit="query"
    ):
        data = copy_document(data, path.parent)
        temporary = target.with_suffix(".yaml.tmp")
        temporary.write_text(yaml.safe_dump(data, sort_keys=False))
        temporary.replace(target)
        prepared = prepare_document(
            runner,
            data,
            root,
            source_dir=path.parent,
            apo_dir=Path("apo") / data["name"],
            **kwargs,
        )
        temporary.write_text(yaml.safe_dump(prepared, sort_keys=False))
        temporary.replace(target)
    return targets
