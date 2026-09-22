"""Copy completed per-seed apos to a second condition without running AtlasFold."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relocate(value, source, target):
    if isinstance(value, str) and value.startswith(str(source) + "/"):
        return str(target) + value[len(str(source)) :]
    if isinstance(value, list):
        return [relocate(v, source, target) for v in value]
    if isinstance(value, dict):
        return {k: relocate(v, source, target) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    copied = 0
    for path in sorted(args.input.glob("*.yaml")):
        query = yaml.safe_load(path.read_text())
        name = query.get("name", path.stem)
        prefixes = [
            f"{'multimer' if 'protein_pair' in entry else 'monomer'}-{i}"
            for i, entry in enumerate(query["sequences"], 1)
            if "protein" in entry or "protein_pair" in entry
        ]
        for seed in args.seeds:
            source = (args.source / name / f"{name}_seed-{seed}").resolve()
            target = (args.target / name / f"{name}_seed-{seed}").resolve()
            if source == target:
                raise ValueError("Source and target must differ")
            for prefix in prefixes:
                for suffix in (".done", "_apo.pdb", "_prior.pdb"):
                    required = source / "apo" / (prefix + suffix)
                    if not required.is_file():
                        raise FileNotFoundError(f"Incomplete prepared apo: {required}")
            prepared = json.loads((source / "query.json").read_text())
            expected = relocate(prepared, source, target)
            target.mkdir(parents=True, exist_ok=True)
            for file in [
                source / "apo_setting.json",
                *sorted((source / "apo").rglob("*")),
            ]:
                if not file.is_file():
                    continue
                dest = target / file.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    if digest(file) != digest(dest):
                        raise ValueError(
                            f"Existing apo differs; refusing to overwrite: {dest}"
                        )
                else:
                    temp = dest.with_name(dest.name + ".copying")
                    shutil.copy2(file, temp)
                    if digest(file) != digest(temp):
                        raise ValueError(f"Copy verification failed: {dest}")
                    temp.replace(dest)
            dest = target / "query.json"
            if dest.exists():
                if json.loads(dest.read_text()) != expected:
                    raise ValueError(f"Existing prepared query differs: {dest}")
            else:
                temp = dest.with_suffix(".json.copying")
                temp.write_text(json.dumps(expected, indent=2) + "\n")
                temp.replace(dest)
            copied += 1
        print(f"Reused apo: {name}, {len(args.seeds)} seeds", flush=True)
    print(
        f"Apo reuse complete: {copied} query/seed preparations; no AtlasFold generation.",
        flush=True,
    )


if __name__ == "__main__":
    main()
