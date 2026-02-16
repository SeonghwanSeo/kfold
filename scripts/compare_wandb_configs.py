#!/usr/bin/env python3
"""Compare wandb run configs between two runs."""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import wandb


@dataclass
class DiffChange:
    key: str
    type: str
    old_value: Any | None = None
    new_value: Any | None = None


@dataclass
class DiffResult:
    """Result of comparing two configs."""

    changes: list[DiffChange] = field(default_factory=list)

    def has_diff(self) -> bool:
        return bool(self.changes)


def diff_recursive(
    config1: dict[str, Any],
    config2: dict[str, Any],
    path: str = "",
) -> list[DiffChange]:
    changes: list[DiffChange] = []
    keys1 = set(config1.keys())
    keys2 = set(config2.keys())

    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common_keys = keys1 & keys2

    for key in sorted(only_in_1):
        full_path = f"{path}.{key}" if path else key
        changes.append(DiffChange(key=full_path, type="removed", old_value=config1[key]))

    for key in sorted(only_in_2):
        full_path = f"{path}.{key}" if path else key
        changes.append(DiffChange(key=full_path, type="added", new_value=config2[key]))

    for key in sorted(common_keys):
        val1 = config1[key]
        val2 = config2[key]
        full_path = f"{path}.{key}" if path else key

        if isinstance(val1, dict) and isinstance(val2, dict):
            changes.extend(diff_recursive(val1, val2, full_path))
        else:
            type1 = type(val1).__name__
            type2 = type(val2).__name__

            if type1 != type2:
                changes.append(
                    DiffChange(
                        key=full_path, type="type_changed", old_value=val1, new_value=val2
                    )
                )
            elif val1 != val2:
                changes.append(
                    DiffChange(
                        key=full_path,
                        type="value_changed",
                        old_value=val1,
                        new_value=val2,
                    )
                )

    return changes


def diff_configs(config1: dict[str, Any], config2: dict[str, Any]) -> DiffResult:
    result = DiffResult()
    result.changes = diff_recursive(config1, config2)
    return result


def print_diff(diff: DiffResult, run1_name: str, run2_name: str) -> None:
    if not diff.has_diff():
        print("No differences found in configs.")
        return

    print(f"\nConfig differences between {run1_name} and {run2_name}:\n")
    print("=" * 80)

    grouped: dict[str, list[DiffChange]] = {}
    for change in diff.changes:
        parts = change.key.split(".")
        prefix = parts[0] if len(parts) > 1 else "root"
        grouped.setdefault(prefix, []).append(change)

    for prefix in sorted(grouped.keys()):
        print(f"\n[{prefix}]")

        for change in grouped[prefix]:
            if change.type == "removed":
                print(f"  - {change.key}")
                print(f"    Only in {run1_name}: {change.old_value}")
            elif change.type == "added":
                print(f"  + {change.key}")
                print(f"    Only in {run2_name}: {change.new_value}")
            elif change.type == "type_changed":
                print(f"  ~ {change.key}")
                print(
                    f"    {run1_name}: {change.old_value} "
                    f"({type(change.old_value).__name__})"
                )
                print(
                    f"    {run2_name}: {change.new_value} "
                    f"({type(change.new_value).__name__})"
                )
            elif change.type == "value_changed":
                print(f"  ~ {change.key}")
                print(f"    {run1_name}: {change.old_value}")
                print(f"    {run2_name}: {change.new_value}")

    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare wandb run configs")
    parser.add_argument("run1", help="First wandb run (e.g., entity/project/run_id)")
    parser.add_argument("run2", help="Second wandb run (e.g., entity/project/run_id)")
    parser.add_argument(
        "--output",
        type=Path,
        help="Save diff to JSON file",
    )
    args = parser.parse_args()

    def parse_run_path(run_path: str) -> tuple[str, str, str]:
        parts = run_path.split("/")
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        elif len(parts) == 1:
            api = wandb.Api()
            run = api.run(run_path)
            return run.entity, run.project, run.id
        else:
            raise ValueError(f"Invalid run path: {run_path}")

    entity1, project1, run_id1 = parse_run_path(args.run1)
    entity2, project2, run_id2 = parse_run_path(args.run2)

    api = wandb.Api()
    run1 = api.run(f"{entity1}/{project1}/{run_id1}")
    run2 = api.run(f"{entity2}/{project2}/{run_id2}")

    config1 = run1.config
    config2 = run2.config

    diff = diff_configs(config1, config2)

    run1_name = f"{entity1}/{project1}/{run_id1}"
    run2_name = f"{entity2}/{project2}/{run_id2}"
    print("\nComparing configs:")
    print(f"  Run 1: {run1_name}")
    print(f"  Run 2: {run2_name}")
    print(f"  Run 1 name: {run1.name}")
    print(f"  Run 2 name: {run2.name}")

    print_diff(diff, run1_name, run2_name)

    if args.output:
        output_data = {
            "run1": {
                "path": run1_name,
                "name": run1.name,
                "config": config1,
            },
            "run2": {
                "path": run2_name,
                "name": run2.name,
                "config": config2,
            },
            "diff": [
                {
                    "key": c.key,
                    "type": c.type,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                }
                for c in diff.changes
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nDiff saved to: {args.output}")


if __name__ == "__main__":
    main()
