import argparse
import numbers
import sys

import wandb


def is_scalar(value):
    return isinstance(value, numbers.Number) or (
        isinstance(value, str) and len(value) < 256
    )


def copy_run(
    source_path: str,
    target_project: str,
    target_entity: str | None = None,
    copy_media: bool = False,
):
    """
    Copy a WandB run to a different project.

    Args:
        source_path: The full path to the source run (entity/project/run_id).
        target_project: The name of the destination project.
        target_entity: The name of the destination entity
                       (optional, defaults to source entity).
        copy_media: Whether to attempt copying media files (experimental).
    """
    api = wandb.Api()

    try:
        src_run = api.run(source_path)
    except wandb.CommError as e:
        print(f"Error accessing run '{source_path}': {e}")
        sys.exit(1)
    except ValueError:
        print(f"Invalid run path format: '{source_path}'.")
        sys.exit(1)

    print(f"Found source run: {src_run.name} ({src_run.id})")

    if target_entity is None:
        target_entity = src_run.entity or api.default_entity

    print(f"Target: {target_entity}/{target_project}")

    # Initialize new run
    # We use 'disabled' mode first to set up config, but we need an active run to log.
    # actually wandb.init() creates a run.

    # Prepare config
    # Filter out wandb-specific config keys if any (usually ok to copy all)
    config = src_run.config

    print("Initializing new run...")
    new_run = wandb.init(
        entity=target_entity,
        project=target_project,
        name=f"{src_run.name}",
        config=config,
        tags=src_run.tags,
        notes=src_run.notes,
        job_type="copy",
    )

    print(f"New run created: {new_run.get_url()}")
    print("Copying history (this may take a while)...")

    # History
    # We use scan_history for efficient iteration
    history = src_run.scan_history()

    count = 0
    for row in history:
        step = row.get("_step")

        # Filter row for valid types if not copying media
        # If copy_media is False, we only keep scalars
        log_data = {}
        for k, v in row.items():
            if k.startswith("_"):
                continue

            if copy_media:
                log_data[k] = v
            else:
                # Simple heuristic: valid json types that aren't complex dicts (media)
                # But lists are fine.
                # WandB media are usually dicts with _type
                if isinstance(v, dict) and "_type" in v:
                    continue
                log_data[k] = v

        if log_data:
            new_run.log(log_data, step=step)
            count += 1
            if count % 100 == 0:
                print(f"Copied {count} steps...", end="\r")

    print(f"\nFinished copying {count} steps.")

    # Summary
    print("Updating summary...")
    for k, v in src_run.summary.items():
        if not k.startswith("_"):
            # Only update if not already set (log updates summary automatically)
            # But specific summary overrides might be needed
            # We can force update
            new_run.summary[k] = v

    new_run.finish()
    print("Copy complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy a WandB run to a different project."
    )
    parser.add_argument(
        "source_path", help="Source run path (e.g., entity/project/run_id)"
    )
    parser.add_argument("target_project", help="Target project name")
    parser.add_argument("--target_entity", help="Target entity name (optional)")
    parser.add_argument(
        "--copy_media", action="store_true", help="Attempt to copy media (experimental)"
    )

    args = parser.parse_args()

    copy_run(args.source_path, args.target_project, args.target_entity, args.copy_media)
