import argparse
import csv
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from tqdm import tqdm

from kfold.config import load_config
from kfold.training.folding.dataset.datamodule import TrainingDataModule
from kfold.training.folding.dataset.dataset import TrainingDataset


def analyze_structure(apo_coords, holo_coords, masks, prefix=""):
    """
    Compute coordinate variance for normalization.
    """
    # Flatten if needed
    if apo_coords.ndim == 3:  # [N, 24, 3]
        apo_coords = apo_coords.reshape(-1, 3)
        holo_coords = holo_coords.reshape(-1, 3)
        masks = masks.reshape(-1)

    # Filter valid atoms
    valid_mask = masks.astype(bool)
    if not np.any(valid_mask):
        return None

    valid_apo = apo_coords[valid_mask]
    valid_holo = holo_coords[valid_mask]

    # 1. Center the coordinates (Normalization requires zero mean)
    valid_apo = valid_apo - np.mean(valid_apo, axis=0)
    valid_holo = valid_holo - np.mean(valid_holo, axis=0)

    # 2. Compute Variance (Mean of squared values across all dimensions)
    # Var = E[x^2] since E[x] = 0
    # We aggregate over N atoms and 3 dimensions
    var_apo = np.mean(valid_apo**2)
    var_holo = np.mean(valid_holo**2)

    return {
        f"{prefix}var_apo": float(var_apo),
        f"{prefix}var_holo": float(var_holo),
        f"{prefix}std_apo": float(np.sqrt(var_apo)),
        f"{prefix}std_holo": float(np.sqrt(var_holo)),
        f"{prefix}num_atoms": int(len(valid_apo)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train-af3-tiny.yaml")
    parser.add_argument("--out_csv", type=str, default="coordinate_variance.csv")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    pl.seed_everything(42)

    config = load_config(Path(args.config))
    config.train.data.return_train_symmetry = False
    config.train.data.return_validation_symmetry = False

    print(f"Loading DataModule from {args.config}...")
    dm = TrainingDataModule(config.train.data)
    dm.setup("fit")

    results = []

    for stage_name, dataset in [("train", dm._train_ds), ("validation", dm._val_ds)]:
        print(f"\nAnalyzing {stage_name} dataset...")

        if isinstance(dataset, TrainingDataset):
            num_total = len(dataset.samples)
            indices = np.random.permutation(num_total)
        else:
            num_total = len(dataset.metadatas)
            indices = np.random.permutation(num_total)

        count = 0
        pbar = tqdm(total=min(args.limit, num_total))

        for idx in indices:
            if count >= args.limit:
                break

            try:
                if isinstance(dataset, TrainingDataset):
                    sample = dataset.samples[idx]
                    record = sample.metadata
                    asym_ids = sample.asym_id
                else:
                    record = dataset.metadatas[idx]
                    asym_ids = None

                struct = dataset.load_tokenized_structure(record)

                # Pre-crop
                if isinstance(dataset, TrainingDataset):
                    struct = dataset.pre_crop_structure(struct, asym_ids=asym_ids)

                struct = dataset.augment_apo_structure(struct)

                # PRE-CROP Variance
                pre_holo = struct.atom.coords[..., 0, :]
                pre_apo = struct.atom.apo_coords[..., 0, :]
                pre_mask = struct.atom.resolved_mask & struct.atom.apo_mask[..., 0]

                pre_stats = analyze_structure(pre_apo, pre_holo, pre_mask, prefix="pre_")

                # Crop & Featurize
                if isinstance(dataset, TrainingDataset):
                    struct = dataset.crop_structure(struct, asym_ids=asym_ids)

                f_input = dataset.featurize(struct, record)

                # POST-CROP Variance
                post_holo = f_input.atom.label_coords.numpy()
                post_apo = f_input.atom.apo_coords.numpy()
                post_mask = (f_input.atom.resolved_mask & f_input.atom.apo_mask).numpy()

                post_stats = analyze_structure(
                    post_apo, post_holo, post_mask, prefix="post_"
                )

                row = {"dataset": stage_name, "id": record.id}
                if pre_stats:
                    row.update(pre_stats)
                if post_stats:
                    row.update(post_stats)

                results.append(row)
                count += 1
                pbar.update(1)

            except Exception as _:
                pass

        pbar.close()

    # Save and Print Summary
    if results:
        keys = results[0].keys()
        with open(args.out_csv, "w") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"\nSaved results to {args.out_csv}")

        train_res = [r for r in results if r["dataset"] == "train"]
        val_res = [r for r in results if r["dataset"] == "validation"]

        for name, res in [("Train", train_res), ("Validation", val_res)]:
            if not res:
                continue
            print(
                f"\n{name} Dataset ({len(res)} samples) - Coordinate Variance Statistics:"
            )

            # Helper to print stats
            def print_metric(key_desc, key, res):
                vals = [r.get(key, 0) for r in res if key in r]
                if vals:
                    mean_v = np.mean(vals)
                    std_v = np.std(vals)
                    print(f"  {key_desc}: Mean={mean_v:.3f} ± {std_v:.3f}")

            print("  [Pre-Crop (Full/Complex)]")
            print_metric("Prior (Apo) Variance ", "pre_var_apo", res)
            print_metric("Label (Holo) Variance", "pre_var_holo", res)

            print("  [Post-Crop (Network Input)]")
            print_metric("Prior (Apo) Variance ", "post_var_apo", res)
            print_metric("Label (Holo) Variance", "post_var_holo", res)


if __name__ == "__main__":
    main()
