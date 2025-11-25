import dataclasses
import json
import pickle
from functools import lru_cache
from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kfold.data.metadata import Metadata
from kfold.data.model_input import FoldingInput
from kfold.utils.registry import DATAMODULE, BaseConfig, Registry

from .cropper import BaseCropper
from .dataset import (
    LMDBTrainingDataset,
    LMDBValidationDataset,
    TrainingDataset,
    ValidationDataset,
)
from .dl_sampler import DistributedWeightedSampler
from .filter import BaseFilter
from .sampler import BaseSampler

# HACK: (SeonghwanSeo): this is hard-coded right now. I'll fix it later.


def collate(batches: list[tuple[FoldingInput, dict]]) -> tuple[FoldingInput, list[dict]]:
    f_input_batched = FoldingInput.from_list([b[0] for b in batches])
    meta_infos = [b[1] for b in batches]
    return f_input_batched, meta_infos


@lru_cache
def load_manifest(manifest_path: Path) -> list[Metadata]:
    format = manifest_path.suffix.lower()
    if format == ".json":
        with open(manifest_path) as f:
            manifest = json.load(f)
    elif format == ".pkl":
        with open(manifest_path, "rb") as f:
            manifest = pickle.load(f)
    else:
        raise ValueError(f"Unsupported manifest format: {format}")
    all_records: list[Metadata] = [Metadata.from_dict(d) for d in manifest]
    return all_records


class DataModuleConfig(BaseConfig):
    # === Common config for data modules === #
    train_batch_size: int = 1
    val_batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    safe_load: bool = True

    # === For debugging === #
    overfit_val: bool = False

    # === Cropping arguments === #
    cropper: BaseCropper.Config

    # === Featurization arguments === #
    featurization_args: dict


# FIXME: remove this (hard-coded)
class LMDBDataModuleConfig(DataModuleConfig):
    # Dataset specific (TODO: move to dataset config)
    lmdb_path: str | Path
    manifest_path: str | Path
    split_path: str | Path
    max_tokens: int  # Used for cropping and padding
    filters: list[BaseFilter.Config] = dataclasses.field(default_factory=list)
    sampler: BaseSampler.Config = dataclasses.field(
        default_factory=BaseSampler.Config
    )  # Default: uniform sampler


@DATAMODULE.register(config_cls=LMDBDataModuleConfig)
class TrainingDataModule(pl.LightningDataModule):
    # HACK: (SeonghwanSeo): currently only supports a single Boltz dataset
    # I'll remove this datamodule and make it better (multiple dataset)
    _train_ds: TrainingDataset
    _val_ds: ValidationDataset

    def __init__(self, config: LMDBDataModuleConfig) -> None:
        super().__init__()
        assert config.max_tokens % 128 == 0, "max_tokens must be a multiple of 128."

        self.config = config
        self.max_tokens: int = config.max_tokens

        self.filters: list[BaseFilter] = [
            Registry.instantiate(config=c) for c in config.filters
        ]
        self.cropper = Registry.instantiate(config=config.cropper)

        self.lmdb_path: Path = Path(config.lmdb_path)
        self.manifest_path: Path = Path(config.manifest_path)
        self.split_path: Path = Path(config.split_path)

        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB path not found: {self.lmdb_path}")

        self.featurization_args = config.featurization_args

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self._train_ds = self.construct_train_dataset()
            self._val_ds = self.construct_val_dataset()
        elif stage == "validate":
            self._val_ds = self.construct_val_dataset()
        else:
            raise NotImplementedError("Not implemented yet.")

    def construct_train_dataset(self) -> TrainingDataset:
        # HACK: (SeonghwanSeo): hard-coded path to rcsb set; single dataset
        # NOTE: (SeonghwanSeo): validation set is excluded during date-filtering.
        def do_filter(r: Metadata) -> bool:
            return all(filt(r) for filt in self.filters)

        def load_split_ids(split_file: Path) -> set[str]:
            with open(split_file) as f:
                ids = set([line.strip().lower() for line in f if line.strip()])
            return ids

        # Load records
        all_records: list[Metadata] = load_manifest(self.manifest_path)
        # By default, use all records
        train_records = all_records

        if self.config.overfit_val:
            # use only validation set for overfitting
            # skip filtering to overfit.
            validation_split = self.split_path / "validation_ids.txt"
            with open(validation_split) as f:
                val_ids = set([line.strip().lower() for line in f])
            train_records = [r for r in all_records if r.id.lower() in val_ids]
            self.print_rank_zero(
                f"Overfitting mode: using {len(train_records)} records "
                "from validation set. Replicated 10 times for more samples."
            )
            train_records = train_records * 10  # replicate to have more samples
        else:
            # If a train split file is provided, use it
            if (train_split_path := self.split_path / "train_ids.txt").exists():
                train_ids = load_split_ids(train_split_path)
                train_records = [r for r in all_records if r.id.lower() in train_ids]
                self.print_rank_zero(
                    f"Loaded train split file with {len(train_ids)} ids."
                    f" Total {len(train_records)} records selected."
                )
            else:
                self.print_rank_zero("No train split file found. Using all records.")

            # If a validation/test split file is provided, exclude those records
            for fn in ["validation_ids.txt", "test_ids.txt"]:
                if (test_split_path := self.split_path / fn).exists():
                    exclude_ids = load_split_ids(test_split_path)
                    train_records = [
                        r for r in train_records if r.id.lower() not in exclude_ids
                    ]

            # Apply filters
            train_records = [r for r in train_records if do_filter(r)]

        self.print_rank_zero(
            f"Constructed training dataset with total {len(train_records)} records "
            "after filtering."
        )

        return LMDBTrainingDataset(
            records=train_records,
            lmdb_path=self.lmdb_path,
            max_tokens=self.max_tokens,
            cropper=self.cropper,
            sampler_config=self.config.sampler,
            safe_load=self.config.safe_load,
            featurization_args=self.featurization_args,
        )

    def construct_val_dataset(self) -> ValidationDataset:
        # HACK: (SeonghwanSeo): hard-coded path to rcsb set; single dataset

        # Load records
        all_records: list[Metadata] = load_manifest(self.manifest_path)

        # get validation records
        validation_split = self.split_path / "validation_ids.txt"
        with open(validation_split) as f:
            val_ids = set([line.strip().lower() for line in f if line.strip()])
        val_records = [r for r in all_records if r.id.lower() in val_ids]

        self.print_rank_zero(
            f"Constructed validation dataset with {len(val_records)} records."
        )

        return LMDBValidationDataset(
            records=val_records,
            lmdb_path=self.lmdb_path,
            safe_load=self.config.safe_load,
            featurization_args=self.featurization_args,
        )

    def train_dataloader(self):
        dataset = self._train_ds

        weights = dataset.weights
        if weights is not None:
            sampler = DistributedWeightedSampler(
                weights=weights,  # type: ignore
                rank=self.trainer.global_rank if self.trainer else 0,
                world_size=self.trainer.world_size if self.trainer else 1,
                epoch=self.trainer.current_epoch if self.trainer else 0,
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            collate_fn=collate,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        # HACK: (SeonghwanSeo): single
        dataset = self._val_ds

        sampler = None
        if self.trainer is not None:
            if self.trainer.world_size > 1:
                sampler = DistributedSampler(
                    dataset,
                    rank=self.trainer.global_rank,
                    num_replicas=self.trainer.world_size,
                    shuffle=False,
                    drop_last=False,
                )

        return DataLoader(
            dataset,
            batch_size=self.config.val_batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )

    def print_rank_zero(self, msg: str, prefix: str = "[DataModule] ") -> None:
        if self.trainer is None or self.trainer.global_rank == 0:
            print(prefix + msg)
