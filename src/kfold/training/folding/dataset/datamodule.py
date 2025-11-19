import dataclasses
import json
import pickle
from functools import lru_cache
from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

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
    # Common config for data modules
    train_batch_size: int = 1
    val_batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    safe_load: bool = True
    # Cropper config
    cropper: BaseCropper.Config


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

        # Load records
        all_records: list[Metadata] = load_manifest(self.manifest_path)

        # Apply filters
        train_records = [r for r in all_records if do_filter(r)]

        return LMDBTrainingDataset(
            records=train_records,
            lmdb_path=self.lmdb_path,
            max_tokens=self.max_tokens,
            cropper=self.cropper,
            sampler_config=self.config.sampler,
            safe_load=self.config.safe_load,
        )

    def construct_val_dataset(self) -> ValidationDataset:
        # HACK: (SeonghwanSeo): hard-coded path to rcsb set; single dataset

        # Load records
        all_records: list[Metadata] = load_manifest(self.manifest_path)

        # get validation records
        validation_split = self.split_path / "validation_ids.txt"
        with open(validation_split) as f:
            val_ids = set([line.strip().lower() for line in f])

        # Apply filters
        val_records = [r for r in all_records if r.id.lower() in val_ids]
        # Sort validation records by the length of sequences (for efficient)
        val_records.sort(key=lambda r: r.num_residues)

        return LMDBValidationDataset(
            records=val_records,
            lmdb_path=self.lmdb_path,
            safe_load=self.config.safe_load,
        )

    def train_dataloader(self):
        dataset = self._train_ds

        weights = dataset.weights
        if weights is not None:
            sampler = WeightedRandomSampler(
                weights=weights,  # type: ignore
                num_samples=len(weights),
                replacement=True,
            )
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True if sampler is None else False,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        # HACK: (SeonghwanSeo): single
        dataset = self._val_ds
        return DataLoader(
            dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate,
        )
