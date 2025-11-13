import dataclasses
import json
from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from kfold.data import model_input
from kfold.data.metadata import Metadata
from kfold.utils.boltz.process import parse_record
from kfold.utils.registry import DATAMODULE, BaseConfig, Registry

from .cropper import BaseCropper
from .dataset import (
    BoltzTrainingDataset,
    BoltzValidationDataset,
    TrainingDataset,
    ValidationDataset,
)
from .filter import BaseFilter
from .sampler import BaseSampler

# HACK: (SeonghwanSeo): this is hard-coded right now. I'll fix it later.


class DataModuleConfig(BaseConfig):
    # Common config for data modules
    train_batch_size: int = 1
    val_batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    # Cropper config
    cropper: BaseCropper.Config


# FIXME: remove this (hard-coded)
class BoltzDataModuleConfig(DataModuleConfig):
    # Dataset specific (TODO: move to dataset config)
    boltz_processed_path: str | Path
    boltz_split_path: str | Path
    max_tokens: int  # Used for cropping and padding
    filters: list[BaseFilter.Config] = dataclasses.field(default_factory=list)
    sampler: BaseSampler.Config = dataclasses.field(
        default_factory=BaseSampler.Config
    )  # Default: uniform sampler


def collate(f_inputs: list[model_input.FoldingInput]) -> model_input.FoldingInput:
    return model_input.FoldingInput.from_list(f_inputs)


@DATAMODULE.register(config_cls=BoltzDataModuleConfig)
class TrainingDataModule(pl.LightningDataModule):
    # HACK: (SeonghwanSeo): currently only supports a single Boltz dataset
    # I'll remove this datamodule and make it better (multiple dataset)
    _train_ds: TrainingDataset
    _val_ds: ValidationDataset

    def __init__(self, config: BoltzDataModuleConfig) -> None:
        super().__init__()
        assert config.max_tokens % 128 == 0, "max_tokens must be a multiple of 128."

        self.config = config
        self.max_tokens: int = config.max_tokens

        self.filters: list[BaseFilter] = [
            Registry.instantiate(config=c) for c in config.filters
        ]
        self.cropper = Registry.instantiate(config=config.cropper)

        self.boltz_processed_path: Path = Path(config.boltz_processed_path)
        self.boltz_split_path: Path = Path(config.boltz_split_path)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self._train_ds = self.construct_train_dataset()
            self._val_ds = self.construct_val_dataset()
        elif stage == "validate":
            self._val_ds = self.construct_val_dataset()
        else:
            raise NotImplementedError("Not implemented yet.")

    def construct_train_dataset(self) -> TrainingDataset:
        # HACK: (SeonghwanSeo): hard-coded path to boltz manifest; single dataset
        # NOTE: (SeonghwanSeo): validation set is excluded during date-filtering.

        def do_filter(r: Metadata) -> bool:
            return all(filt(r) for filt in self.filters)

        boltz_processed_path = self.boltz_processed_path
        boltz_manifest_path = boltz_processed_path / "manifest.json"
        boltz_structure_path = boltz_processed_path / "structures"

        # Load records
        with open(boltz_manifest_path) as f:
            all_records: list[Metadata] = [parse_record(r) for r in json.load(f)]

        # Apply filters
        train_records = [r for r in all_records if do_filter(r)]

        return BoltzTrainingDataset(
            records=train_records,
            structure_dir=boltz_structure_path,
            max_tokens=self.max_tokens,
            cropper=self.cropper,
            sampler_config=self.config.sampler,
        )

    def construct_val_dataset(self) -> ValidationDataset:
        # HACK: (SeonghwanSeo): hard-coded path to boltz manifest; single dataset

        boltz_processed_path = self.boltz_processed_path
        boltz_manifest_path = boltz_processed_path / "manifest.json"
        boltz_structure_path = boltz_processed_path / "structures"

        # Load records
        with open(boltz_manifest_path) as f:
            all_records: list[Metadata] = [parse_record(r) for r in json.load(f)]

        # get validation records
        validation_split = self.boltz_split_path / "validation_ids.txt"
        with open(validation_split) as f:
            val_ids = set([line.strip().lower() for line in f])

        # Apply filters
        val_records = [r for r in all_records if r.id.lower() in val_ids]

        return BoltzValidationDataset(
            records=val_records,
            structure_dir=boltz_structure_path,
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
