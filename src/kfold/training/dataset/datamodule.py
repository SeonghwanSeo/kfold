import dataclasses
import logging
from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kfold.data.types.ccd import CCD
from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import DATAMODULE, BaseConfig

from .dataset import (
    MultiTrainingDataset,
    TrainingDatasetConfig,
    ValidationDataset,
    ValidationDatasetConfig,
)
from .dl_sampler import DistributedWeightedSampler


def collate(batches: list[tuple[FoldingInput, dict]]) -> tuple[FoldingInput, list[dict]]:
    f_input_batched = FoldingInput.from_list([b[0] for b in batches], pad_to_max=False)
    meta_infos = [b[1] for b in batches]
    return f_input_batched, meta_infos


class DataModuleConfig(BaseConfig):
    # === Common config for data modules === #
    train_batch_size: int = 1
    val_batch_size: int = 1
    num_workers: int = 0
    persistent_workers: bool = True
    pin_memory: bool = True
    safe_load: bool = True

    # === Training hyperparameters === #
    max_chains: int = 20
    max_tokens: int = 384
    max_sequence_tokens: int = 768

    # === CCD path === #
    ccd_path: Path

    # === Dataset configs === #
    train_datasets: list[TrainingDatasetConfig] = dataclasses.field(default_factory=list)
    val_datasets: list[ValidationDatasetConfig] = dataclasses.field(default_factory=list)


@DATAMODULE.register(config_cls=DataModuleConfig)
class TrainingDataModule(pl.LightningDataModule):
    # HACK: (SeonghwanSeo): currently only supports a single Boltz dataset
    # I'll remove this datamodule and make it better (multiple dataset)
    _train_ds: MultiTrainingDataset
    _val_ds: ValidationDataset

    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()
        self.config = config

        # Load CCD
        self.ccd: CCD = CCD.load(config.ccd_path)
        self.logger = logging.getLogger("[DataModule]")

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self._train_ds = self.construct_train_dataset()
            self._val_ds = self.construct_val_dataset()
        elif stage == "validate":
            self._val_ds = self.construct_val_dataset()
        else:
            raise NotImplementedError("Not implemented yet.")

    def construct_train_dataset(self) -> MultiTrainingDataset:
        """Construct training dataset."""
        multi_ds = MultiTrainingDataset(
            configs=self.config.train_datasets,
            ccd=self.ccd,
            max_chains=self.config.max_chains,
            max_tokens=self.config.max_tokens,
            safe_load=self.config.safe_load,
        )
        # Print dataset info
        for d in multi_ds.datasets:
            self.print_rank_zero(
                f"Constructed training dataset '{d.name}':\n"
                f"  Weights: {d.config.weight}\n"
                f"  Num complexes: {len(d.metadatas)}\n"
                f"  Num samples: {len(d)}"
            )
        return multi_ds

    def construct_val_dataset(self) -> ValidationDataset:
        # TODO: (SeonghwanSeo): currently only supports a single validation dataset
        if len(self.config.val_datasets) != 1:
            raise NotImplementedError(
                "Currently only single validation dataset is supported."
            )

        ds = ValidationDataset(
            config=self.config.val_datasets[0],
            ccd=self.ccd,
            safe_load=self.config.safe_load,
        )
        self.print_rank_zero(
            f"Constructed validation dataset '{ds.name}':\n"
            f"  Num complexes: {len(ds.metadatas)}\n"
        )
        return ds

    def train_dataloader(self):
        dataset = self._train_ds

        sampler = DistributedWeightedSampler(
            weights=dataset.weights,
            rank=self.trainer.global_rank if self.trainer else 0,
            world_size=self.trainer.world_size if self.trainer else 1,
            epoch=self.trainer.current_epoch if self.trainer else 0,
            replacement=True,
        )
        persistent_workers = (
            self.config.persistent_workers and self.config.num_workers > 0
        )
        return DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
            collate_fn=collate,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=persistent_workers,
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
            pin_memory=False,
            persistent_workers=False,
        )

    def print_rank_zero(self, msg: str) -> None:
        if self.trainer is None or self.trainer.global_rank == 0:
            self.logger.info(f"{msg}")
