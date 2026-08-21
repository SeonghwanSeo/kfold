import dataclasses
import gc
import logging
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kfold.data.pipelines import featurization, prior_sampling, tokenization
from kfold.data.types.ccd import CCD
from kfold.data.types.model_input import FoldingInput
from kfold.utils.registry import DATAMODULE, BaseConfig

from .datasets import (
    TrainingDataset,
    TrainingDatasetConfig,
    ValidationDataset,
    ValidationDatasetConfig,
    get_training_dataset_cls,
)
from .dl_sampler import DistributedWeightedSampler


def collate(batches: list[tuple[FoldingInput, dict]]) -> tuple[FoldingInput, list[dict]]:
    f_input_batched = FoldingInput.from_list([b[0] for b in batches], pad_to_max=False)
    meta_infos = [b[1] for b in batches]
    return f_input_batched, meta_infos


class DataModuleConfig(BaseConfig):
    # === Common config for data modules === #
    train_batch_size: int = 1
    num_workers: int = 0
    persistent_workers: bool = True
    pin_memory: bool = True
    safe_load: bool = True

    # === Training hyperparameters === #
    max_chains: int = 20
    max_apo: int = 5
    max_tokens: int = 384
    max_atoms: int = 4608
    max_sequence_tokens: int = 768

    # === CCD path === #
    ccd_path: str | Path

    # === Dataset configs === #
    data_root: str | Path
    train_datasets: list[TrainingDatasetConfig] = dataclasses.field(default_factory=list)
    val_datasets: list[ValidationDatasetConfig] = dataclasses.field(default_factory=list)

    # === Other configs === #
    prior_sampler: prior_sampling.PriorSamplerConfig = dataclasses.field(
        default_factory=prior_sampling.PriorSamplerConfig
    )


class MultiTrainingDataset(torch.utils.data.Dataset):
    """A dataset that combines multiple training datasets with different weights."""

    def __init__(
        self,
        configs: list[TrainingDatasetConfig],
        ccd: CCD,
        tokenizer: tokenization.Tokenizer,
        featurizer: featurization.InputFeaturizer,
        prior_sampler: prior_sampling.PriorSampler | None,
        max_chains: int,
        max_apo: int,
        max_tokens: int,
        max_atoms: int,
        max_sequence_tokens: int,
        safe_load: bool = True,
    ) -> None:
        self.datasets: list[TrainingDataset] = [
            get_training_dataset_cls(config)(
                config=config,
                ccd=ccd,
                tokenizer=tokenizer,
                featurizer=featurizer,
                prior_sampler=prior_sampler,
                safe_load=safe_load,
                max_chains=max_chains,
                max_apo=max_apo,
                max_tokens=max_tokens,
                max_atoms=max_atoms,
                max_sequence_tokens=max_sequence_tokens,
            )
            for config in configs
        ]
        self.cumulative_sizes: np.ndarray = np.cumsum([len(ds) for ds in self.datasets])
        self.weights: np.ndarray = np.concatenate(
            [
                config.weight * (ds.weights / ds.weights.sum())
                for ds, config in zip(self.datasets, configs, strict=True)
            ]
        )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> tuple[FoldingInput, dict]:
        # Find the dataset index
        dataset_idx = np.searchsorted(self.cumulative_sizes, index, side="right")
        # Find the sample index within the dataset
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


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

        # Update dataset configs with data root
        for cfg in self.config.train_datasets:
            if cfg.data_path is None:
                cfg.data_path = os.path.join(self.config.data_root, cfg.name)
        for cfg in self.config.val_datasets:
            if cfg.data_path is None:
                cfg.data_path = os.path.join(self.config.data_root, cfg.name)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self._train_ds = self.construct_train_dataset()
            self._val_ds = self.construct_val_dataset()
        elif stage == "validate":
            self._val_ds = self.construct_val_dataset()
        else:
            raise NotImplementedError("Not implemented yet.")
        gc.collect()
        gc.freeze()

    def construct_train_dataset(self) -> MultiTrainingDataset:
        """Construct training dataset."""
        if hasattr(self, "_train_ds"):
            return self._train_ds

        tokenizer = tokenization.Tokenizer(self.ccd, mode="train")
        featurizer = featurization.InputFeaturizer()
        prior_sampler = prior_sampling.PriorSampler(self.config.prior_sampler)

        multi_ds = MultiTrainingDataset(
            configs=self.config.train_datasets,
            tokenizer=tokenizer,
            featurizer=featurizer,
            prior_sampler=prior_sampler,
            ccd=self.ccd,
            max_chains=self.config.max_chains,
            max_apo=self.config.max_apo,
            max_tokens=self.config.max_tokens,
            max_atoms=self.config.max_atoms,
            max_sequence_tokens=self.config.max_sequence_tokens,
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
        """Construct validation dataset."""
        if hasattr(self, "_val_ds"):
            return self._val_ds

        # TODO: (SeonghwanSeo): currently only supports a single validation dataset
        if len(self.config.val_datasets) != 1:
            raise NotImplementedError(
                "Currently only single validation dataset is supported."
            )

        tokenizer = tokenization.Tokenizer(self.ccd, mode="train")
        featurizer = featurization.InputFeaturizer()
        prior_sampler = prior_sampling.PriorSampler.inference_mode()
        prior_sampler.chain_translation_scale = (
            self.config.prior_sampler.chain_translation_scale
        )

        ds = ValidationDataset(
            config=self.config.val_datasets[0],
            tokenizer=tokenizer,
            featurizer=featurizer,
            prior_sampler=prior_sampler,
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
            batch_size=None,  # No batching for validation
            sampler=sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )

    def print_rank_zero(self, msg: str) -> None:
        if self.trainer is None or self.trainer.global_rank == 0:
            self.logger.info(f"{msg}")
