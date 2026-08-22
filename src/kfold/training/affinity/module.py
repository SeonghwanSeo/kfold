"""Lightning task for the cached frozen Stage-2 affinity ranking head."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F

from kfold.model.modules.affinity_pairformer import AffinityPairformer

from .loss import boltz2_continuous_affinity_loss, grouped_affinity_loss


@dataclass(frozen=True, kw_only=True)
class AffinityRankingConfig:
    mode: str = "boltz2_continuous"
    version: str = "boltz2_continuous_affinity_v1"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    huber_delta: float = 0.5
    difference_weight: float = 0.9
    absolute_weight: float = 0.1
    ranking_weight: float = 0.5
    pairwise_temperature: float = 0.2
    near_tie_delta: float = 0.1


def validation_metrics(
    records: list[tuple[str, str, float, float]],
    *,
    huber_delta: float = 0.5,
    near_tie_delta: float = 0.1,
) -> dict[str, float | int]:
    """Compute exhaustive label and within-assay validation metrics."""
    if not records:
        return {
            "mean_assay_pearson": float("nan"),
            "huber": float("nan"),
            "mae": float("nan"),
            "pairwise_accuracy": float("nan"),
            "pearson_assays": 0,
            "pairwise_comparisons": 0,
            "ranking_replicate_records": 0,
        }
    target = torch.tensor([record[3] for record in records], dtype=torch.float32)
    prediction = torch.tensor([record[2] for record in records], dtype=torch.float32)
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for assay, ligand, pred, label in records:
        grouped[assay][ligand].append((pred, label))
    pearsons: list[float] = []
    pairwise_correct = 0
    pairwise_total = 0
    ranking_replicate_records = 0
    for by_ligand in grouped.values():
        ranking_replicate_records += sum(len(values) - 1 for values in by_ligand.values())
        values = [
            (
                sum(prediction for prediction, _ in replicates) / len(replicates),
                sum(label for _, label in replicates) / len(replicates),
            )
            for _, replicates in sorted(by_ligand.items())
        ]
        pred = torch.tensor([value[0] for value in values], dtype=torch.float32)
        label = torch.tensor([value[1] for value in values], dtype=torch.float32)
        if len(values) >= 3:
            pred_centered = pred - pred.mean()
            label_centered = label - label.mean()
            denominator = pred_centered.norm() * label_centered.norm()
            if denominator > torch.finfo(torch.float32).eps:
                pearsons.append(
                    float((pred_centered * label_centered).sum() / denominator)
                )
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                delta = label[left] - label[right]
                if abs(float(delta)) < near_tie_delta:
                    continue
                pairwise_total += 1
                pairwise_correct += int(
                    torch.sign(delta) == torch.sign(pred[left] - pred[right])
                )
    return {
        "mean_assay_pearson": (
            sum(pearsons) / len(pearsons) if pearsons else float("nan")
        ),
        "huber": float(F.huber_loss(prediction, target, delta=huber_delta)),
        "mae": float(F.l1_loss(prediction, target)),
        "pairwise_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else float("nan")
        ),
        "pearson_assays": len(pearsons),
        "pairwise_comparisons": pairwise_total,
        "ranking_replicate_records": ranking_replicate_records,
    }


class AffinityRankingModule(pl.LightningModule):
    """Train one scalar head from cache; the Stage-2 backbone is not present."""

    def __init__(
        self,
        *,
        model_config: AffinityPairformer.Config,
        task_config: AffinityRankingConfig | None = None,
        use_kernels: bool = False,
    ) -> None:
        super().__init__()
        self.model = AffinityPairformer(model_config, use_kernels=use_kernels)
        self.task_config = task_config or AffinityRankingConfig()
        self._validation_records: list[tuple[str, str, float, float]] = []
        # The resolved run config already records these frozen dataclasses.
        # Lightning's CSV logger recursively mutates hparams while serializing,
        # so retaining either dataclass here prevents trainer startup.
        self.save_hyperparameters(ignore=["model_config", "task_config"])

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.model(
            s_inputs=batch["s_inputs"],
            s_lm=batch["s_lm"],
            z=batch["z"],
            distogram_features=batch["distogram_features"],
            token_mask=batch["token_mask"],
            protein_mask=batch["protein_mask"],
            ligand_mask=batch["ligand_mask"],
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        prediction = self(batch)
        if self.task_config.mode == "boltz2_continuous":
            objective = boltz2_continuous_affinity_loss(
                prediction,
                batch["target"],
                batch["group_index"],
                batch["ligand_index"],
                batch["valid_mask"],
                huber_delta=self.task_config.huber_delta,
                difference_weight=self.task_config.difference_weight,
                absolute_weight=self.task_config.absolute_weight,
            )
        elif self.task_config.mode == "legacy_huber_ranking":
            objective = grouped_affinity_loss(
                prediction,
                batch["target"],
                batch["group_index"],
                batch["ligand_index"],
                batch["valid_mask"],
                huber_delta=self.task_config.huber_delta,
                ranking_weight=self.task_config.ranking_weight,
                pairwise_temperature=self.task_config.pairwise_temperature,
                near_tie_delta=self.task_config.near_tie_delta,
            )
        else:
            raise ValueError(
                f"Unsupported affinity loss mode: {self.task_config.mode!r}."
            )
        self.log(
            "train/loss", objective.loss, on_step=True, on_epoch=True, sync_dist=True
        )
        self.log(
            "train/huber",
            objective.regression,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/ranking",
            objective.ranking,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/active_ranking_groups",
            float(objective.active_ranking_groups),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/ranking_replicate_records",
            float(objective.ranking_replicate_records),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/difference_pairs",
            float(objective.difference_pairs),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        valid = batch["valid_mask"]
        for name, mask in (
            ("crop_tokens", batch["token_mask"]),
            ("protein_tokens", batch["protein_mask"]),
            ("ligand_tokens", batch["ligand_mask"]),
        ):
            self.log(
                f"train/{name}_mean",
                mask[valid].sum(dim=-1).float().mean(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        return objective.loss

    def on_train_epoch_start(self) -> None:
        datamodule = self.trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "set_train_epoch"):
            datamodule.set_train_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self._validation_records.clear()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        prediction = self(batch)
        valid = batch["valid_mask"]
        for assay, ligand, pred, target, is_valid in zip(
            batch["assay_keys"],
            batch["canonical_smiles"],
            prediction.detach().float().cpu().tolist(),
            batch["target"].detach().float().cpu().tolist(),
            valid.detach().cpu().tolist(),
            strict=True,
        ):
            if is_valid:
                assert assay is not None
                assert ligand is not None
                self._validation_records.append((assay, ligand, pred, target))

    def on_validation_epoch_end(self) -> None:
        records = self._validation_records
        if dist.is_available() and dist.is_initialized():
            all_records: list[list[tuple[str, str, float, float]] | None] = [
                None for _ in range(dist.get_world_size())
            ]
            dist.all_gather_object(all_records, records)
            records = [record for part in all_records if part for record in part]
        metrics = validation_metrics(
            records,
            huber_delta=self.task_config.huber_delta,
            near_tie_delta=self.task_config.near_tie_delta,
        )
        for name in ("mean_assay_pearson", "huber", "mae", "pairwise_accuracy"):
            value = metrics[name]
            if isinstance(value, float) and torch.isfinite(torch.tensor(value)):
                self.log(f"val/{name}", value, on_epoch=True, sync_dist=True)
        self.log(
            "val/pearson_assays",
            float(metrics["pearson_assays"]),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/pairwise_comparisons",
            float(metrics["pairwise_comparisons"]),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/ranking_replicate_records",
            float(metrics["ranking_replicate_records"]),
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.task_config.learning_rate,
            weight_decay=self.task_config.weight_decay,
        )
