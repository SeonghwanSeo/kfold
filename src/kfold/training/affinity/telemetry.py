"""Rank-zero runtime and GPU-memory telemetry for affinity training runs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch


def validate_direct_activity_cliff_batch(batch: dict[str, Any]) -> None:
    """Require the production 12x5 assay + 4 BindingDB B64 layout."""
    valid = batch["valid_mask"].bool()
    if valid.shape != (64,) or not bool(valid.all()):
        raise ValueError("Direct affinity smoke batches must contain 64 valid labels.")
    group_index = batch["group_index"].long()
    unique, counts = torch.unique(group_index, return_counts=True)
    count_values = sorted(int(value) for value in counts.tolist())
    if len(unique) != 16 or count_values != [1] * 4 + [5] * 12:
        raise ValueError("Affinity smoke batch is not 12 assay groups x 5 + 4 singles.")
    origins = [str(value) for value in batch["origins"]]
    for group in unique.tolist():
        positions = torch.nonzero(group_index == group, as_tuple=False).flatten().tolist()
        if len(positions) == 1:
            if origins[positions[0]] != "BindingDB-residual":
                raise ValueError(
                    "Regression-only smoke slots must be BindingDB residual."
                )
        elif any(origins[position] == "BindingDB-residual" for position in positions):
            raise ValueError("BindingDB residual labels cannot enter ranking groups.")


class AffinityPerformanceSmokeCallback(pl.Callback):
    """Measure 100 post-warmup optimizer steps and enforce the B64 speed gate."""

    def __init__(
        self,
        *,
        report_path: str | Path,
        warmup_steps: int,
        measured_steps: int,
        max_p50_seconds: float,
        max_p95_seconds: float,
        max_data_wait_fraction: float,
    ) -> None:
        super().__init__()
        if warmup_steps < 0 or measured_steps <= 0:
            raise ValueError("Performance smoke step counts are invalid.")
        self.report_path = Path(report_path)
        self.warmup_steps = warmup_steps
        self.measured_steps = measured_steps
        self.max_p50_seconds = max_p50_seconds
        self.max_p95_seconds = max_p95_seconds
        self.max_data_wait_fraction = max_data_wait_fraction
        self._batch_started_at: float | None = None
        self._previous_batch_ended_at: float | None = None
        self._pending_wait = 0.0
        self._step_seconds: list[float] = []
        self._wait_seconds: list[float] = []
        self._finite_gradients = False
        self._validated_layout = False

    @staticmethod
    def _synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if trainer.world_size != 1:
            raise ValueError("The direct B64 performance smoke requires one GPU.")

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch_idx
        if not self._validated_layout:
            validate_direct_activity_cliff_batch(batch)
            self._validated_layout = True
        self._synchronize()
        now = time.perf_counter()
        self._pending_wait = (
            0.0
            if self._previous_batch_ended_at is None
            else now - self._previous_batch_ended_at
        )
        self._batch_started_at = now

    def on_after_backward(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if int(trainer.global_step) + 1 != self.warmup_steps + self.measured_steps:
            return
        self._finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in pl_module.parameters()
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        self._synchronize()
        now = time.perf_counter()
        if self._batch_started_at is None:
            raise RuntimeError("Performance smoke batch timing did not start.")
        step = int(trainer.global_step)
        if isinstance(outputs, torch.Tensor) and not bool(torch.isfinite(outputs).all()):
            raise ValueError("Performance smoke produced a non-finite loss.")
        if step > self.warmup_steps:
            self._step_seconds.append(now - self._batch_started_at)
            self._wait_seconds.append(self._pending_wait)
        self._previous_batch_ended_at = now

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        if len(self._step_seconds) != self.measured_steps:
            raise RuntimeError(
                "Performance smoke did not measure the requested optimizer steps: "
                f"{len(self._step_seconds)} != {self.measured_steps}."
            )
        p50 = self._percentile(self._step_seconds, 0.50)
        p95 = self._percentile(self._step_seconds, 0.95)
        step_total = sum(self._step_seconds)
        wait_total = sum(self._wait_seconds)
        data_wait_fraction = wait_total / max(step_total + wait_total, 1e-12)
        passed = (
            p50 <= self.max_p50_seconds
            and p95 <= self.max_p95_seconds
            and data_wait_fraction <= self.max_data_wait_fraction
            and self._finite_gradients
        )
        payload = {
            "schema_version": "affinity_direct_b64_performance_smoke_v1",
            "state": "passed" if passed else "failed",
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "step_seconds_p50": p50,
            "step_seconds_p95": p95,
            "data_wait_fraction": data_wait_fraction,
            "finite_gradients": self._finite_gradients,
            "max_memory_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available()
                else 0.0
            ),
            "gates": {
                "max_p50_seconds": self.max_p50_seconds,
                "max_p95_seconds": self.max_p95_seconds,
                "max_data_wait_fraction": self.max_data_wait_fraction,
            },
        }
        self.report_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not passed:
            raise RuntimeError("Direct affinity B64 performance acceptance gate failed.")


class AffinityRunTelemetryCallback(pl.Callback):
    """Log throughput/ETA and write one immutable rank-zero run summary."""

    def __init__(self, *, summary_path: str | Path, log_every_n_steps: int) -> None:
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive.")
        self.summary_path = Path(summary_path)
        self.log_every_n_steps = log_every_n_steps
        self._started_at: float | None = None
        self._last_log_at: float | None = None
        self._last_log_step = 0

    @staticmethod
    def _gpu_metrics() -> dict[str, float | str]:
        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return {
            "gpu/name": properties.name,
            "gpu/memory_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
            "gpu/memory_reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
            "gpu/max_memory_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20
            ),
            "gpu/max_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self._started_at = time.perf_counter()
        self._last_log_at = self._started_at
        self._last_log_step = int(trainer.global_step)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step <= 0 or step % self.log_every_n_steps:
            return
        assert self._last_log_at is not None
        now = time.perf_counter()
        elapsed = now - self._last_log_at
        completed = step - self._last_log_step
        if elapsed <= 0 or completed <= 0:
            return
        steps_per_second = completed / elapsed
        remaining = max(int(trainer.max_steps) - step, 0)
        metrics: dict[str, float | str] = {
            "runtime/steps_per_second": steps_per_second,
            "runtime/estimated_remaining_seconds": remaining / steps_per_second,
            **self._gpu_metrics(),
        }
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=step)
        self._last_log_at = now
        self._last_log_step = step

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        assert self._started_at is not None
        final_metrics: dict[str, float] = {}
        for name, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                scalar = float(value.detach().float().cpu())
                if torch.isfinite(torch.tensor(scalar)):
                    final_metrics[name] = scalar
        payload: dict[str, object] = {
            "status": "completed",
            "global_step": int(trainer.global_step),
            "elapsed_seconds": time.perf_counter() - self._started_at,
            **self._gpu_metrics(),
        }
        payload["final_metrics"] = final_metrics
        self.summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
