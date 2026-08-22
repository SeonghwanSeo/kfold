import hashlib
import json

import pytest

from scripts.affinity.train_affinity import (
    AtomicMilestoneCheckpoint,
    configure_torch_runtime,
)


class _FakeStrategy:
    def __init__(self) -> None:
        self.barriers: list[str] = []

    def barrier(self, name: str) -> None:
        self.barriers.append(name)


class _FakeTrainer:
    def __init__(self, *, global_step: int = 0) -> None:
        self.global_step = global_step
        self.is_global_zero = True
        self.world_size = 32
        self.strategy = _FakeStrategy()
        self.save_calls: list[tuple[str, bool]] = []

    def save_checkpoint(self, path: str, *, weights_only: bool) -> None:
        self.save_calls.append((path, weights_only))
        with open(path, "wb") as handle:
            handle.write(f"checkpoint-step-{self.global_step}".encode())


def test_atomic_milestone_publishes_exact_step_checksum_marker(tmp_path) -> None:
    run_metadata = tmp_path / "run_metadata.json"
    run_metadata.write_text('{"source_commit":"abc123"}\n')
    callback = AtomicMilestoneCheckpoint(
        dirpath=tmp_path / "checkpoints",
        milestones=[400, 1000],
        run_metadata_path=run_metadata,
    )
    trainer = _FakeTrainer(global_step=399)
    callback.on_fit_start(trainer, None)

    callback.on_train_batch_end(trainer, None, None, None, 398)
    assert trainer.save_calls == []

    trainer.global_step = 400
    callback.on_train_batch_end(trainer, None, None, None, 399)
    checkpoint = tmp_path / "checkpoints/milestone-step=00000400.ckpt"
    marker_path = tmp_path / "checkpoints/milestone-step=00000400.ckpt.complete.json"
    assert checkpoint.is_file()
    assert marker_path.is_file()
    assert not (
        tmp_path / "checkpoints/.milestone-step=00000400.ckpt.incomplete"
    ).exists()
    marker = json.loads(marker_path.read_text())
    assert marker["schema_version"] == "affinity_milestone_checkpoint_v1"
    assert marker["global_step"] == 400
    assert marker["world_size"] == 32
    assert marker["checkpoint"] == checkpoint.name
    assert marker["checkpoint_bytes"] == checkpoint.stat().st_size
    assert (
        marker["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert marker["run_metadata"] == "../run_metadata.json"
    assert (
        marker["run_metadata_sha256"]
        == hashlib.sha256(run_metadata.read_bytes()).hexdigest()
    )

    callback.on_train_batch_end(trainer, None, None, None, 399)
    assert len(trainer.save_calls) == 1


@pytest.mark.parametrize("milestones", [[0], [-1], [400, 400]])
def test_atomic_milestone_rejects_invalid_steps(tmp_path, milestones) -> None:
    with pytest.raises(ValueError):
        AtomicMilestoneCheckpoint(dirpath=tmp_path, milestones=milestones)


def test_affinity_torch_runtime_is_driven_by_resolved_config(monkeypatch) -> None:
    observed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "scripts.affinity.train_affinity.torch.multiprocessing.get_all_sharing_strategies",
        lambda: {"file_descriptor", "file_system"},
    )
    monkeypatch.setattr(
        "scripts.affinity.train_affinity.torch.multiprocessing.set_sharing_strategy",
        lambda value: observed.append(("sharing", value)),
    )
    monkeypatch.setattr(
        "scripts.affinity.train_affinity.torch.set_float32_matmul_precision",
        lambda value: observed.append(("matmul", value)),
    )

    configure_torch_runtime(
        multiprocessing_sharing_strategy="file_system",
        float32_matmul_precision="high",
    )

    assert observed == [("sharing", "file_system"), ("matmul", "high")]


@pytest.mark.parametrize(
    ("sharing_strategy", "matmul_precision"),
    [("unsupported", "high"), ("file_system", "unsupported")],
)
def test_affinity_torch_runtime_rejects_unknown_config_values(
    monkeypatch, sharing_strategy, matmul_precision
) -> None:
    monkeypatch.setattr(
        "scripts.affinity.train_affinity.torch.multiprocessing.get_all_sharing_strategies",
        lambda: {"file_system"},
    )
    with pytest.raises(ValueError):
        configure_torch_runtime(
            multiprocessing_sharing_strategy=sharing_strategy,
            float32_matmul_precision=matmul_precision,
        )
