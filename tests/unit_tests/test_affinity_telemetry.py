from __future__ import annotations

import pytest
import torch

from kfold.training.affinity.telemetry import validate_direct_activity_cliff_batch


def direct_batch() -> dict[str, object]:
    groups = torch.cat(
        (
            torch.arange(12).repeat_interleave(5),
            torch.arange(12, 16),
        )
    )
    return {
        "valid_mask": torch.ones(64, dtype=torch.bool),
        "group_index": groups,
        "origins": ["SAIR"] * 60 + ["BindingDB-residual"] * 4,
    }


def test_direct_activity_batch_requires_twelve_by_five_plus_four() -> None:
    validate_direct_activity_cliff_batch(direct_batch())


def test_direct_activity_batch_rejects_bindingdb_in_ranking_group() -> None:
    batch = direct_batch()
    batch["origins"][0] = "BindingDB-residual"
    with pytest.raises(ValueError, match="cannot enter ranking groups"):
        validate_direct_activity_cliff_batch(batch)


def test_direct_activity_batch_rejects_invalid_slot() -> None:
    batch = direct_batch()
    batch["valid_mask"][3] = False
    with pytest.raises(ValueError, match="64 valid labels"):
        validate_direct_activity_cliff_batch(batch)
