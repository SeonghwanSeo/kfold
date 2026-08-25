import pytest
import torch

from kfold.training.affinity.loss import (
    boltz2_continuous_affinity_loss,
    grouped_affinity_loss,
)


def test_singleton_group_is_regression_only() -> None:
    result = grouped_affinity_loss(
        torch.tensor([5.0]),
        torch.tensor([6.0]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([True]),
    )
    assert result.active_ranking_groups == 0
    assert result.ranking == 0


def test_two_sample_group_uses_pairwise_except_near_tie() -> None:
    result = grouped_affinity_loss(
        torch.tensor([2.0, 1.0]),
        torch.tensor([8.0, 6.0]),
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        torch.tensor([True, True]),
    )
    assert result.pairwise_groups == 1
    assert result.ranking < 0.01
    near_tie = grouped_affinity_loss(
        torch.tensor([2.0, 1.0]),
        torch.tensor([8.0, 7.95]),
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        torch.tensor([True, True]),
    )
    assert near_tie.active_ranking_groups == 0


def test_three_sample_group_uses_pearson_and_constant_targets_are_masked() -> None:
    result = grouped_affinity_loss(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0, 6.0]),
        torch.tensor([0, 0, 0]),
        torch.tensor([0, 1, 2]),
        torch.tensor([True, True, True]),
    )
    assert result.pearson_groups == 1
    assert result.ranking < 1e-6
    constant = grouped_affinity_loss(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 4.0, 4.0]),
        torch.tensor([0, 0, 0]),
        torch.tensor([0, 1, 2]),
        torch.tensor([True, True, True]),
    )
    assert constant.active_ranking_groups == 0
    assert torch.isfinite(constant.loss)


def test_same_ligand_replicates_are_regression_only_for_ranking() -> None:
    result = grouped_affinity_loss(
        torch.tensor([3.0, -3.0]),
        torch.tensor([8.0, 6.0]),
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        torch.tensor([True, True]),
    )
    assert result.active_ranking_groups == 0
    assert result.ranking_replicate_records == 1
    assert torch.isfinite(result.regression)


def test_boltz2_continuous_loss_uses_same_group_differences_with_strong_weight() -> None:
    result = boltz2_continuous_affinity_loss(
        torch.tensor([1.0, 3.0, 2.0]),
        torch.tensor([1.0, 3.0, 2.0]),
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 1, 2]),
        torch.tensor([True, True, True]),
    )
    assert result.loss == 0
    assert result.active_ranking_groups == 1
    assert result.difference_pairs == 1

    weighted = boltz2_continuous_affinity_loss(
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 2.0]),
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        torch.tensor([True, True]),
    )
    assert weighted.ranking > weighted.regression
    assert weighted.loss == pytest.approx(
        0.1 * weighted.regression + 0.9 * weighted.ranking
    )


def test_boltz2_continuous_loss_collapses_replicates_before_differences() -> None:
    result = boltz2_continuous_affinity_loss(
        torch.tensor([1.0, 3.0]),
        torch.tensor([1.0, 3.0]),
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        torch.tensor([True, True]),
    )
    assert result.active_ranking_groups == 0
    assert result.difference_pairs == 0
    assert result.ranking_replicate_records == 1
