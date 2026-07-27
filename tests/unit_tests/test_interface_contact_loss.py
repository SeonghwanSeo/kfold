import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from kfold.training.loss.interface_contact import InterfaceContactBalancedLoss

_NUM_BINS = 64
_NUM_CONTACT_BINS = 19


def _folding_input(
    coordinates: list[list[float]],
    asym_id: list[int],
) -> SimpleNamespace:
    num_tokens = len(coordinates)
    token = SimpleNamespace(
        repr_coords=torch.tensor([coordinates], dtype=torch.float32),
        repr_mask=torch.ones((1, num_tokens), dtype=torch.bool),
        asym_id=torch.tensor([asym_id], dtype=torch.long),
        chain_type=torch.zeros((1, num_tokens), dtype=torch.long),
    )
    return SimpleNamespace(token=token)


def _logits_with_contact_probabilities(
    num_tokens: int,
    probabilities: dict[tuple[int, int], float],
) -> torch.Tensor:
    logits = torch.zeros((1, num_tokens, num_tokens, _NUM_BINS))
    for (token_i, token_j), probability in probabilities.items():
        logits[0, token_i, token_j, :_NUM_CONTACT_BINS] = math.log(
            probability / _NUM_CONTACT_BINS
        )
        logits[0, token_i, token_j, _NUM_CONTACT_BINS:] = math.log(
            (1.0 - probability) / (_NUM_BINS - _NUM_CONTACT_BINS)
        )
    return logits.requires_grad_()


def test_interface_contact_loss_matches_detached_normalized_formula_and_gradient() -> (
    None
):
    loss_fn = InterfaceContactBalancedLoss()
    f_input = _folding_input(
        coordinates=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ],
        asym_id=[1, 1, 2, 2],
    )
    pair_probability = {
        (0, 2): 0.20,
        (1, 2): 0.65,
        (0, 3): 0.10,
        (1, 3): 0.55,
    }
    logits = _logits_with_contact_probabilities(
        num_tokens=4,
        probabilities=pair_probability,
    )

    loss, metrics = loss_fn(logits, f_input)

    positive_probability = torch.tensor([0.20, 0.65])
    far_probability = torch.tensor([0.10, 0.55])
    positive_weight = (1.0 - positive_probability).pow(2)
    far_weight = far_probability.pow(2)
    positive_denominator = positive_weight.sum() + loss_fn.eps
    far_denominator = torch.tensor(2.0 + loss_fn.eps)
    expected_fn = (
        positive_weight * -torch.log(positive_probability)
    ).sum() / positive_denominator
    expected_fp = (far_weight * -torch.log1p(-far_probability)).sum() / far_denominator
    expected_loss = 0.75 * expected_fn + 0.25 * expected_fp

    assert torch.isclose(loss, expected_loss, atol=2e-6)
    assert torch.isclose(
        metrics["interface_contact_fn_loss"],
        expected_fn,
        atol=2e-6,
    )
    assert torch.isclose(
        metrics["interface_contact_fp_loss"],
        expected_fp,
        atol=2e-6,
    )
    assert metrics["interface_contact_positive_pairs"].item() == 2
    assert metrics["interface_contact_far_pairs"].item() == 2
    assert metrics["interface_contact_selected_far_pairs"].item() == 2
    assert metrics["interface_contact_active_interfaces"].item() == 1

    loss.backward()
    assert logits.grad is not None
    for pair_index, probability, weight in zip(
        ((0, 2), (1, 2)),
        positive_probability,
        positive_weight,
        strict=True,
    ):
        expected_gradient = 0.75 * weight / positive_denominator * (probability - 1.0)
        actual_gradient = logits.grad[0, *pair_index, :_NUM_CONTACT_BINS].sum()
        assert torch.isclose(actual_gradient, expected_gradient, atol=2e-6)
    for pair_index, probability, weight in zip(
        ((0, 3), (1, 3)),
        far_probability,
        far_weight,
        strict=True,
    ):
        expected_gradient = 0.25 * weight / far_denominator * probability
        actual_gradient = logits.grad[0, *pair_index, :_NUM_CONTACT_BINS].sum()
        assert torch.isclose(actual_gradient, expected_gradient, atol=2e-6)

    standard_positive_probability = positive_probability.clone().requires_grad_()
    standard_far_probability = far_probability.clone().requires_grad_()
    standard_positive_weight = (1.0 - standard_positive_probability).pow(2)
    standard_far_weight = standard_far_probability.pow(2)
    standard_loss = 0.75 * (
        (standard_positive_weight * -torch.log(standard_positive_probability)).sum()
        / (standard_positive_weight.sum() + loss_fn.eps)
    ) + 0.25 * (
        (standard_far_weight * -torch.log1p(-standard_far_probability)).sum()
        / (2.0 + loss_fn.eps)
    )
    standard_loss.backward()
    standard_logit_gradient = (
        standard_positive_probability.grad
        * positive_probability
        * (1.0 - positive_probability)
    )
    actual_positive_gradient = torch.stack(
        [
            logits.grad[0, token_i, token_j, :_NUM_CONTACT_BINS].sum()
            for token_i, token_j in ((0, 2), (1, 2))
        ]
    )
    assert not torch.allclose(
        actual_positive_gradient,
        standard_logit_gradient,
        atol=2e-6,
    )
    plain_fn = -torch.log(positive_probability).mean()
    assert not torch.isclose(expected_fn, plain_fn, atol=1e-3)


def test_interface_contact_fp_tail_counts_pairs_until_budget_then_caps() -> None:
    loss_fn = InterfaceContactBalancedLoss()
    base_input = _folding_input(
        coordinates=[[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
        asym_id=[1, 2, 2],
    )
    budget_input = _folding_input(
        coordinates=[
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [60.0, 0.0, 0.0],
        ],
        asym_id=[1, 2, 2, 2, 2, 2],
    )
    overflow_input = _folding_input(
        coordinates=[
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [60.0, 0.0, 0.0],
            [70.0, 0.0, 0.0],
        ],
        asym_id=[1, 2, 2, 2, 2, 2, 2],
    )
    base_logits = _logits_with_contact_probabilities(
        num_tokens=3,
        probabilities={(0, 1): 0.4, (0, 2): 0.05},
    )
    budget_logits = _logits_with_contact_probabilities(
        num_tokens=6,
        probabilities={
            (0, 1): 0.4,
            (0, 2): 0.05,
            (0, 3): 0.05,
            (0, 4): 0.05,
            (0, 5): 0.05,
        },
    )
    overflow_logits = _logits_with_contact_probabilities(
        num_tokens=7,
        probabilities={
            (0, 1): 0.4,
            (0, 2): 0.05,
            (0, 3): 0.05,
            (0, 4): 0.05,
            (0, 5): 0.05,
            (0, 6): 0.01,
        },
    )

    base_loss, base_metrics = loss_fn(base_logits, base_input)
    budget_loss, budget_metrics = loss_fn(budget_logits, budget_input)
    overflow_loss, overflow_metrics = loss_fn(overflow_logits, overflow_input)

    assert budget_loss > base_loss
    assert torch.isclose(budget_loss, overflow_loss, atol=2e-6)
    assert base_metrics["interface_contact_selected_far_pairs"].item() == 1
    assert budget_metrics["interface_contact_selected_far_pairs"].item() == 4
    assert overflow_metrics["interface_contact_selected_far_pairs"].item() == 4


def test_interface_contact_gamma_zero_reduces_to_separate_bce_means() -> None:
    loss_fn = InterfaceContactBalancedLoss(focal_gamma=0.0)
    f_input = _folding_input(
        coordinates=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ],
        asym_id=[1, 1, 2, 2],
    )
    logits = _logits_with_contact_probabilities(
        num_tokens=4,
        probabilities={
            (0, 2): 0.20,
            (1, 2): 0.65,
            (0, 3): 0.10,
            (1, 3): 0.55,
        },
    )

    loss, _ = loss_fn(logits, f_input)

    positive_bce = -torch.log(torch.tensor([0.20, 0.65]))
    far_bce = -torch.log1p(-torch.tensor([0.10, 0.55]))
    expected = 0.75 * positive_bce.sum() / (2.0 + loss_fn.eps) + (
        0.25 * far_bce.sum() / (2.0 + loss_fn.eps)
    )
    assert torch.isclose(loss, expected, atol=2e-6)


def test_interface_contact_fp_gradient_only_uses_selected_tail() -> None:
    loss_fn = InterfaceContactBalancedLoss(fp_budget_ratio=1.0)
    f_input = _folding_input(
        coordinates=[
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
        ],
        asym_id=[1, 2, 2, 2],
    )
    logits = _logits_with_contact_probabilities(
        num_tokens=4,
        probabilities={
            (0, 1): 0.60,
            (0, 2): 0.80,
            (0, 3): 0.40,
        },
    )

    loss, metrics = loss_fn(logits, f_input)
    loss.backward()

    assert metrics["interface_contact_selected_far_pairs"].item() == 1
    assert logits.grad is not None
    selected_gradient = logits.grad[0, 0, 2, :_NUM_CONTACT_BINS].sum()
    expected_gradient = (
        0.25 * torch.tensor(0.80).pow(2) / (1.0 + loss_fn.eps) * torch.tensor(0.80)
    )
    assert torch.isclose(selected_gradient, expected_gradient, atol=2e-6)
    unselected_gradient = logits.grad[0, 0, 3, :_NUM_CONTACT_BINS].sum()
    assert unselected_gradient.item() == 0.0


def test_interface_contact_fp_gradient_survives_probability_saturation() -> None:
    loss_fn = InterfaceContactBalancedLoss()
    f_input = _folding_input(
        coordinates=[
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ],
        asym_id=[1, 2, 2],
    )
    logits = torch.zeros((1, 3, 3, _NUM_BINS))
    logits[0, 0, 2, :_NUM_CONTACT_BINS] = 50.0
    logits.requires_grad_()

    loss, metrics = loss_fn(logits, f_input)

    far_logits = logits[0, 0, 2]
    contact_logit = torch.logsumexp(far_logits[:_NUM_CONTACT_BINS], dim=-1)
    noncontact_logit = torch.logsumexp(far_logits[_NUM_CONTACT_BINS:], dim=-1)
    log_odds = contact_logit - noncontact_logit
    probability = torch.sigmoid(log_odds)
    expected_fp = (
        probability.pow(2) * F.softplus(log_odds) / (torch.tensor(1.0) + loss_fn.eps)
    )

    assert torch.isfinite(loss)
    assert torch.isclose(
        metrics["interface_contact_fp_loss"],
        expected_fp,
        atol=2e-6,
    )
    assert metrics["interface_contact_fp_loss"] > 40.0

    loss.backward()
    assert logits.grad is not None
    expected_gradient = (
        0.25 * probability.pow(2) * probability / (torch.tensor(1.0) + loss_fn.eps)
    )
    contact_gradient = logits.grad[0, 0, 2, :_NUM_CONTACT_BINS].sum()
    assert torch.isclose(contact_gradient, expected_gradient, atol=2e-6)


def test_interface_contact_zero_positive_interface_is_ignored() -> None:
    loss_fn = InterfaceContactBalancedLoss()
    f_input = _folding_input(
        coordinates=[[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
        asym_id=[1, 2],
    )
    logits = _logits_with_contact_probabilities(
        num_tokens=2,
        probabilities={(0, 1): 0.60},
    )

    loss, metrics = loss_fn(logits, f_input)

    assert loss.item() == 0.0
    assert metrics["interface_contact_fn_loss"].item() == 0.0
    assert metrics["interface_contact_fp_loss"].item() == 0.0
    assert metrics["interface_contact_selected_far_pairs"].item() == 0


def test_interface_contact_far_absent_keeps_fixed_mixture_weight() -> None:
    loss_fn = InterfaceContactBalancedLoss()
    f_input = _folding_input(
        coordinates=[[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        asym_id=[1, 2],
    )
    logits = _logits_with_contact_probabilities(
        num_tokens=2,
        probabilities={(0, 1): 0.35},
    )

    loss, metrics = loss_fn(logits, f_input)

    probability = torch.tensor(0.35)
    focal_weight = (1.0 - probability).pow(2)
    expected_fn = focal_weight * -torch.log(probability) / (focal_weight + loss_fn.eps)
    assert torch.isclose(loss, 0.75 * expected_fn, atol=2e-6)
    assert metrics["interface_contact_fp_loss"].item() == 0.0


def test_interface_contact_loss_keeps_zero_batch_connected_to_logits() -> None:
    loss_fn = InterfaceContactBalancedLoss()
    f_input = _folding_input(
        coordinates=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        asym_id=[1, 1],
    )
    logits = torch.zeros((1, 2, 2, 64), requires_grad=True)

    loss, metrics = loss_fn(logits, f_input)

    assert loss.item() == 0.0
    assert metrics["interface_contact_active_interfaces"].item() == 0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0


def test_interface_contact_loss_accepts_sparse_asym_ids() -> None:
    loss_fn = InterfaceContactBalancedLoss()
    coordinates = [
        [0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        [30.0, 0.0, 0.0],
    ]
    dense_input = _folding_input(
        coordinates=coordinates,
        asym_id=[1, 2, 2],
    )
    sparse_input = _folding_input(
        coordinates=coordinates,
        asym_id=[10_001, 80_002, 80_002],
    )
    logits = torch.zeros((1, 3, 3, 64))

    dense_loss, dense_metrics = loss_fn(logits, dense_input)
    sparse_loss, sparse_metrics = loss_fn(logits, sparse_input)

    assert torch.isclose(dense_loss, sparse_loss, atol=1e-6)
    assert (
        dense_metrics["interface_contact_active_interfaces"]
        == sparse_metrics["interface_contact_active_interfaces"]
    )
    assert (
        dense_metrics["interface_contact_positive_pairs"]
        == sparse_metrics["interface_contact_positive_pairs"]
    )
    assert (
        dense_metrics["interface_contact_far_pairs"]
        == sparse_metrics["interface_contact_far_pairs"]
    )
