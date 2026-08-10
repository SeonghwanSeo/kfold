import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from kfold.training.loss.distogram import DistogramLoss
from kfold.training.loss.interface_contact import InterfaceContactBalancedLoss

_NUM_BINS = 64
_NUM_CONTACT_BINS = 19
_BIN_BOUNDARIES = torch.linspace(2.3125, 21.6875, _NUM_BINS - 1)


def _folding_input(
    coordinates: list[list[float]],
    asym_id: list[int],
) -> SimpleNamespace:
    num_tokens = len(coordinates)
    token = SimpleNamespace(
        repr_coords=torch.tensor([coordinates], dtype=torch.float32),
        repr_mask=torch.ones((1, num_tokens), dtype=torch.bool),
        asym_id=torch.tensor([asym_id], dtype=torch.long),
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


def _distogram_out(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "logits": logits,
        "bin_boundaries": _BIN_BOUNDARIES.to(logits.device),
    }


def _target_bin(
    f_input: SimpleNamespace,
    pair: tuple[int, int],
) -> torch.Tensor:
    token_i, token_j = pair
    distance = (
        f_input.token.repr_coords[0, token_i] - f_input.token.repr_coords[0, token_j]
    ).norm()
    return (distance > _BIN_BOUNDARIES).sum().long()


def test_exact_bin_formula_and_detached_gradient() -> None:
    loss_fn = InterfaceContactBalancedLoss()
    f_input = _folding_input(
        coordinates=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        asym_id=[1, 1, 2, 2],
    )
    probabilities = {
        (0, 2): 0.20,
        (1, 2): 0.65,
        (0, 3): 0.10,
        (1, 3): 0.55,
    }
    logits = _logits_with_contact_probabilities(4, probabilities)

    loss, metrics = loss_fn(_distogram_out(logits), f_input)

    reference_logits = logits.detach().clone().requires_grad_()
    log_prob = F.log_softmax(reference_logits, dim=-1)
    positive_pairs = ((0, 2), (1, 2))
    negative_pairs = ((0, 3), (1, 3))
    positive_probability = torch.stack(
        [
            log_prob[0, *pair, :_NUM_CONTACT_BINS].logsumexp(dim=-1).exp()
            for pair in positive_pairs
        ]
    )
    negative_probability = torch.stack(
        [
            log_prob[0, *pair, :_NUM_CONTACT_BINS].logsumexp(dim=-1).exp()
            for pair in negative_pairs
        ]
    )
    positive_ce = torch.stack(
        [-log_prob[0, *pair, _target_bin(f_input, pair)] for pair in positive_pairs]
    )
    negative_ce = torch.stack(
        [-log_prob[0, *pair, _target_bin(f_input, pair)] for pair in negative_pairs]
    )
    positive_focal = (1.0 - positive_probability).pow(2).detach()
    negative_focal = negative_probability.pow(2).detach()
    expected_fn = (positive_focal * positive_ce).sum() / (2.0 + loss_fn.eps)
    expected_fp = (negative_focal * negative_ce).sum() / (2.0 + loss_fn.eps)
    expected_loss = 0.75 * expected_fn + 0.25 * expected_fp
    expected_gradient = torch.autograd.grad(expected_loss, reference_logits)[0]

    torch.testing.assert_close(loss, expected_loss, atol=2e-6, rtol=0.0)
    torch.testing.assert_close(
        metrics["interface_contact_fn_loss"],
        expected_fn,
        atol=2e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        metrics["interface_contact_fp_loss"],
        expected_fp,
        atol=2e-6,
        rtol=0.0,
    )
    assert metrics["interface_contact_positive_pairs"].item() == 2
    assert metrics["interface_contact_negative_pairs"].item() == 2
    assert metrics["interface_contact_selected_negative_pairs"].item() == 2

    loss.backward()
    assert logits.grad is not None
    torch.testing.assert_close(
        logits.grad,
        expected_gradient,
        atol=2e-6,
        rtol=0.0,
    )

    normalized_fn = (positive_focal * positive_ce.detach()).sum() / (
        positive_focal.sum() + loss_fn.eps
    )
    binary_fn = (positive_focal * -positive_probability.detach().log()).sum() / (
        2.0 + loss_fn.eps
    )
    plain_fn = positive_ce.detach().mean()
    assert not torch.isclose(expected_fn.detach(), normalized_fn, atol=1e-3)
    assert not torch.isclose(expected_fn.detach(), binary_fn, atol=1e-3)
    assert not torch.isclose(expected_fn.detach(), plain_fn, atol=1e-3)


def test_positive_count_denominator_reduces_pressure_when_easy_pair_is_added() -> None:
    loss_fn = InterfaceContactBalancedLoss(
        positive_weight=1.0,
        negative_weight=0.0,
    )
    one_pair_input = _folding_input(
        coordinates=[[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        asym_id=[1, 2],
    )
    two_pair_input = _folding_input(
        coordinates=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        asym_id=[1, 1, 2],
    )
    one_pair_logits = _logits_with_contact_probabilities(
        2,
        {(0, 1): 0.20},
    )
    two_pair_logits = _logits_with_contact_probabilities(
        3,
        {(0, 2): 0.20, (1, 2): 0.999},
    )

    one_pair_loss, _ = loss_fn(_distogram_out(one_pair_logits), one_pair_input)
    two_pair_loss, _ = loss_fn(_distogram_out(two_pair_logits), two_pair_input)
    one_pair_loss.backward()
    two_pair_loss.backward()

    assert one_pair_logits.grad is not None
    assert two_pair_logits.grad is not None
    one_pair_gradient = one_pair_logits.grad[
        0,
        0,
        1,
        :_NUM_CONTACT_BINS,
    ].sum()
    two_pair_gradient = two_pair_logits.grad[
        0,
        0,
        2,
        :_NUM_CONTACT_BINS,
    ].sum()
    torch.testing.assert_close(
        two_pair_gradient,
        one_pair_gradient / 2.0,
        atol=2e-6,
        rtol=0.0,
    )


def test_selected_pair_gradients_are_collinear_with_distogram_ce() -> None:
    loss_fn = InterfaceContactBalancedLoss()
    distogram_loss_fn = DistogramLoss()
    f_input = _folding_input(
        coordinates=[
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        asym_id=[1, 2, 2],
    )
    auxiliary_logits = _logits_with_contact_probabilities(
        3,
        {(0, 1): 0.20, (0, 2): 0.80},
    )
    distogram_logits = auxiliary_logits.detach().clone().requires_grad_()

    auxiliary_loss, _ = loss_fn(_distogram_out(auxiliary_logits), f_input)
    distogram_loss = distogram_loss_fn(
        _distogram_out(distogram_logits),
        f_input,
    ).mean()
    auxiliary_loss.backward()
    distogram_loss.backward()

    assert auxiliary_logits.grad is not None
    assert distogram_logits.grad is not None
    for pair in ((0, 1), (0, 2)):
        auxiliary_gradient = auxiliary_logits.grad[0, *pair]
        distogram_gradient = distogram_logits.grad[0, *pair]
        cosine = F.cosine_similarity(
            auxiliary_gradient,
            distogram_gradient,
            dim=0,
        )
        torch.testing.assert_close(
            cosine,
            torch.tensor(1.0),
            atol=2e-6,
            rtol=0.0,
        )


def test_gamma_zero_is_selected_exact_bin_ce_mean() -> None:
    loss_fn = InterfaceContactBalancedLoss(focal_gamma=0.0)
    f_input = _folding_input(
        coordinates=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        asym_id=[1, 1, 2, 2],
    )
    logits = _logits_with_contact_probabilities(
        4,
        {
            (0, 2): 0.20,
            (1, 2): 0.65,
            (0, 3): 0.10,
            (1, 3): 0.55,
        },
    )

    loss, _ = loss_fn(_distogram_out(logits), f_input)

    log_prob = F.log_softmax(logits, dim=-1)
    positive_ce = torch.stack(
        [-log_prob[0, *pair, _target_bin(f_input, pair)] for pair in ((0, 2), (1, 2))]
    )
    negative_ce = torch.stack(
        [-log_prob[0, *pair, _target_bin(f_input, pair)] for pair in ((0, 3), (1, 3))]
    )
    expected = 0.75 * positive_ce.sum() / (
        2.0 + loss_fn.eps
    ) + 0.25 * negative_ce.sum() / (2.0 + loss_fn.eps)
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=0.0)


def test_top_negative_selects_high_probability_midrange_pair() -> None:
    loss_fn = InterfaceContactBalancedLoss(fp_budget_ratio=1.0)
    f_input = _folding_input(
        coordinates=[
            [0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ],
        asym_id=[1, 2, 2, 2],
    )
    logits = _logits_with_contact_probabilities(
        4,
        {
            (0, 1): 0.60,
            (0, 2): 0.80,
            (0, 3): 0.40,
        },
    )

    loss, metrics = loss_fn(_distogram_out(logits), f_input)
    loss.backward()

    assert metrics["interface_contact_negative_pairs"].item() == 2
    assert metrics["interface_contact_selected_negative_pairs"].item() == 1
    assert logits.grad is not None
    selected_gradient = logits.grad[0, 0, 2, :_NUM_CONTACT_BINS].sum()
    expected_gradient = (
        0.25 * torch.tensor(0.80).pow(2) * torch.tensor(0.80) / (1.0 + loss_fn.eps)
    )
    torch.testing.assert_close(
        selected_gradient,
        expected_gradient,
        atol=2e-6,
        rtol=0.0,
    )
    assert logits.grad[0, 0, 3].count_nonzero().item() == 0


def test_missing_negative_keeps_fixed_mixture_weight() -> None:
    loss_fn = InterfaceContactBalancedLoss()
    f_input = _folding_input(
        coordinates=[[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        asym_id=[1, 2],
    )
    logits = _logits_with_contact_probabilities(2, {(0, 1): 0.35})

    loss, metrics = loss_fn(_distogram_out(logits), f_input)

    log_prob = F.log_softmax(logits, dim=-1)
    target = _target_bin(f_input, (0, 1))
    ce = -log_prob[0, 0, 1, target]
    expected_fn = (1.0 - torch.tensor(0.35)).pow(2) * ce / (1.0 + loss_fn.eps)
    torch.testing.assert_close(
        loss,
        0.75 * expected_fn,
        atol=2e-6,
        rtol=0.0,
    )
    assert metrics["interface_contact_fp_loss"].item() == 0.0


def test_zero_positive_interface_is_ignored_but_graph_connected() -> None:
    loss_fn = InterfaceContactBalancedLoss()
    f_input = _folding_input(
        coordinates=[[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
        asym_id=[1, 2],
    )
    logits = _logits_with_contact_probabilities(2, {(0, 1): 0.60})

    loss, metrics = loss_fn(_distogram_out(logits), f_input)

    assert loss.item() == 0.0
    assert metrics["interface_contact_active_interfaces"].item() == 0
    assert metrics["interface_contact_selected_negative_pairs"].item() == 0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.count_nonzero().item() == 0


def test_sparse_asym_ids_are_equivalent() -> None:
    loss_fn = InterfaceContactBalancedLoss()
    coordinates = [
        [0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        [9.0, 0.0, 0.0],
    ]
    dense_input = _folding_input(coordinates, asym_id=[1, 2, 2])
    sparse_input = _folding_input(coordinates, asym_id=[10_001, 80_002, 80_002])
    dense_logits = _logits_with_contact_probabilities(
        3,
        {(0, 1): 0.40, (0, 2): 0.70},
    )
    sparse_logits = dense_logits.detach().clone().requires_grad_()

    dense_loss, dense_metrics = loss_fn(_distogram_out(dense_logits), dense_input)
    sparse_loss, sparse_metrics = loss_fn(
        _distogram_out(sparse_logits),
        sparse_input,
    )

    torch.testing.assert_close(dense_loss, sparse_loss)
    assert (
        dense_metrics["interface_contact_active_interfaces"]
        == sparse_metrics["interface_contact_active_interfaces"]
    )
    assert (
        dense_metrics["interface_contact_selected_negative_pairs"]
        == sparse_metrics["interface_contact_selected_negative_pairs"]
    )


def test_saturated_false_positive_is_finite() -> None:
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

    loss, metrics = loss_fn(_distogram_out(logits), f_input)

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["interface_contact_fp_loss"])
    assert metrics["interface_contact_fp_loss"] > 40.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
