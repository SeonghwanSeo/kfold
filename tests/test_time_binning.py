from types import SimpleNamespace

import pytest
import torch
from torchmetrics import MeanMetric

from kfold.training.training_module import (
    KFoldTrainingModule,
    _get_diffusion_time_for_binning,
    _get_structure_module_for_binning,
)
from kfold.training.utils.binned_loss_logging import (
    EntityBinConfig,
    EntityBinnedLossLogger,
    TimeBinConfig,
    TimeBinnedLossLogger,
    _get_time_bounds,
)
from kfold.training.utils.time_binning import binned_sum_and_count, compute_bin_index


def test_compute_bin_index_basic():
    u = torch.tensor([0.0, 0.05, 0.10, 0.15, 0.99, 1.0])
    nbins = 10
    idx = compute_bin_index(u, nbins)
    # 0.00,0.05 -> 0; 0.10,0.15 -> 1; 0.99,1.00 -> 9
    assert idx.tolist() == [0, 0, 1, 1, 9, 9]


def test_binned_sum_and_count():
    u = torch.tensor([0.05, 0.15, 0.95])
    nbins = 10
    idx = compute_bin_index(u, nbins)
    values = torch.tensor([1.0, 3.0, 5.0])
    bin_sum, bin_count = binned_sum_and_count(values, idx, nbins)
    assert bin_sum[0].item() == 1.0
    assert bin_count[0].item() == 1.0
    assert bin_sum[1].item() == 3.0
    assert bin_count[1].item() == 1.0
    assert bin_sum[9].item() == 5.0
    assert bin_count[9].item() == 1.0


def test_meanmetric_weighted_updates_matches_global_mean():
    # Simulate per-step bin means with different counts.
    m = MeanMetric()

    # Step 1: values [1,2,3] -> mean=2, count=3
    m.update(torch.tensor(2.0), torch.tensor(3.0))
    # Step 2: values [10] -> mean=10, count=1
    m.update(torch.tensor(10.0), torch.tensor(1.0))

    # Global mean = (1+2+3+10)/4 = 4
    assert torch.isclose(m.compute(), torch.tensor(4.0))


def test_get_time_bounds_supports_ecsi_time_attrs():
    structure_module = SimpleNamespace(time_min=1e-8, time_max=0.9999)

    assert _get_time_bounds(structure_module) == (1e-8, 0.9999)


def test_get_time_bounds_supports_sampling_attrs():
    structure_module = SimpleNamespace(
        sampling=SimpleNamespace(time_min=0.001, time_max=0.999)
    )

    assert _get_time_bounds(structure_module) == (0.001, 0.999)


def test_get_time_bounds_supports_scaled_edm_sigma_attrs():
    structure_module = SimpleNamespace(
        sigma_min=0.0004,
        sigma_max=160.0,
        sigma_data=16.0,
    )

    assert _get_time_bounds(structure_module) == (0.0064, 2560.0)


def test_get_diffusion_time_for_binning_prefers_t_hat_and_falls_back_to_t():
    t_hat = torch.tensor([[0.1]])
    t = torch.tensor([[0.2]])

    assert _get_diffusion_time_for_binning({"t_hat": t_hat, "t": t}) is t_hat
    assert _get_diffusion_time_for_binning({"t": t}) is t
    assert _get_diffusion_time_for_binning({}) is None


def test_get_structure_module_for_binning_prefers_kfold_diffusion_head():
    diffusion_head = SimpleNamespace(time_min=0.0, time_max=1.0)
    legacy_structure_module = SimpleNamespace(time_min=0.1, time_max=0.9)
    model = SimpleNamespace(
        diffusion_head=diffusion_head,
        structure_module=legacy_structure_module,
    )

    assert _get_structure_module_for_binning(model) is diffusion_head


def test_get_structure_module_for_binning_supports_legacy_structure_module():
    structure_module = SimpleNamespace(time_min=0.0, time_max=1.0)
    model = SimpleNamespace(structure_module=structure_module)

    assert _get_structure_module_for_binning(model) is structure_module


def test_get_structure_module_for_binning_raises_for_unknown_model_shape():
    with pytest.raises(AttributeError, match="diffusion_head or structure_module"):
        _get_structure_module_for_binning(SimpleNamespace())


def test_time_binned_loss_logger_updates_with_ecsi_style_t():
    logger = TimeBinnedLossLogger(TimeBinConfig(enabled=True, width=0.1))
    structure_module = SimpleNamespace(time_min=0.0, time_max=1.0)

    logger.update(
        t_hat=torch.tensor([[0.05, 0.15, 0.95]]),
        structure_module=structure_module,
        diffusion_per_sample={
            "mse_loss": torch.tensor([[1.0, 3.0, 5.0]]),
            "diffusion_loss": torch.tensor([[2.0, 4.0, 6.0]]),
            "smooth_lddt_loss": torch.tensor([[0.1, 0.3, 0.5]]),
        },
        distogram_loss_per_batch=torch.tensor([0.0]),
        loss_weights={"diffusion": 1.0, "distogram": 0.0},
    )

    out = logger.flush()
    assert torch.isclose(out["train_time_bin/mse_loss_interval1"], torch.tensor(1.0))
    assert torch.isclose(out["train_time_bin/mse_loss_interval2"], torch.tensor(3.0))
    assert torch.isclose(out["train_time_bin/mse_loss_interval10"], torch.tensor(5.0))
    assert torch.isclose(out["train_time_bin/loss_interval1"], torch.tensor(2.0))
    assert torch.isclose(out["train_time_bin/loss_interval2"], torch.tensor(4.0))
    assert torch.isclose(out["train_time_bin/loss_interval10"], torch.tensor(6.0))


def test_time_binned_loss_logger_broadcasts_scalar_distogram_loss():
    logger = TimeBinnedLossLogger(TimeBinConfig(enabled=True, width=0.1))
    structure_module = SimpleNamespace(time_min=0.0, time_max=1.0)

    logger.update(
        t_hat=torch.tensor([[0.05, 0.15]]),
        structure_module=structure_module,
        diffusion_per_sample={
            "mse_loss": torch.tensor([[1.0, 3.0]]),
            "diffusion_loss": torch.tensor([[2.0, 4.0]]),
        },
        distogram_loss_per_batch=torch.tensor(10.0),
        loss_weights={"diffusion": 1.0, "distogram": 0.5},
    )

    out = logger.flush()
    assert torch.isclose(out["train_time_bin/loss_interval1"], torch.tensor(7.0))
    assert torch.isclose(out["train_time_bin/loss_interval2"], torch.tensor(9.0))


def test_entity_binned_loss_logger_does_not_require_diffusion_time():
    logger = EntityBinnedLossLogger(EntityBinConfig(enabled=True, nbins=10))
    f_input = SimpleNamespace(
        token=SimpleNamespace(
            asym_id=torch.tensor([[1, 1, 2, 0]]),
            pad_mask=torch.tensor([[True, True, True, False]]),
        )
    )

    logger.update(
        f_input=f_input,
        diffusion_per_sample={
            "mse_loss": torch.tensor([[1.0, 3.0]]),
            "diffusion_loss": torch.tensor([[2.0, 4.0]]),
        },
        distogram_loss_per_batch=torch.tensor([0.0]),
        loss_weights={"diffusion": 1.0, "distogram": 0.0},
    )

    out = logger.flush()
    assert torch.isclose(out["train_entity_bin/mse_loss_interval2"], torch.tensor(2.0))
    assert torch.isclose(out["train_entity_bin/loss_interval2"], torch.tensor(3.0))


def test_entity_binned_loss_logger_broadcasts_scalar_distogram_loss():
    logger = EntityBinnedLossLogger(EntityBinConfig(enabled=True, nbins=10))
    f_input = SimpleNamespace(
        token=SimpleNamespace(
            asym_id=torch.tensor([[1, 1, 2]]),
            pad_mask=torch.tensor([[True, True, True]]),
        )
    )

    logger.update(
        f_input=f_input,
        diffusion_per_sample={
            "mse_loss": torch.tensor([[1.0, 3.0]]),
            "diffusion_loss": torch.tensor([[2.0, 4.0]]),
        },
        distogram_loss_per_batch=torch.tensor(10.0),
        loss_weights={"diffusion": 1.0, "distogram": 0.5},
    )

    out = logger.flush()
    assert torch.isclose(out["train_entity_bin/loss_interval2"], torch.tensor(8.0))


def test_compute_distogram_loss_caches_per_batch_loss_for_binned_logging():
    class FakeDistogramLoss:
        def __call__(self, distogram_out, f_input):
            return torch.tensor(
                [1.0, 3.0],
                device=distogram_out["logits"].device,
            )

    module = KFoldTrainingModule.__new__(KFoldTrainingModule)
    module.distogram_loss = FakeDistogramLoss()
    module._binned_cache_enabled = True
    module.train_diffusion_head = True
    module._timebin_last_distogram_loss_per_batch = None

    loss, metrics = KFoldTrainingModule.compute_distogram_loss(
        module,
        distogram_out={
            "logits": torch.empty(2, 1, 1, 1),
            "bin_boundaries": torch.empty(0),
        },
        f_input=SimpleNamespace(),
    )

    assert torch.isclose(loss, torch.tensor(2.0))
    assert torch.isclose(metrics["distogram_loss"], torch.tensor(2.0))
    assert torch.equal(
        module._timebin_last_distogram_loss_per_batch,
        torch.tensor([1.0, 3.0]),
    )
