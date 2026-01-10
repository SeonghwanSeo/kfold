import torch
from torchmetrics import MeanMetric

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
