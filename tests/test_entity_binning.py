import torch

from kfold.training.utils.entity_binning import entity_bin_index_from_asym_id


def test_entity_bin_index_from_asym_id_basic():
    # B=3, Lt=6
    asym_id = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],  # unique=1 -> interval1 -> idx 0
            [1, 2, 2, 2, 2, 2],  # unique=2 -> interval2 -> idx 1
            [1, 2, 3, 4, 5, 6],  # unique=6 -> interval6 -> idx 5
        ]
    )
    pad_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    idx = entity_bin_index_from_asym_id(asym_id=asym_id, pad_mask=pad_mask)
    assert idx.tolist() == [0, 1, 5]


def test_entity_bin_ge10():
    asym_id = torch.arange(1, 13)[None, :]  # unique=12 -> idx 9
    pad_mask = torch.ones_like(asym_id, dtype=torch.bool)
    idx = entity_bin_index_from_asym_id(asym_id=asym_id, pad_mask=pad_mask)
    assert idx.tolist() == [9]
