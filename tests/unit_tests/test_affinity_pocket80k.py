import numpy as np

import kfold.constants as C
from kfold.training.affinity.pair_storage import (
    PAIR_STORAGE_CROSS_ONLY_BF16_TRI_LOGITS,
    pack_compact_pocket_storage,
    pack_cropped_pair_storage,
    unpack_cross_only_payload,
)
from kfold.training.affinity.schema import AFFINITY_CACHE_SCHEMA_POCKET_CROPPED_V1


def test_compact_pocket_payload_is_explicit_and_roundtrips() -> None:
    rng = np.random.default_rng(4)
    length = 5
    logits = rng.normal(size=(length, length, 8)).astype(np.float32)
    logits += logits.transpose(1, 0, 2)
    cropped = {
        "s_inputs": rng.normal(size=(length, 4)).astype(np.float32),
        "s_lm": rng.normal(size=(length, 3)).astype(np.float32),
        "z": rng.normal(size=(length, length, 6)).astype(np.float32),
        "token_mask": np.ones(length, dtype=bool),
        "chain_type": np.asarray(
            [C.ChainType.PROTEIN.value] * 3 + [C.ChainType.LIGAND.value] * 2
        ),
        "crop_indices": np.asarray([4, 5, 6, 10, 11], dtype=np.int32),
        "distogram_logits": logits,
    }
    sparse = pack_cropped_pair_storage(
        cropped, mode=PAIR_STORAGE_CROSS_ONLY_BF16_TRI_LOGITS
    )
    compact = pack_compact_pocket_storage(sparse.arrays)
    assert compact.arrays["cache_schema"].item() == (
        AFFINITY_CACHE_SCHEMA_POCKET_CROPPED_V1
    )
    restored = unpack_cross_only_payload(compact.arrays)
    assert restored["crop_indices"].tolist() == [4, 5, 6, 10, 11]
    assert restored["z"].shape == (5, 5, 6)
