import numpy as np

from kfold.data.apo_perturbation import ApoPerturbation


def test_entity_translation_is_cached_per_entity_key():
    apo = ApoPerturbation(
        use_entity_random_translation=True,
        entity_random_translation_sigma=2.0,
        use_perturbation=False,
        use_random_rotation=False,
        seed=0,
    )
    rng = np.random.default_rng(0)
    cache: dict[tuple[int, int, int], np.ndarray] = {}

    key1 = (1, 10, 240)
    t1 = apo._get_or_sample_entity_translation(key1, rng, cache)
    t2 = apo._get_or_sample_entity_translation(key1, rng, cache)
    assert t1.shape == (3,)
    assert np.allclose(t1, t2)

    key2 = (2, 10, 240)
    t3 = apo._get_or_sample_entity_translation(key2, rng, cache)
    assert t3.shape == (3,)
    assert not np.allclose(t1, t3)


def test_masked_translation_only_moves_masked_atoms():
    apo = ApoPerturbation(
        use_entity_random_translation=True,
        entity_random_translation_sigma=1.0,
        use_perturbation=False,
        use_random_rotation=False,
        seed=0,
    )
    coords = np.zeros((2, 24, 3), dtype=np.float32)
    mask = np.zeros((2, 24), dtype=bool)
    mask[0, 0] = True
    mask[1, 5] = True

    t = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    out = apo._apply_masked_translation(coords, mask, t)

    # Masked atoms translated
    assert np.allclose(out[0, 0], t)
    assert np.allclose(out[1, 5], t)

    # Unmasked atoms remain zero
    assert np.allclose(out[0, 1], 0.0)
    assert np.allclose(out[1, 0], 0.0)


def test_translation_disabled_is_noop():
    apo = ApoPerturbation(
        use_entity_random_translation=False,
        entity_random_translation_sigma=10.0,
        use_perturbation=False,
        use_random_rotation=False,
        seed=0,
    )
    rng = np.random.default_rng(0)
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    key = (1, 10, 240)
    t = apo._get_or_sample_entity_translation(key, rng, cache)
    assert np.allclose(t, 0.0)
