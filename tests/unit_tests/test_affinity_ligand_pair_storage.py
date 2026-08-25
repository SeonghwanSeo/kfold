from dataclasses import replace

import numpy as np
import pytest
import torch

import kfold.constants as C
from kfold.data.types.ccd import Component
from kfold.model.modules.affinity_pairformer import AffinityPairformer
from kfold.training.affinity.cache import FeatureCacheWriter
from kfold.training.affinity.ccd_ligand import (
    canonical_component_smiles,
    ccd_ligand_apo,
)
from kfold.training.affinity.ligand import (
    LigandApoLookup,
    generate_ligand_etkdg,
    store_ligand_apo,
)
from kfold.training.affinity.pair_storage import (
    AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1,
    PAIR_STORAGE_CROSS_ONLY_BF16_TRI_DISTOGRAM,
    PAIR_STORAGE_CROSS_ONLY_SPARSE,
    PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS,
    affinity_cross_pair_mask,
    crop_full_cross_payload,
    full_cross_pl_distogram_profile,
    is_full_cross_payload,
    pack_pair_storage,
    unpack_cross_only_payload,
)


def _source_arrays() -> dict[str, np.ndarray]:
    length = 5
    channels_s = 4
    channels_z = 3
    z = np.arange(length * length * channels_z, dtype=np.float32).reshape(
        length, length, channels_z
    )
    return {
        "s_inputs": np.arange(length * channels_s, dtype=np.float32).reshape(
            length, channels_s
        ),
        "s_lm": np.arange(length * channels_s, dtype=np.float32).reshape(
            length, channels_s
        ),
        "z": z,
        "token_mask": np.ones(length, dtype=bool),
        "chain_type": np.asarray(
            [
                C.ChainType.PROTEIN.value,
                C.ChainType.PROTEIN.value,
                C.ChainType.PROTEIN.value,
                C.ChainType.LIGAND.value,
                C.ChainType.LIGAND.value,
            ]
        ),
        "contact_probability": np.zeros((length, length), dtype=np.float32),
        "expected_distance": np.ones((length, length), dtype=np.float32),
        "distogram_entropy": np.full((length, length), 0.5, dtype=np.float32),
        "crop_indices": np.arange(length, dtype=np.int32),
    }


def test_cross_only_storage_omits_pp_and_round_trips_active_pairs() -> None:
    source = _source_arrays()
    packed = pack_pair_storage(source, mode=PAIR_STORAGE_CROSS_ONLY_SPARSE)
    assert packed.stored_pair_count == 2 * 3 * 2 + 2 * 2
    assert "z" not in packed.arrays
    restored = unpack_cross_only_payload(packed.arrays)
    mask = affinity_cross_pair_mask(source["token_mask"], source["chain_type"])
    assert torch.equal(restored["z"][mask], torch.from_numpy(source["z"])[mask])
    assert torch.count_nonzero(restored["z"][~mask]) == 0
    assert torch.count_nonzero(restored["distogram_features"][~mask]) == 0


def test_bf16_cross_only_storage_triangularly_packs_symmetric_distogram() -> None:
    source = _source_arrays()
    packed = pack_pair_storage(
        source,
        mode=PAIR_STORAGE_CROSS_ONLY_BF16_TRI_DISTOGRAM,
    )
    assert packed.stored_pair_count == 2 * 3 * 2 + 2 * 2
    assert packed.stored_distogram_pair_count == 3 * 2 + 2 * (2 + 1) // 2
    assert "z_pair_values" not in packed.arrays
    assert "z_pair_values_bf16" in packed.arrays
    restored = unpack_cross_only_payload(packed.arrays)
    mask = affinity_cross_pair_mask(source["token_mask"], source["chain_type"])
    source_z = torch.from_numpy(source["z"])
    assert torch.allclose(restored["z"][mask], source_z[mask], atol=0.25, rtol=0.01)
    assert torch.count_nonzero(restored["z"][~mask]) == 0
    assert torch.allclose(
        restored["distogram_features"],
        restored["distogram_features"].transpose(0, 1),
    )


def test_bf16_triangular_distogram_rejects_asymmetric_features() -> None:
    source = _source_arrays()
    source["expected_distance"][0, 1] = 2.0
    with pytest.raises(ValueError, match="not symmetric"):
        pack_pair_storage(
            source,
            mode=PAIR_STORAGE_CROSS_ONLY_BF16_TRI_DISTOGRAM,
        )


def test_ligand_etkdg_v3_is_deterministic_and_cacheable(tmp_path) -> None:
    first = generate_ligand_etkdg("CCO")
    second = generate_ligand_etkdg("CCO")
    assert first.source == "etkdg_v3"
    assert np.array_equal(first.coords, second.coords)
    with FeatureCacheWriter(tmp_path, map_size_bytes=2**20) as writer:
        row = store_ligand_apo(writer, apo=first)
    lookup = LigandApoLookup([row], cache_root=str(tmp_path))
    try:
        restored = lookup.get("CCO")
    finally:
        lookup.close()
    assert restored.structure_sha256 == first.structure_sha256
    assert np.array_equal(restored.coords, first.coords)


def test_existing_ccd_conformer_is_reordered_into_source_smiles_order() -> None:
    ccd_component = Component.from_smiles("EOH", "OCC")
    ccd_coords = np.arange(9, dtype=np.float32).reshape(3, 3)
    ccd_component = replace(ccd_component, ideal_coords=ccd_coords.astype(np.float16))
    canonical_smiles = "CCO"
    assert canonical_component_smiles(ccd_component) == canonical_smiles
    apo = ccd_ligand_apo(
        canonical_smiles=canonical_smiles,
        ccd_code="EOH",
        component=ccd_component,
    )
    assert apo is not None
    source_component = Component.from_smiles("LIG", canonical_smiles)
    atom_map = source_component.mol.GetSubstructMatch(
        ccd_component.mol,
        useChirality=True,
    )
    expected = np.empty_like(ccd_coords)
    expected[np.asarray(atom_map)] = ccd_coords
    assert apo.source == "ccd_ideal"
    assert apo.source_id == "EOH"
    assert np.array_equal(apo.coords, expected)


def test_cross_only_affinity_head_ignores_pp_z_and_distogram_cells() -> None:
    torch.manual_seed(7)
    model = AffinityPairformer(
        AffinityPairformer.Config(
            channel_s=8,
            channel_z=8,
            num_distogram_features=3,
            num_heads_attn=2,
            num_blocks=1,
            dropout=0.0,
            cross_pair_only=True,
        )
    ).eval()
    token_mask = torch.ones((1, 4), dtype=torch.bool)
    protein_mask = torch.tensor([[True, True, False, False]])
    ligand_mask = ~protein_mask
    inputs = {
        "s_inputs": torch.randn((1, 4, 8)),
        "s_lm": torch.randn((1, 4, 8)),
        "z": torch.randn((1, 4, 4, 8)),
        "distogram_features": torch.randn((1, 4, 4, 3)),
        "token_mask": token_mask,
        "protein_mask": protein_mask,
        "ligand_mask": ligand_mask,
    }
    pp = protein_mask[:, :, None] & protein_mask[:, None, :]
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["z"][pp] += 1_000.0
    changed["distogram_features"][pp] -= 1_000.0
    with torch.inference_mode():
        reference = model(**inputs)
        result = model(**changed)
    assert torch.allclose(reference, result, atol=1e-6, rtol=1e-6)


def test_full_cross_storage_keeps_all_singles_but_never_persists_pp() -> None:
    source = _source_arrays()
    logits = np.zeros((5, 5, 8), dtype=np.float32)
    source["distogram_logits"] = logits + np.swapaxes(logits, 0, 1)
    packed = pack_pair_storage(source, mode=PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS)
    assert is_full_cross_payload(packed.arrays)
    assert packed.arrays["cache_schema"].item() == AFFINITY_CACHE_SCHEMA_FULL_CROSS_V1
    assert "crop_indices" not in packed.arrays
    assert packed.arrays["s_inputs_bf16"].shape[0] == 5
    mask = affinity_cross_pair_mask(source["token_mask"], source["chain_type"])
    pair_indices = packed.arrays["pair_indices"]
    assert np.array_equal(pair_indices, np.stack(np.nonzero(mask), axis=-1))
    restored = unpack_cross_only_payload(packed.arrays)
    assert torch.count_nonzero(restored["z"][~torch.from_numpy(mask)]) == 0


def test_full_cross_payload_crops_at_training_time_from_pl_distogram() -> None:
    source = _source_arrays()
    logits = np.full((5, 5, 8), -8.0, dtype=np.float32)
    # Protein token 0 has strongest PL contact, token 1 the next strongest.
    for protein_index, logit in ((0, 8.0), (1, 6.0), (2, 2.0)):
        for ligand_index in (3, 4):
            logits[protein_index, ligand_index, 0] = logit
            logits[ligand_index, protein_index, 0] = logit
    source["distogram_logits"] = logits
    packed = pack_pair_storage(source, mode=PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS)
    crop = crop_full_cross_payload(packed.arrays, max_tokens=4, max_protein_tokens=2)
    assert crop["crop_indices"].tolist() == [0, 1, 3, 4]
    assert crop["s_inputs"].shape[0] == 4
    assert crop["z"].shape[:2] == (4, 4)


def test_v1_pl_profile_computes_exact_15a_pocket_entropy() -> None:
    source = _source_arrays()
    logits = np.zeros((5, 5, 8), dtype=np.float32)
    # Protein 0 is confidently near, protein 1 is uncertain and near, and
    # protein 2 is confidently farther than 15 A. Exact HLP excludes protein 2.
    for ligand in (3, 4):
        logits[0, ligand] = logits[ligand, 0] = -8.0
        logits[0, ligand, 0] = logits[ligand, 0, 0] = 8.0
        logits[1, ligand] = logits[ligand, 1] = 0.0
        logits[2, ligand] = logits[ligand, 2] = -8.0
        logits[2, ligand, -1] = logits[ligand, 2, -1] = 8.0
    source["distogram_logits"] = logits
    packed = pack_pair_storage(source, mode=PAIR_STORAGE_FULL_CROSS_BF16_TRI_LOGITS)
    profile = full_cross_pl_distogram_profile(packed.arrays, chunk_pairs=2)
    hlp, protein_tokens = profile.exact_pocket_hlp(distance_cutoff=15.0)
    assert protein_tokens == 2
    assert hlp is not None and 0.49 < hlp < 0.51
    assert profile.protein_min_expected_distance[-1] > 15.0
