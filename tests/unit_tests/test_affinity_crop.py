import pytest
import torch

import kfold.constants as C
from kfold.training.affinity.crop import (
    crop_distogram_features,
    distogram_feature_maps,
    head_crop_token_count,
    protein_ligand_distogram_profile,
    query_adaptive_final_crop_indices,
    select_ligand_preserving_crop,
    select_ligand_preserving_crop_from_scores,
    select_pocket_annotation_crop,
    select_query_adaptive_delta,
    shape_bucket_for_tokens,
)


def test_ligand_preserving_crop_uses_contact_distance_then_index() -> None:
    token_mask = torch.ones(5, dtype=torch.bool)
    chain_type = torch.tensor(
        [
            C.ChainType.LIGAND.value,
            C.ChainType.PROTEIN.value,
            C.ChainType.PROTEIN.value,
            C.ChainType.PROTEIN.value,
            C.ChainType.PROTEIN.value,
        ]
    )
    contact = torch.zeros(5, 5)
    expected = torch.full((5, 5), 20.0)
    # Residues 1 and 2 tie in contact; residue 2 wins by shorter distance.
    contact[1, 0] = contact[2, 0] = 0.9
    expected[1, 0] = 7.0
    expected[2, 0] = 6.0
    contact[3, 0] = 0.8
    selected = select_ligand_preserving_crop(
        token_mask=token_mask,
        chain_type=chain_type,
        contact_probability=contact,
        expected_distance=expected,
        max_tokens=3,
        max_protein_tokens=2,
    )
    assert selected.tolist() == [0, 1, 2]


def test_distogram_pocket_uses_distance_cutoff_and_entropy_confidence() -> None:
    selected = select_ligand_preserving_crop_from_scores(
        token_mask=torch.ones(4, dtype=torch.bool),
        chain_type=torch.tensor(
            [
                C.ChainType.LIGAND.value,
                C.ChainType.PROTEIN.value,
                C.ChainType.PROTEIN.value,
                C.ChainType.PROTEIN.value,
            ]
        ),
        contact_score=torch.tensor([0.9, 0.9, 1.0]),
        distance_score=torch.tensor([6.0, 6.0, 16.0]),
        entropy_score=torch.tensor([0.8, 0.2, 0.0]),
        pocket_distance_cutoff=15.0,
        max_tokens=2,
        max_protein_tokens=1,
    )
    assert selected.tolist() == [0, 2]


def test_crop_budget_overflow_and_feature_slicing() -> None:
    token_mask = torch.ones(3, dtype=torch.bool)
    chain_type = torch.tensor(
        [C.ChainType.LIGAND.value, C.ChainType.LIGAND.value, C.ChainType.PROTEIN.value]
    )
    with pytest.raises(ValueError, match="exhausts"):
        select_ligand_preserving_crop(
            token_mask=token_mask,
            chain_type=chain_type,
            contact_probability=torch.zeros(3, 3),
            expected_distance=torch.zeros(3, 3),
            max_tokens=2,
        )
    crop = crop_distogram_features(
        s_inputs=torch.randn(4, 3),
        s_lm=torch.randn(4, 3),
        z=torch.randn(4, 4, 2),
        token_mask=torch.ones(4, dtype=torch.bool),
        chain_type=torch.tensor(
            [
                C.ChainType.LIGAND.value,
                C.ChainType.PROTEIN.value,
                C.ChainType.PROTEIN.value,
                C.ChainType.PROTEIN.value,
            ]
        ),
        crop_indices=torch.tensor([0, 2, 3]),
        contact_probability=torch.randn(4, 4),
        expected_distance=torch.randn(4, 4),
        entropy=torch.rand(4, 4),
    )
    assert crop["z"].shape == (3, 3, 2)
    assert crop["distogram_features"].shape == (3, 3, 3)


def test_distogram_features_are_normalized_entropy() -> None:
    logits = torch.zeros(1, 2, 2, 4)
    contact, expected, entropy = distogram_feature_maps(
        logits, min_dist=2.0, max_dist=10.0, contact_cutoff=8.0
    )
    assert torch.allclose(entropy, torch.ones_like(entropy))
    assert torch.all(expected > 2.0)
    assert torch.all(contact > 0.0)


def test_head_crop_token_count_and_static_shape_bucket() -> None:
    assert head_crop_token_count(protein_tokens=156, ligand_tokens=12) == 168
    assert head_crop_token_count(protein_tokens=300, ligand_tokens=56) == 256
    assert shape_bucket_for_tokens(168) == 192
    with pytest.raises(ValueError, match="above largest"):
        shape_bucket_for_tokens(257)


def test_pocket_annotation_crop_keeps_ligand_and_expands_neighborhood() -> None:
    token_mask = torch.ones(13, dtype=torch.bool)
    chain_type = torch.tensor(
        [C.ChainType.LIGAND.value] + [C.ChainType.PROTEIN.value for _ in range(12)]
    )
    distances = torch.tensor(
        [9.0, 8.0, 7.0, 6.0, 5.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )
    selected = select_pocket_annotation_crop(
        token_mask=token_mask,
        chain_type=chain_type,
        protein_min_distance=distances,
        max_tokens=13,
        max_protein_tokens=12,
        neighborhood_size=10,
    )
    assert selected.tolist() == list(range(13))


def test_pocket_annotation_crop_uses_minimum_window_and_rejects_ligand_only() -> None:
    token_mask = torch.ones(12, dtype=torch.bool)
    chain_type = torch.tensor(
        [C.ChainType.PROTEIN.value] * 11 + [C.ChainType.LIGAND.value]
    )
    distances = torch.arange(11, dtype=torch.float32)
    selected = select_pocket_annotation_crop(
        token_mask=token_mask,
        chain_type=chain_type,
        protein_min_distance=distances,
        max_tokens=11,
        max_protein_tokens=10,
        neighborhood_size=10,
    )
    assert selected.tolist() == list(range(10)) + [11]

    many_ligands = torch.tensor(
        [C.ChainType.PROTEIN.value] * 11 + [C.ChainType.LIGAND.value] * 247
    )
    with pytest.raises(ValueError, match="cannot retain any protein"):
        select_pocket_annotation_crop(
            token_mask=torch.ones(258, dtype=torch.bool),
            chain_type=many_ligands,
            protein_min_distance=distances,
            max_tokens=256,
            max_protein_tokens=200,
            neighborhood_size=10,
        )


def test_consensus_crop_requires_one_contiguous_monomer_and_exact_window() -> None:
    chain_type = torch.tensor(
        [C.ChainType.PROTEIN.value] * 12 + [C.ChainType.LIGAND.value]
    )
    selected = select_pocket_annotation_crop(
        token_mask=torch.ones(13, dtype=torch.bool),
        chain_type=chain_type,
        protein_min_distance=torch.tensor(
            [9.0, 8.0, 7.0, 6.0, 5.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ),
        max_tokens=11,
        max_protein_tokens=10,
        neighborhood_size=10,
        require_contiguous_monomer=True,
    )
    assert selected.tolist() == list(range(10)) + [12]

    interrupted = torch.tensor(
        [C.ChainType.PROTEIN.value] * 6
        + [C.ChainType.LIGAND.value]
        + [C.ChainType.PROTEIN.value] * 6
    )
    with pytest.raises(ValueError, match="contiguous monomer"):
        select_pocket_annotation_crop(
            token_mask=torch.ones(13, dtype=torch.bool),
            chain_type=interrupted,
            protein_min_distance=torch.arange(12, dtype=torch.float32),
            neighborhood_size=10,
            require_contiguous_monomer=True,
        )


def test_80k_profile_uses_15a_pocket_entropy() -> None:
    logits = torch.full((4, 4, 4), -10.0)
    # Bin centers are 4, 8, 12, 16 A. Residue 0 is a confident 4 A pocket;
    # residue 1 is a confident 16 A non-pocket.
    logits[0, 2:, 0] = logits[2:, 0, 0] = 10.0
    logits[1, 2:, 3] = logits[2:, 1, 3] = 10.0
    profile = protein_ligand_distogram_profile(
        logits=logits,
        token_mask=torch.ones(4, dtype=torch.bool),
        chain_type=torch.tensor(
            [C.ChainType.PROTEIN.value] * 2 + [C.ChainType.LIGAND.value] * 2
        ),
        distance_cutoff=15.0,
    )
    assert profile.pocket_residue_count == 1
    assert profile.protein_min_expected_distance[0] < 15
    assert profile.protein_min_expected_distance[1] >= 15
    assert profile.pocket_hlp_15a is not None and profile.pocket_hlp_15a < 0.01


def test_query_adaptive_delta_reserves_only_tokens_outside_canonical_crop() -> None:
    protein_tokens = 220
    ligand_tokens = 2
    chain_type = torch.tensor(
        [C.ChainType.PROTEIN.value] * protein_tokens
        + [C.ChainType.LIGAND.value] * ligand_tokens
    )
    base = torch.tensor(list(range(200)) + [220, 221])
    selection = select_query_adaptive_delta(
        token_mask=torch.ones(protein_tokens + ligand_tokens, dtype=torch.bool),
        chain_type=chain_type,
        base_crop_indices=base,
        query_min_expected_distance=torch.tensor([20.0] * 200 + [5.0] * 20),
        query_mean_normalized_entropy=torch.linspace(0.1, 0.9, protein_tokens),
        max_tokens=256,
        max_protein_tokens=200,
        neighborhood_size=10,
        adaptive_fraction=0.2,
        distance_cutoff=15.0,
    )
    assert selection.extra_protein_source_indices.tolist() == list(range(200, 220))
    assert selection.accepted_query_windows.shape[1] == 10
    assert len(selection.accepted_query_windows) >= 2
    assert not set(selection.extra_protein_source_indices.tolist()).intersection(
        base.tolist()
    )


def test_query_adaptive_direct_replay_preserves_overlap_and_whole_tail_order() -> None:
    final = query_adaptive_final_crop_indices(
        base_crop_indices=torch.tensor(list(range(20)) + [25, 26]),
        base_chain_type=torch.tensor(
            [C.ChainType.PROTEIN.value] * 20 + [C.ChainType.LIGAND.value] * 2
        ),
        extra_protein_source_indices=torch.tensor([20, 21, 22]),
        target_consensus_window_order=torch.tensor(
            [list(range(10)), list(range(10, 20))]
        ),
        accepted_query_windows=torch.tensor([list(range(15, 25))]),
        max_tail_tokens=40,
    )
    assert set(range(15, 23)).issubset(final.tolist())
    assert not set(range(10, 15)).intersection(final.tolist())
    assert {25, 26}.issubset(final.tolist())
