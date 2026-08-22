import numpy as np

from kfold.training.affinity.pocket import (
    DistogramPocketEvidence,
    Pocket80kEvidence,
    PocketEvidence,
    build_pocket80k_target_consensus,
    protein_residue_min_ligand_distances,
    select_best_structure_sample,
    select_consensus_pocket_evidence,
    select_distogram_consensus_pocket,
)


def test_pocket_consensus_selects_highest_mean_top_residue_overlap() -> None:
    common = dict(
        protein_key="protein",
        iptm=0.8,
        structure_sha256="a" * 64,
        evidence_contract_sha256="b" * 64,
    )
    evidence = [
        PocketEvidence(
            request_id="r0",
            protein_residue_min_distance=np.asarray([0.0, 1.0, 9.0, 10.0]),
            **common,
        ),
        PocketEvidence(
            request_id="r1",
            protein_residue_min_distance=np.asarray([0.1, 1.1, 9.0, 10.0]),
            **common,
        ),
        PocketEvidence(
            request_id="r2",
            protein_residue_min_distance=np.asarray([10.0, 9.0, 0.0, 1.0]),
            **common,
        ),
    ]
    selected, overlap = select_consensus_pocket_evidence(
        evidence,
        closest_residues=2,
    )
    assert selected.request_id == "r0"
    assert overlap == 2 / 3


def test_pocket_evidence_uses_highest_iptm_and_residue_atom_minimum() -> None:
    sample_index, iptm = select_best_structure_sample(
        np.asarray([0.2, 0.8, 0.8], dtype=np.float32)
    )
    assert sample_index == 1
    assert np.isclose(iptm, 0.8)
    distances = protein_residue_min_ligand_distances(
        protein_atom_coords=np.asarray(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        ligand_atom_coords=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        residue_atom_starts=np.asarray([0, 2]),
        residue_atom_ends=np.asarray([2, 3]),
    )
    assert distances.tolist() == [1.0, 9.0]


def test_distogram_consensus_is_order_independent_with_stable_tie() -> None:
    evidence = [
        DistogramPocketEvidence(
            system_id=system_id,
            protein_key="protein",
            canonical_smiles=ligand,
            protein_token_min_expected_distance=np.asarray(distances),
        )
        for system_id, ligand, distances in (
            ("system-b", "ligand-b", [0.0, 1.0, 9.0, 10.0]),
            ("system-a", "ligand-a", [0.0, 1.0, 9.0, 10.0]),
        )
    ]
    selected, overlap = select_distogram_consensus_pocket(
        evidence[::-1], closest_tokens=2
    )
    assert selected.system_id == "system-a"
    assert overlap == 1.0


def test_pocket80k_consensus_freezes_the_nearest500_medoid_profile() -> None:
    evidence = []
    for index in range(10):
        evidence.append(
            Pocket80kEvidence(
                system_id=f"system-{index:02d}",
                protein_key="protein",
                canonical_smiles=f"ligand-{index:02d}",
                protein_min_expected_distance=np.asarray(
                    [5.0, 7.0 if index < 8 else 17.0, 6.0 if index < 2 else 18.0]
                ),
                protein_mean_normalized_entropy=np.asarray([0.3, 0.2, 0.1]),
                pocket_hlp_15a=0.3,
                pocket_residue_count=2,
            )
        )
    consensus = build_pocket80k_target_consensus(evidence[::-1], closest_residues=2)
    assert consensus.medoid_system_id == "system-02"
    assert consensus.protein_residue_min_distance.tolist() == [5.0, 7.0, 18.0]


def test_pocket80k_consensus_uses_every_valid_profile_beyond_ten() -> None:
    evidence = []
    for index in range(11):
        group_b = index >= 5
        evidence.append(
            Pocket80kEvidence(
                system_id=f"system-{index:02d}",
                protein_key="protein",
                canonical_smiles=f"ligand-{index:02d}",
                protein_min_expected_distance=np.asarray(
                    [9.0, 8.0, 0.0, 1.0] if group_b else [0.0, 1.0, 9.0, 8.0]
                ),
                protein_mean_normalized_entropy=np.full(4, 0.2),
                pocket_hlp_15a=0.2,
                pocket_residue_count=4,
            )
        )
    consensus = build_pocket80k_target_consensus(evidence, closest_residues=2)
    assert consensus.medoid_system_id == "system-05"
