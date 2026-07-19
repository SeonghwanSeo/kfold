import math

import numpy as np

import kfold.constants as C
from kfold.data.types.structure import AtomLayout, BondLayout, Chain, ResidueLayout
from kfold.data.utils.dna_utils import (
    DNA_HELIX_RISE,
    DNA_PHOSPHATE_BRIDGE_BOND_LENGTH,
    DNA_PHOSPHODIESTER_O3_ANGLE_DEGREES,
    DNA_PHOSPHODIESTER_O5_ANGLE_DEGREES,
    build_dna_single_helix,
    build_dna_single_helix_for_chain,
)


def _atom_names(residue_name: str) -> list[str]:
    return [atom.value for atom in C.atom.RESIDUE_ATOMS[C.ResidueName[residue_name]]]


def _split_coords(
    coords: np.ndarray,
    residue_names: list[str],
) -> list[dict[str, np.ndarray]]:
    out = []
    atom_i = 0
    for residue_name in residue_names:
        atoms = _atom_names(residue_name)
        res_coords = coords[atom_i : atom_i + len(atoms)]
        out.append(dict(zip(atoms, res_coords, strict=True)))
        atom_i += len(atoms)
    return out


def _angle(atom_a: np.ndarray, atom_b: np.ndarray, atom_c: np.ndarray) -> float:
    direction_a = atom_a - atom_b
    direction_c = atom_c - atom_b
    direction_a /= np.linalg.norm(direction_a)
    direction_c /= np.linalg.norm(direction_c)
    cosine = float(np.clip(np.dot(direction_a, direction_c), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def test_build_dna_single_helix_has_finite_centered_coordinates():
    residue_names = ["DA", "DG", "DC", "DT", "DN"]
    coords = build_dna_single_helix(
        [_atom_names(name) for name in residue_names],
        residue_names=residue_names,
    )

    assert coords.dtype == np.float32
    assert coords.shape == (sum(len(_atom_names(name)) for name in residue_names), 3)
    assert np.isfinite(coords).all()
    np.testing.assert_allclose(coords.mean(axis=0), 0.0, atol=1e-5)


def test_build_dna_single_helix_is_right_handed_with_expected_rise():
    residue_names = ["DA", "DG", "DC", "DT"]
    coords = build_dna_single_helix(
        [_atom_names(name) for name in residue_names],
        residue_names=residue_names,
        center=False,
    )
    residues = _split_coords(coords, residue_names)
    c1_coords = np.stack([res["C1'"] for res in residues], axis=0)

    radii = np.linalg.norm(c1_coords[:, :2], axis=-1)
    np.testing.assert_allclose(radii, 6.15, atol=2e-3)
    np.testing.assert_allclose(np.diff(c1_coords[:, 2]), DNA_HELIX_RISE, atol=5e-3)

    twist = []
    for c1_i, c1_j in zip(c1_coords[:-1], c1_coords[1:], strict=True):
        cross_z = c1_i[0] * c1_j[1] - c1_i[1] * c1_j[0]
        dot = float(np.dot(c1_i[:2], c1_j[:2]))
        twist.append(math.degrees(math.atan2(cross_z, dot)))
    np.testing.assert_allclose(twist, 36.0, atol=5e-2)


def test_build_dna_single_helix_backbone_and_glycosidic_distances_are_sane():
    residue_names = ["DA", "DG", "DC", "DT"]
    coords = build_dna_single_helix(
        [_atom_names(name) for name in residue_names],
        residue_names=residue_names,
        center=False,
    )
    residues = _split_coords(coords, residue_names)

    same_residue_bonds = [
        ("C1'", "C2'"),
        ("C2'", "C3'"),
        ("C3'", "C4'"),
        ("C4'", "O4'"),
        ("O4'", "C1'"),
        ("C4'", "C5'"),
        ("C5'", "O5'"),
        ("O5'", "P"),
        ("C3'", "O3'"),
    ]
    for residue in residues:
        for atom_a, atom_b in same_residue_bonds:
            distance = np.linalg.norm(residue[atom_a] - residue[atom_b])
            assert 1.0 <= distance <= 1.8

    for residue_i, residue_j in zip(residues[:-1], residues[1:], strict=True):
        distance = np.linalg.norm(residue_i["O3'"] - residue_j["P"])
        assert 1.0 <= distance <= 1.9

    glycosidic_atoms = ["N9", "N9", "N1", "N1"]
    for residue, atom_name in zip(residues, glycosidic_atoms, strict=True):
        distance = np.linalg.norm(residue["C1'"] - residue[atom_name])
        assert 1.2 <= distance <= 1.7
        base_radius = np.linalg.norm(residue[atom_name][:2])
        c1_radius = np.linalg.norm(residue["C1'"][:2])
        p_radius = np.linalg.norm(residue["P"][:2])
        assert base_radius < c1_radius < p_radius


def test_build_dna_single_helix_refines_internal_phosphates():
    residue_names = ["DA", "DG", "DC", "DT"]
    coords = build_dna_single_helix(
        [_atom_names(name) for name in residue_names],
        residue_names=residue_names,
        center=False,
    )
    residues = _split_coords(coords, residue_names)

    target_angle = math.degrees(math.acos(-1.0 / 3.0))
    for prev_residue, residue in zip(residues[:-1], residues[1:], strict=True):
        p = residue["P"]
        prev_o3 = prev_residue["O3'"]
        o5 = residue["O5'"]
        phosphate_atoms = {
            "O5'": o5,
            "prev_O3'": prev_o3,
            "OP1": residue["OP1"],
            "OP2": residue["OP2"],
        }
        directions = {
            name: (coord - p) / np.linalg.norm(coord - p)
            for name, coord in phosphate_atoms.items()
        }

        np.testing.assert_allclose(np.linalg.norm(residue["OP1"] - p), 1.48, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(residue["OP2"] - p), 1.48, atol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(prev_o3 - p),
            DNA_PHOSPHATE_BRIDGE_BOND_LENGTH,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.linalg.norm(o5 - p),
            DNA_PHOSPHATE_BRIDGE_BOND_LENGTH,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            _angle(prev_residue["C3'"], prev_o3, p),
            DNA_PHOSPHODIESTER_O3_ANGLE_DEGREES,
            atol=0.2,
        )
        np.testing.assert_allclose(
            _angle(residue["C5'"], o5, p),
            DNA_PHOSPHODIESTER_O5_ANGLE_DEGREES,
            atol=0.2,
        )
        for op_name in ("OP1", "OP2"):
            for bridge_name in ("O5'", "prev_O3'"):
                angle = math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                np.dot(directions[op_name], directions[bridge_name]),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                np.testing.assert_allclose(angle, target_angle, atol=1e-4)


def test_build_dna_single_helix_for_chain_preserves_atom_order():
    residue_names = ["DA", "DT"]
    atom_names = [_atom_names(name) for name in residue_names]
    flat_atom_names = [atom for atoms in atom_names for atom in atoms]
    num_atoms = len(flat_atom_names)
    chain = Chain(
        chain_type=C.ChainType.DNA.value,
        entity_id=1,
        asym_id=1,
        sym_id=1,
        residue=ResidueLayout(
            name=np.array(residue_names, dtype=np.dtype("<U6")),
            num_atoms=np.array([len(atoms) for atoms in atom_names], dtype=np.uint8),
            is_standard=np.ones((len(residue_names),), dtype=bool),
        ),
        atom=AtomLayout(
            name=np.array(flat_atom_names, dtype=np.dtype("<U4")),
            element=np.zeros((num_atoms,), dtype=np.uint8),
            charge=np.zeros((num_atoms,), dtype=np.int8),
            coords=np.full((num_atoms, 3), np.nan, dtype=np.float32),
            bfactor=np.full((num_atoms,), np.nan, dtype=np.float16),
        ),
        bond=BondLayout(
            residue_index=np.empty((0, 2), dtype=np.uint32),
            atom_name=np.empty((0, 2), dtype=np.dtype("<U4")),
            bond_type=np.empty((0,), dtype=np.uint8),
        ),
    )

    chain_coords = build_dna_single_helix_for_chain(chain, center=False)
    raw_coords = build_dna_single_helix(
        atom_names,
        residue_names=residue_names,
        center=False,
    )

    np.testing.assert_allclose(chain_coords, raw_coords)
