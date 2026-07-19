"""Utilities for deterministic DNA apo-structure construction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

import kfold.constants as C
from kfold.data.types.structure import Chain

DNA_HELIX_RISE = 3.4
DNA_HELIX_TWIST_DEGREES = 36.0
DNA_HELIX_AXIS_OFFSET = (2.79167851, 7.66193258, 2.53938045)
DNA_HELIX_CONFORMER_ROTATION = (
    -1.94663419,
    0.62796543,
    1.61080456,
)
DNA_PHOSPHATE_BRIDGE_BOND_LENGTH = 1.60
DNA_PHOSPHATE_OXYGEN_BOND_LENGTH = 1.48
DNA_PHOSPHATE_TETRAHEDRAL_COS = -1.0 / 3.0
DNA_PHOSPHODIESTER_O5_ANGLE_DEGREES = 119.5
DNA_PHOSPHODIESTER_O3_ANGLE_DEGREES = 120.9
DNA_PHOSPHATE_BRIDGE_SEARCH_STEPS = 360

_DNA_RESIDUE_ALIASES = {
    "A": "DA",
    "G": "DG",
    "C": "DC",
    "T": "DT",
    "N": "DN",
}

_DNA_BASE_RESIDUES = ("DA", "DG", "DC", "DT")
_DNA_BACKBONE_ATOMS = (
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
)

_DNA_REF_ATOM_POSITIONS: dict[str, dict[str, tuple[float, float, float]]] = {
    "DA": {
        "P": (1.849, -1.175, -3.725),
        "OP1": (1.678, -2.622, -3.455),
        "OP2": (1.165, -0.434, -4.815),
        "O5'": (1.524, -0.372, -2.377),
        "C5'": (2.152, -0.744, -1.165),
        "C4'": (1.528, 0.000, -0.000),
        "O4'": (1.939, 1.390, -0.029),
        "C3'": (0.000, 0.000, 0.000),
        "O3'": (-0.450, -0.245, 1.295),
        "C2'": (-0.368, 1.399, -0.486),
        "C1'": (0.804, 2.244, -0.000),
        "N9": (1.108, 3.391, -0.848),
        "C8": (1.140, 3.419, -2.216),
        "N7": (1.476, 4.582, -2.716),
        "C5": (1.695, 5.375, -1.602),
        "C6": (2.084, 6.722, -1.457),
        "N6": (2.334, 7.532, -2.490),
        "N1": (2.209, 7.208, -0.202),
        "C2": (1.965, 6.395, 0.834),
        "N3": (1.596, 5.117, 0.825),
        "C4": (1.476, 4.659, -0.439),
    },
    "DG": {
        "P": (2.170, -0.975, -3.711),
        "OP1": (2.157, -2.450, -3.680),
        "OP2": (1.536, -0.252, -4.834),
        "O5'": (1.520, -0.458, -2.357),
        "C5'": (2.159, -0.773, -1.130),
        "C4'": (1.530, -0.000, 0.000),
        "O4'": (1.937, 1.384, -0.089),
        "C3'": (0.000, 0.000, 0.000),
        "O3'": (-0.467, -0.217, 1.329),
        "C2'": (-0.357, 1.396, -0.523),
        "C1'": (0.814, 2.217, -0.000),
        "N9": (1.081, 3.433, -0.767),
        "C8": (1.301, 3.538, -2.119),
        "N7": (1.504, 4.764, -2.515),
        "C5": (1.426, 5.514, -1.352),
        "C6": (1.569, 6.910, -1.153),
        "O6": (1.803, 7.793, -2.002),
        "N1": (1.412, 7.252, 0.191),
        "C2": (1.148, 6.354, 1.208),
        "N2": (1.025, 6.859, 2.444),
        "N3": (1.018, 5.053, 1.027),
        "C4": (1.168, 4.707, -0.270),
    },
    "DC": {
        "P": (2.413, -1.601, -3.467),
        "OP1": (2.530, -3.035, -3.134),
        "OP2": (1.746, -1.098, -4.688),
        "O5'": (1.696, -0.836, -2.255),
        "C5'": (2.161, -1.021, -0.921),
        "C4'": (1.529, -0.000, 0.000),
        "O4'": (1.913, 1.320, -0.429),
        "C3'": (0.000, 0.000, 0.000),
        "O3'": (-0.486, -0.529, 1.212),
        "C2'": (-0.389, 1.471, -0.152),
        "C1'": (0.940, 2.213, 0.000),
        "N1": (0.993, 3.427, -0.837),
        "C2": (0.971, 4.679, -0.225),
        "O2": (0.945, 4.738, 1.011),
        "N3": (0.981, 5.792, -1.007),
        "C4": (1.018, 5.671, -2.341),
        "N4": (1.028, 6.788, -3.078),
        "C5": (1.032, 4.397, -2.979),
        "C6": (1.013, 3.315, -2.195),
    },
    "DT": {
        "P": (2.039, -1.193, -3.707),
        "OP1": (1.817, -2.637, -3.474),
        "OP2": (1.362, -0.438, -4.787),
        "O5'": (1.631, -0.417, -2.356),
        "C5'": (2.118, -0.867, -1.095),
        "C4'": (1.525, -0.000, -0.000),
        "O4'": (1.942, 1.369, -0.222),
        "C3'": (0.000, 0.000, 0.000),
        "O3'": (-0.516, -0.584, 1.166),
        "C2'": (-0.419, 1.460, -0.117),
        "C1'": (0.867, 2.248, -0.000),
        "N1": (0.925, 3.295, -1.002),
        "C2": (0.712, 4.583, -0.615),
        "O2": (0.499, 4.901, 0.541),
        "N3": (0.779, 5.502, -1.630),
        "C4": (1.017, 5.242, -2.967),
        "O4": (1.057, 6.123, -3.807),
        "C5": (1.212, 3.838, -3.301),
        "C7": (1.483, 3.417, -4.715),
        "C6": (1.143, 2.955, -2.315),
    },
    "DN": {
        "P": (2.348, -1.552, -3.536),
        "OP1": (2.353, -2.979, -3.199),
        "OP2": (1.720, -0.988, -4.759),
        "O5'": (1.700, -0.726, -2.314),
        "C5'": (2.171, -0.960, -0.992),
        "C4'": (1.528, 0.000, 0.000),
        "O4'": (1.952, 1.367, -0.240),
        "C3'": (0.000, 0.000, 0.000),
        "O3'": (-0.445, -0.384, 1.281),
        "C2'": (-0.405, 1.436, -0.335),
        "C1'": (0.862, 2.237, -0.000),
    },
}


def build_dna_single_helix_for_chain(
    chain: Chain,
    *,
    rise: float = DNA_HELIX_RISE,
    twist_degrees: float = DNA_HELIX_TWIST_DEGREES,
    axis_offset: tuple[float, float, float] = DNA_HELIX_AXIS_OFFSET,
    conformer_rotation: tuple[float, float, float] = DNA_HELIX_CONFORMER_ROTATION,
    center: bool = True,
) -> np.ndarray:
    """Build heuristic right-handed single-helix coordinates for a DNA chain.

    The returned coordinates follow the chain's exact atom order and have shape
    ``[chain.num_atoms, 3]``.
    """
    if not chain.is_dna:
        raise ValueError("DNA single-helix construction requires a DNA chain.")

    atom_names_by_residue = [
        chain.atom.name[chain.residue.get_atom_slice(res_i + 1)].tolist()
        for res_i in range(chain.num_residues)
    ]
    return build_dna_single_helix(
        atom_names_by_residue,
        residue_names=chain.residue.name.tolist(),
        rise=rise,
        twist_degrees=twist_degrees,
        axis_offset=axis_offset,
        conformer_rotation=conformer_rotation,
        center=center,
    )


def build_dna_single_helix(
    atom_names_by_residue: Sequence[Sequence[str]],
    *,
    residue_names: Sequence[str] | None = None,
    rise: float = DNA_HELIX_RISE,
    twist_degrees: float = DNA_HELIX_TWIST_DEGREES,
    axis_offset: tuple[float, float, float] = DNA_HELIX_AXIS_OFFSET,
    conformer_rotation: tuple[float, float, float] = DNA_HELIX_CONFORMER_ROTATION,
    center: bool = True,
) -> np.ndarray:
    """Build an idealized right-handed single-stranded DNA helix.

    Parameters
    ----------
    atom_names_by_residue
        Atom names for each residue, in the output order.
    residue_names
        DNA residue names (``DA``, ``DG``, ``DC``, ``DT``, or ``DN``). If omitted,
        all residues use the unknown-base backbone template.
    rise
        Helical rise per residue in Angstrom.
    twist_degrees
        Right-handed twist per residue in degrees.
    axis_offset
        Local-frame translation applied before the helical transform. This sets
        the helix axis relative to the rigid nucleotide conformer.
    conformer_rotation
        Rotation vector in radians applied to the local nucleotide conformer before
        axis placement. The default orients bases toward the helix axis.
    center
        Whether to subtract the finite-coordinate centroid.
    """
    if residue_names is not None and len(residue_names) != len(atom_names_by_residue):
        raise ValueError(
            "residue_names must have the same length as atom_names_by_residue."
        )

    if residue_names is None:
        residue_names = ["DN"] * len(atom_names_by_residue)
    twist_radians = math.radians(twist_degrees)
    conformer_rotation_matrix = _rotation_matrix_from_rotvec(conformer_rotation)
    axis_offset_arr = np.array(axis_offset, dtype=np.float32)
    coords: list[np.ndarray] = []

    for res_i, (res_name, atom_names) in enumerate(
        zip(residue_names, atom_names_by_residue, strict=True)
    ):
        template = _get_dna_residue_template(res_name)
        for atom_name in atom_names:
            atom_name = str(atom_name).strip().upper()
            local_coord = template.get(atom_name)
            if local_coord is None:
                local_coord = _fallback_atom_coord(atom_name, template)
            local_coord_arr = conformer_rotation_matrix @ np.array(
                local_coord, dtype=np.float32
            )
            local_coord = tuple((local_coord_arr + axis_offset_arr).tolist())
            coords.append(
                _apply_helical_transform(
                    local_coord,
                    res_i,
                    rise=rise,
                    twist_radians=twist_radians,
                )
            )

    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.float32)

    arr = np.stack(coords, axis=0).astype(np.float32, copy=False)
    arr = _refine_phosphate_oxygens(arr, atom_names_by_residue)
    if center:
        arr = _center_finite_coords(arr)
    return arr


def _get_dna_residue_template(
    residue_name: str | C.ResidueName,
) -> dict[str, tuple[float, float, float]]:
    if isinstance(residue_name, C.ResidueName):
        residue_name = residue_name.name
    else:
        residue_name = str(residue_name).strip().upper()
    residue_name = _DNA_RESIDUE_ALIASES.get(residue_name, residue_name)
    if residue_name not in C.residue.DNA_RESIDUES_STR_SET:
        residue_name = "DN"
    return _DNA_REF_ATOM_POSITIONS[residue_name]


def _rotation_matrix_from_rotvec(rotvec: tuple[float, float, float]) -> np.ndarray:
    vec = np.array(rotvec, dtype=np.float32)
    theta = float(np.linalg.norm(vec))
    if theta == 0.0:
        return np.eye(3, dtype=np.float32)

    axis = vec / theta
    x, y, z = axis.tolist()
    c = math.cos(theta)
    s = math.sin(theta)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def _apply_helical_transform(
    local_coord: tuple[float, float, float],
    residue_index: int,
    *,
    rise: float,
    twist_radians: float,
) -> np.ndarray:
    x, y, z = local_coord
    theta = residue_index * twist_radians
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array(
        [
            x * cos_theta - y * sin_theta,
            x * sin_theta + y * cos_theta,
            z + residue_index * rise,
        ],
        dtype=np.float32,
    )


def _refine_phosphate_oxygens(
    coords: np.ndarray,
    atom_names_by_residue: Sequence[Sequence[str]],
) -> np.ndarray:
    """Refine internal phosphate groups from the actual polymer bridge geometry."""
    residue_atom_maps: list[dict[str, int]] = []
    atom_i = 0
    for atom_names in atom_names_by_residue:
        atom_map = {}
        for atom_name in atom_names:
            atom_map[str(atom_name).strip().upper()] = atom_i
            atom_i += 1
        residue_atom_maps.append(atom_map)

    refined = coords.copy()
    _refine_bridging_phosphate_positions(refined, residue_atom_maps)

    for res_i, atom_map in enumerate(residue_atom_maps):
        required = {"P", "OP1", "OP2", "O5'"}
        if not required <= set(atom_map):
            continue
        if res_i == 0 or "O3'" not in residue_atom_maps[res_i - 1]:
            continue

        p = refined[atom_map["P"]]
        o5 = refined[atom_map["O5'"]]
        prev_o3 = refined[residue_atom_maps[res_i - 1]["O3'"]]
        op_dirs = _get_nonbridging_phosphate_directions(p, o5, prev_o3)
        if op_dirs is None:
            continue

        op1_idx = atom_map["OP1"]
        op2_idx = atom_map["OP2"]
        old_op1 = refined[op1_idx]
        old_op2 = refined[op2_idx]
        cand_a = p + DNA_PHOSPHATE_OXYGEN_BOND_LENGTH * op_dirs[0]
        cand_b = p + DNA_PHOSPHATE_OXYGEN_BOND_LENGTH * op_dirs[1]
        keep_score = np.linalg.norm(cand_a - old_op1) + np.linalg.norm(cand_b - old_op2)
        swap_score = np.linalg.norm(cand_b - old_op1) + np.linalg.norm(cand_a - old_op2)
        if swap_score < keep_score:
            cand_a, cand_b = cand_b, cand_a
        refined[op1_idx] = cand_a.astype(np.float32)
        refined[op2_idx] = cand_b.astype(np.float32)

    return refined


def _refine_bridging_phosphate_positions(
    coords: np.ndarray,
    residue_atom_maps: Sequence[dict[str, int]],
) -> None:
    """Place internal P atoms between O3'(i - 1) and O5'(i)."""
    target = DNA_PHOSPHATE_BRIDGE_BOND_LENGTH
    for res_i, atom_map in enumerate(residue_atom_maps):
        if res_i == 0 or "P" not in atom_map or "O5'" not in atom_map:
            continue
        prev_map = residue_atom_maps[res_i - 1]
        if "O3'" not in prev_map:
            continue

        p_idx = atom_map["P"]
        old_p = coords[p_idx]
        prev_o3 = coords[prev_map["O3'"]]
        o5 = coords[atom_map["O5'"]]
        prev_c3 = coords[prev_map["C3'"]] if "C3'" in prev_map else None
        c5 = coords[atom_map["C5'"]] if "C5'" in atom_map else None
        new_p = _place_phosphate_between_bridge_oxygens(
            prev_o3,
            o5,
            old_p,
            target,
            prev_c3=prev_c3,
            c5=c5,
        )
        if new_p is not None:
            coords[p_idx] = new_p.astype(np.float32)


def _place_phosphate_between_bridge_oxygens(
    prev_o3: np.ndarray,
    o5: np.ndarray,
    old_p: np.ndarray,
    target_length: float,
    *,
    prev_c3: np.ndarray | None = None,
    c5: np.ndarray | None = None,
) -> np.ndarray | None:
    bridge = o5 - prev_o3
    bridge_len = float(np.linalg.norm(bridge))
    if bridge_len < 1e-6 or bridge_len > 2.0 * target_length:
        return None

    bridge_axis = bridge / bridge_len
    midpoint = (prev_o3 + o5) / 2.0
    half_bridge = bridge_len / 2.0
    height_sq = target_length**2 - half_bridge**2
    if height_sq < 0.0:
        return None
    height = math.sqrt(height_sq)

    radial = old_p - midpoint
    radial -= np.dot(radial, bridge_axis) * bridge_axis
    radial_dir = _normalize(radial)
    if radial_dir is None:
        radial_dir = _normalize(np.cross(bridge_axis, np.array([0.0, 0.0, 1.0])))
    if radial_dir is None:
        radial_dir = _normalize(np.cross(bridge_axis, np.array([1.0, 0.0, 0.0])))
    if radial_dir is None:
        return None

    if prev_c3 is not None and c5 is not None:
        angle_refined = _select_phosphate_bridge_position(
            midpoint=midpoint,
            height=height,
            bridge_axis=bridge_axis,
            radial_dir=radial_dir,
            old_p=old_p,
            prev_o3=prev_o3,
            o5=o5,
            prev_c3=prev_c3,
            c5=c5,
        )
        if angle_refined is not None:
            return angle_refined

    return midpoint + height * radial_dir


def _select_phosphate_bridge_position(
    *,
    midpoint: np.ndarray,
    height: float,
    bridge_axis: np.ndarray,
    radial_dir: np.ndarray,
    old_p: np.ndarray,
    prev_o3: np.ndarray,
    o5: np.ndarray,
    prev_c3: np.ndarray,
    c5: np.ndarray,
) -> np.ndarray | None:
    """Choose the P position that best matches experimental phosphodiester angles."""
    ortho_dir = _normalize(np.cross(bridge_axis, radial_dir))
    if ortho_dir is None:
        return None

    best_score = math.inf
    best_candidate: np.ndarray | None = None
    for step_i in range(DNA_PHOSPHATE_BRIDGE_SEARCH_STEPS):
        theta = 2.0 * math.pi * step_i / DNA_PHOSPHATE_BRIDGE_SEARCH_STEPS
        candidate = midpoint + height * (
            math.cos(theta) * radial_dir + math.sin(theta) * ortho_dir
        )
        o5_angle = _angle_degrees(c5, o5, candidate)
        o3_angle = _angle_degrees(prev_c3, prev_o3, candidate)
        if o5_angle is None or o3_angle is None:
            continue

        score = (
            (o5_angle - DNA_PHOSPHODIESTER_O5_ANGLE_DEGREES) ** 2
            + (o3_angle - DNA_PHOSPHODIESTER_O3_ANGLE_DEGREES) ** 2
            + 0.01 * float(np.sum((candidate - old_p) ** 2))
        )
        if score < best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def _get_nonbridging_phosphate_directions(
    p: np.ndarray,
    o5: np.ndarray,
    prev_o3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    bridge_a = _normalize(o5 - p)
    bridge_b = _normalize(prev_o3 - p)
    if bridge_a is None or bridge_b is None:
        return None

    bridge_sum = bridge_a + bridge_b
    bisector = _normalize(bridge_sum)
    normal = _normalize(np.cross(bridge_a, bridge_b))
    if bisector is None or normal is None:
        return None

    bridge_dot = float(np.dot(bisector, bridge_a))
    if abs(bridge_dot) < 1e-6:
        return None

    bisector_scale = DNA_PHOSPHATE_TETRAHEDRAL_COS / bridge_dot
    normal_scale_sq = 1.0 - bisector_scale**2
    if normal_scale_sq < 0.0:
        normal_scale_sq = 0.0
    normal_scale = math.sqrt(normal_scale_sq)

    op1 = _normalize(bisector_scale * bisector + normal_scale * normal)
    op2 = _normalize(bisector_scale * bisector - normal_scale * normal)
    if op1 is None or op2 is None:
        return None
    return op1, op2


def _normalize(v: np.ndarray, eps: float = 1e-6) -> np.ndarray | None:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return None
    return v / norm


def _angle_degrees(
    atom_a: np.ndarray,
    atom_b: np.ndarray,
    atom_c: np.ndarray,
) -> float | None:
    direction_a = _normalize(atom_a - atom_b)
    direction_c = _normalize(atom_c - atom_b)
    if direction_a is None or direction_c is None:
        return None
    cosine = float(np.clip(np.dot(direction_a, direction_c), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _fallback_atom_coord(
    atom_name: str,
    template: dict[str, tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Place unexpected DNA atoms near a chemically related anchor."""
    if atom_name.startswith(("OP", "P")):
        anchor_name = "P"
    elif atom_name.startswith(("O5", "C5")):
        anchor_name = "O5'"
    elif atom_name.startswith(("O3", "C3")):
        anchor_name = "O3'"
    else:
        anchor_name = "C1'"

    anchor = np.array(template[anchor_name], dtype=np.float32)
    seed = sum((i + 1) * ord(char) for i, char in enumerate(atom_name))
    angle = math.radians(seed % 360)
    radius = 0.8 + 0.15 * (seed % 5)
    dz = 0.2 * ((seed // 5) % 5 - 2)
    offset = np.array(
        [radius * math.cos(angle), radius * math.sin(angle), dz],
        dtype=np.float32,
    )
    return tuple((anchor + offset).tolist())


def _center_finite_coords(coords: np.ndarray) -> np.ndarray:
    mask = np.isfinite(coords).all(axis=-1)
    if mask.any():
        coords = coords.copy()
        coords[mask] -= coords[mask].mean(axis=0, keepdims=True)
    return coords


def _standardize_dna_backbone_templates(
    templates: dict[str, dict[str, tuple[float, float, float]]],
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Use one experimental sugar-phosphate backbone for every DNA base."""
    common_backbone = templates["DN"]
    common_origin, common_frame = _get_sugar_frame(common_backbone)
    standardized = {"DN": common_backbone.copy()}

    for residue_name in _DNA_BASE_RESIDUES:
        source_template = templates[residue_name]
        source_origin, source_frame = _get_sugar_frame(source_template)
        residue_template = {
            atom_name: common_backbone[atom_name]
            for atom_name in _DNA_BACKBONE_ATOMS
            if atom_name in common_backbone
        }

        for atom_name, coord in source_template.items():
            if atom_name in _DNA_BACKBONE_ATOMS:
                continue
            source_coord = np.array(coord, dtype=np.float64)
            local_coord = source_frame.T @ (source_coord - source_origin)
            target_coord = common_origin + common_frame @ local_coord
            residue_template[atom_name] = tuple(target_coord.tolist())

        standardized[residue_name] = residue_template

    return standardized


def _get_sugar_frame(
    template: dict[str, tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.array(template["C1'"], dtype=np.float64)
    c2_direction = _normalize(np.array(template["C2'"], dtype=np.float64) - origin)
    if c2_direction is None:
        raise ValueError("DNA reference conformer has degenerate C1'-C2' geometry.")

    o4_vector = np.array(template["O4'"], dtype=np.float64) - origin
    o4_direction = _normalize(o4_vector - np.dot(o4_vector, c2_direction) * c2_direction)
    if o4_direction is None:
        raise ValueError("DNA reference conformer has degenerate C1'-O4' geometry.")

    normal = np.cross(c2_direction, o4_direction)
    frame = np.stack([c2_direction, o4_direction, normal], axis=1)
    return origin, frame


_DNA_REF_ATOM_POSITIONS = _standardize_dna_backbone_templates(_DNA_REF_ATOM_POSITIONS)
