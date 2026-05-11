import dataclasses

__all__ = ["Constraint"]


@dataclasses.dataclass(frozen=True)
class Constraint:
    """Represents a distance constraint between two residues(atoms).

    Attributes
    ----------
    asym_id: np.ndarray
        Chain asym id pairs in the constraint.
    residue_index: tuple[int, int]
        Residue index pairs in the constraint.
    atom_name: tuple[str, str]
        Atom names pairs in the constraint.
        For polymer, the atom name is the center atom,
        i.e., CA for protein, C1' for nucleic acids.
    lower_bound: float
        Minimum distance constraints.
        -1 indicates no minimum distance constraint.
    upper_bound: float
        Maximum distance constraints.
        -1 indicates no maximum distance constraint.
    """

    asym_id: tuple[int, int]
    residue_index: tuple[int, int]
    atom_name: tuple[str, str]
    lower_bound: float = -1
    upper_bound: float = -1
