import enum

# TODO (After stabilizing):


class ConstraintType(enum.IntEnum):
    """Types of constraints between atoms."""

    UNSPECIFIED = 0
    INTERACTION = 1
    COVALENT = 2
