class KFoldError(Exception):
    """Base class for exceptions in this module."""

    pass


# === During training === #
class BoltzDataProcessingError(KFoldError):
    """Exception raised for errors in the Boltz structure processing to
    TokenizedStructure."""


# === During inference === #
class PDBWriterMaxChainError(KFoldError):
    """Exception raised when the number of chains exceeds the maximum
    allowed for PDB format."""
