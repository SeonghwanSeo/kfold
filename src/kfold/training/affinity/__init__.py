"""Data, cache, loss, and trainer utilities for affinity ranking.

Keep this package initializer free of Torch imports so CPU-only manifest and
FEP-filter preparation can run in a lightweight preprocessing environment.
"""

from .data import AffinityRecord, Endpoint

__all__ = ["AffinityRecord", "Endpoint"]
