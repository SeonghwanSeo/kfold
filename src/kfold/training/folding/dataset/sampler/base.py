# Started from https://github.com/jwohlwend/boltz
from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
from numpy.random import RandomState

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_SAMPLER


class Sample(NamedTuple):
    """A sampled item from the dataset.

    Parameters
    ----------
    record : Metadata
        The metadata record of the sampled item.
    asym_ids : tuple[int, ...] or None
        The cropping constraint; asym_ids of chains / interfaces to include.
        If None, no constraint is applied.
    """

    record: Metadata
    asym_ids: tuple[int, ...] | None = None


@DATA_SAMPLER.register()
class BaseSampler:
    """Base class for samplers."""

    items: list[Sample]
    weights: np.ndarray

    def setup(self, records: list[Metadata]) -> None:
        """Setup the sampler with the given records.

        Parameters
        ----------
        records : list[Metadata]
            The metadata records to setup the sampler.

        """
        self.items = [Sample(record, None) for record in records]
        self.weights = np.ones(len(self.items)) / len(self.items)

    def sample(self, rng: RandomState) -> Iterator[Sample]:
        """Sample a structure from the dataset infinitely.

        Parameters
        ----------
        rng : RandomState
            The random state for reproducibility.

        Yields
        ------
        tuple[Metadata, tuple[int, ...]]
            The sampled record and the indices of chains / interfaces.

        """
        num_items = len(self.items)
        while True:
            item_idx = rng.choice(num_items, p=self.weights)
            yield self.items[item_idx]
