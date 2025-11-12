# Started from https://github.com/jwohlwend/boltz
from typing import NamedTuple

import numpy as np

from kfold.data.metadata import Metadata
from kfold.utils.registry import DATA_SAMPLER, BaseConfig


class Sample(NamedTuple):
    """Chain/Interface sample for model training, used in AlphaFold3.

    Parameters
    ----------
    metadata : Metadata
        The metadata record of the sampled item.
    asym_ids : tuple[int, ...] or None
        The cropping constraint; asym_ids of chains / interfaces to include.
        If None, no constraint is applied.
    """

    metadata: Metadata
    asym_ids: tuple[int, ...] | None = None


@DATA_SAMPLER.register()
class BaseSampler:
    """Base class for samplers for model training."""

    class Config(BaseConfig):
        """Configuration for BaseSampler."""

        pass

    def get_samples(
        self, records: list[Metadata]
    ) -> tuple[list[Sample], np.ndarray | None]:
        """Get samples and their weights from metadata records.

        Parameters
        ----------
        records : list[Metadata]
            The metadata records to setup the sampler.

        Returns
        -------
        samples : list[Sample]
            The sampled items.
        weights : np.ndarray | None
            The weights for each sampled item.
            If None, uniform weights are assumed.
        """
        samples: list[Sample] = [Sample(record, None) for record in records]
        return samples, None
