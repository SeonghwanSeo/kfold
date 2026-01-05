# Started from https://github.com/jwohlwend/boltz
from typing import NamedTuple

import numpy as np

from kfold.data.types.metadata import Metadata
from kfold.utils.registry import DATA_SAMPLER, BaseConfig


class Sample(NamedTuple):
    """Chain/Interface sample for model training, used in AlphaFold3.

    Parameters
    ----------
    metadata : Metadata
        The metadata of the sampled item.
    asym_id : int | tuple[int, int] | None
        The cropping constraint; asym_id(s) of chain / interface to include.
        If None, no constraint is applied.
    """

    metadata: Metadata
    asym_id: int | tuple[int, int] | None


@DATA_SAMPLER.register()
class BaseSampler:
    """Base class for samplers for model training."""

    class Config(BaseConfig):
        """Configuration for BaseSampler."""

        pass

    def get_samples(self, metadatas: list[Metadata]) -> tuple[list[Sample], np.ndarray]:
        """Get samples and their weights from metadata.

        Parameters
        ----------
        metadatas : list[Metadata]
            The metadata to setup the sampler.

        Returns
        -------
        samples : list[Sample]
            The sampled items.
        weights : np.ndarray
            The weights for each sampled item.
        """
        samples: list[Sample] = [Sample(record, None) for record in metadatas]
        weights = np.ones(len(samples), dtype=np.float32)
        return samples, weights
