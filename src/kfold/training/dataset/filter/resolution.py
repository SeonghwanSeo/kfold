from kfold.utils.registry import DATA_FILTER

from .base import BaseFilter


@DATA_FILTER.register()
class ResolutionFilter(BaseFilter):
    """Base class for samplers for model training."""

    class Config(BaseFilter.Config):
        """Configuration for BaseFilter."""

        min: float = 0.1
        max: float = 4.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.min = config.min
        self.max = config.max

    def filter_fn(self, m: dict) -> bool:
        """Get samples and their weights from metadata.

        Parameters
        ----------
        m : dict
            The metadata to setup the sampler.

        Returns
        -------
        bool
            Whether the metadata passes the filter.
        """
        assert "exp" in m, "metadata must contain 'exp' key for resolution filter"
        resolution = m["exp"].get("resolution")
        return resolution is not None and self.min <= resolution <= self.max
