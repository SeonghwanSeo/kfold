from kfold.utils.registry import DATA_FILTER, BaseConfig


@DATA_FILTER.register()
class BaseFilter:
    """Base class for samplers for model training."""

    class Config(BaseConfig):
        """Configuration for BaseFilter."""

        pass

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, metadatas: list[dict]) -> list[dict]:
        """Filter metadata and return the filtered metadata.

        Parameters
        ----------
        metadatas : list[dict]
            The metadata to setup the filter.

        Returns
        -------
        filtered_metadatas : list[dict]
            The filtered metadata.
        """
        return self.filter(metadatas)

    def filter(self, metadatas: list[dict]) -> list[dict]:
        """Filter metadata and return the filtered metadata.

        Parameters
        ----------
        metadatas : list[dict]
            The metadata to setup the filter.

        Returns
        -------
        filtered_metadatas : list[dict]
            The filtered metadata.
        """
        return [m for m in metadatas if self.filter_fn(m)]

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
        return True
