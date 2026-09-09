"""Download shared assets required by inference."""

import json
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from kfold.data.types.ccd import CCD

logger = logging.getLogger(__name__)

ASSETS_REPO_ID = "SeonghwanSeo/kfold-assets"
MODEL_REPO_ID = "SeonghwanSeo/kfold"


def load_model(
    cache_dir: str | Path | None = None,
    *,
    weight: str | Path | None = None,
    config: str | Path | None = None,
    atlaslm=None,
):
    from kfold.model import KFold
    from kfold.utils.config import load_config

    missing = []
    if config is None:
        missing.append("config.yaml")
    if weight is None:
        missing.append("weights/kfold.pth")
    if missing:
        directory = Path(
            snapshot_download(
                repo_id=MODEL_REPO_ID,
                allow_patterns=missing,
                cache_dir=cache_dir,
            )
        )
        config = directory / "config.yaml" if config is None else config
        weight = directory / "weights/kfold.pth" if weight is None else weight
    overrides = None
    if cache_dir is not None:
        model_config = load_config(config)
        overrides = [
            f"{encoder}.cache_dir={json.dumps(str(cache_dir))}"
            for encoder in (
                "protein_sequence_encoder",
                "protein_structure_encoder",
                "rna_sequence_encoder",
            )
            if model_config.get(encoder) is not None
        ]
    return KFold.from_checkpoint(
        config,
        weight,
        override_args=overrides,
        atlaslm=atlaslm,
    )


def load_ccd(cache_dir: str | Path | None = None) -> CCD:
    path = hf_hub_download(
        repo_id=ASSETS_REPO_ID,
        filename="assets/ccd.pkl",
        cache_dir=cache_dir,
    )
    logger.info("Loading CCD data from: %s", path)
    return CCD.load(path)
