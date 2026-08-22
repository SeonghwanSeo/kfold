from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_affinity_presets_keep_behavior_in_portable_config() -> None:
    for name in ("train-affinity-ranking.yaml", "train-affinity-pocket80k-target.yaml"):
        config = OmegaConf.load(REPO_ROOT / "configs" / name).train
        assert config.out_dir == "outputs/affinity"
        assert config.runtime.multiprocessing_sharing_strategy == "file_system"
        assert config.runtime.float32_matmul_precision == "high"
        assert config.lineage.snapshot_digest is None
        assert config.trainer.use_distributed_sampler is False
        assert config.checkpoint.monitor == "val/mean_assay_pearson"
        assert config.checkpoint.monitor_mode == "max"


def test_affinity_training_has_no_custom_behavior_environment_override() -> None:
    source = (REPO_ROOT / "scripts/affinity/train_affinity.py").read_text()
    assert "AFFINITY_SNAPSHOT_DIGEST" not in source
