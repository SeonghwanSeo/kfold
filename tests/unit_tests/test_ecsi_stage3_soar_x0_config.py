from pathlib import Path

from kfold.config import load_config, to_dict

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_soar_x0_preset_inherits_latest_stage3_data_contract() -> None:
    base = load_config(REPO_ROOT / "configs/train-ecsi-stage3.yaml")
    candidate = load_config(REPO_ROOT / "configs/train-ecsi-stage3-soar-x0.yaml")
    assert to_dict(candidate.train.data) == to_dict(base.train.data)
    assert candidate.train.data.max_tokens == 768
    assert candidate.train.data.max_atoms == 9216
    assert candidate.train.data.max_sequence_tokens == 1536
    names = [dataset.name for dataset in candidate.train.data.train_datasets]
    weights = [float(dataset.weight) for dataset in candidate.train.data.train_datasets]
    assert names == [
        "rcsb-train-upd",
        "AF2-MGnify-long",
        "Rfam",
        "AFDB-homodimer",
        "AFDB-heterodimer",
        "JASPAR",
        "ENCORE",
        "TPD",
    ]
    assert weights == [0.75, 0.05, 0.05, 0.1, 0.02, 0.01, 0.01, 0.01]


def test_soar_x0_preset_changes_only_declared_training_surfaces() -> None:
    config = load_config(REPO_ROOT / "configs/train-ecsi-stage3-soar-x0.yaml")
    assert config.model.patch_pair_geometry.enabled is True
    assert config.model.diffusion_head.train_x_0_perturb_time_min == 0.4
    assert config.model.diffusion_head.train_x_0_perturb_time_max == 0.8
    assert config.model.diffusion_head.train_x_0_perturb_prob == 0.5
    assert config.model.diffusion_head.train_x_0_perturb_rotation_deg == 8.0
    assert config.model.diffusion_head.train_x_0_perturb_translation_distance == 1.2
    assert config.train.global_hparams.diffusion_batch_size == 16
    assert config.train.global_hparams.global_batch_size == 160
    assert config.train.load_opt_state is False
    assert config.train.load_global_step is True
    assert list(config.train.init_from_ema) == ["trunk"]
    assert config.train.loss.weights.patch_geometry == 0.0

    soar = config.train.training.soar
    assert soar.mode == "model_sampler"
    assert soar.root_time_policy == "mid_high_schedule_stratified"
    assert soar.auxiliary_transition == "exact_markov"
    assert soar.apply_rollout_churn is False
    assert soar.auxiliary_samples_per_root == 4
    assert soar.forward_retention_min == 0.5
    assert soar.lambda_aux == 1.0
    assert soar.mid_time_lower == 0.2
    assert soar.high_time_split == 0.8
    assert soar.mid_time_probability == 1.0
