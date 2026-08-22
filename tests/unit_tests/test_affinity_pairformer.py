import pytest
import torch

import kfold.constants as C
from kfold.model.modules.affinity_pairformer import AffinityPairformer
from kfold.training.affinity.module import (
    AffinityRankingConfig,
    AffinityRankingModule,
    validation_metrics,
)


def test_affinity_pairformer_has_only_head_gradients() -> None:
    model = AffinityPairformer(
        AffinityPairformer.Config(
            channel_s=16,
            channel_z=8,
            num_heads_attn=4,
            num_blocks=1,
            dropout=0.0,
        )
    )
    token_mask = torch.ones(2, 4, dtype=torch.bool)
    chain_type = torch.tensor(
        [
            C.ChainType.PROTEIN.value,
            C.ChainType.PROTEIN.value,
            C.ChainType.LIGAND.value,
            C.ChainType.LIGAND.value,
        ]
    )
    prediction = model(
        s_inputs=torch.randn(2, 4, 16),
        s_lm=torch.randn(2, 4, 16),
        z=torch.randn(2, 4, 4, 8),
        distogram_features=torch.randn(2, 4, 4, 3),
        token_mask=token_mask,
        protein_mask=(chain_type == C.ChainType.PROTEIN.value).expand(2, -1),
        ligand_mask=(chain_type == C.ChainType.LIGAND.value).expand(2, -1),
    )
    prediction.sum().backward()
    assert prediction.shape == (2,)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.readout[-1].weight.grad is not None


def test_cross_only_stack_honors_activation_checkpointing() -> None:
    """The custom cross-only route must not bypass the stack checkpoint policy."""
    torch.manual_seed(0)
    no_checkpoint = AffinityPairformer(
        AffinityPairformer.Config(
            channel_s=16,
            channel_z=8,
            num_heads_attn=4,
            num_blocks=2,
            dropout=0.0,
            blocks_per_ckpt=None,
        )
    )
    checkpointed = AffinityPairformer(
        AffinityPairformer.Config(
            channel_s=16,
            channel_z=8,
            num_heads_attn=4,
            num_blocks=2,
            dropout=0.0,
            blocks_per_ckpt=1,
        )
    )
    checkpointed.load_state_dict(no_checkpoint.state_dict())
    token_mask = torch.ones(2, 4, dtype=torch.bool)
    chain_type = torch.tensor(
        [
            C.ChainType.PROTEIN.value,
            C.ChainType.PROTEIN.value,
            C.ChainType.LIGAND.value,
            C.ChainType.LIGAND.value,
        ]
    )
    inputs = {
        "s_inputs": torch.randn(2, 4, 16),
        "s_lm": torch.randn(2, 4, 16),
        "z": torch.randn(2, 4, 4, 8),
        "distogram_features": torch.randn(2, 4, 4, 3),
        "token_mask": token_mask,
        "protein_mask": (chain_type == C.ChainType.PROTEIN.value).expand(2, -1),
        "ligand_mask": (chain_type == C.ChainType.LIGAND.value).expand(2, -1),
    }
    calls = 0

    def count_pair_block_calls(*_args: object) -> None:
        nonlocal calls
        calls += 1

    handle = checkpointed.stack.blocks[0].pair_block.register_forward_hook(
        count_pair_block_calls
    )
    try:
        expected = no_checkpoint(**inputs)
        actual = checkpointed(**inputs)
        expected.sum().backward()
        actual.sum().backward()
    finally:
        handle.remove()

    assert torch.allclose(actual, expected)
    assert calls >= 2  # Initial forward plus checkpointed backward recomputation.
    for expected_parameter, actual_parameter in zip(
        no_checkpoint.parameters(), checkpointed.parameters(), strict=True
    ):
        assert torch.allclose(expected_parameter.grad, actual_parameter.grad)


def test_lightning_hparams_exclude_frozen_task_config() -> None:
    module = AffinityRankingModule(
        model_config=AffinityPairformer.Config(
            channel_s=16,
            channel_z=8,
            num_heads_attn=4,
            num_blocks=1,
            dropout=0.0,
        ),
        task_config=AffinityRankingConfig(),
    )
    assert "model_config" not in module.hparams
    assert "task_config" not in module.hparams


def test_validation_metrics_are_per_assay() -> None:
    metrics = validation_metrics(
        [
            ("assay-a", "lig-a", 1.0, 1.0),
            ("assay-a", "lig-b", 2.0, 2.0),
            ("assay-a", "lig-c", 3.0, 3.0),
            ("assay-b", "lig-d", 3.0, 1.0),
            ("assay-b", "lig-e", 1.0, 3.0),
        ]
    )
    assert metrics["pearson_assays"] == 1
    assert metrics["mean_assay_pearson"] == pytest.approx(1.0)
    assert metrics["pairwise_comparisons"] == 4


def test_validation_metrics_collapse_same_ligand_replicates() -> None:
    metrics = validation_metrics(
        [
            ("assay", "lig-a", 1.0, 1.0),
            ("assay", "lig-a", 3.0, 3.0),
            ("assay", "lig-b", 2.0, 2.0),
            ("assay", "lig-c", 4.0, 4.0),
        ]
    )
    assert metrics["pearson_assays"] == 1
    assert metrics["ranking_replicate_records"] == 1
    assert metrics["pairwise_comparisons"] == 2
