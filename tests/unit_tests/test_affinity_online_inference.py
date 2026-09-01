from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import kfold.constants as C
from kfold.inference.affinity import (
    AUTO_AFFINITY_HEAD_CHECKPOINT,
    DEFAULT_AFFINITY_HEAD_FILENAME,
    PerQueryAffinityConfig,
    PerQueryAffinityPredictor,
    affinity_head_state_dict,
    build_per_query_affinity_inputs,
    resolve_affinity_head_checkpoint,
    resolve_affinity_ligand_asym_id,
    validate_affinity_system,
)
from kfold.inference.pl_client import InferenceConfig, KFoldInferenceClient
from kfold.model.modules.affinity_pairformer import AffinityPairformer


def _symmetric_logits(length: int, *, generator: torch.Generator) -> torch.Tensor:
    logits = torch.randn(length, length, 64, generator=generator)
    return (logits + logits.transpose(0, 1)) / 2


def _reference_crop(
    logits: torch.Tensor,
    *,
    protein_tokens: int,
    ligand_tokens: int,
    max_tokens: int,
    max_protein_tokens: int,
    neighborhood_size: int,
) -> torch.Tensor:
    values = logits[:protein_tokens, protein_tokens:].to(torch.bfloat16).float()
    centers = torch.linspace(
        2.0 + 20.0 / 128,
        22.0 - 20.0 / 128,
        64,
    )
    distance = (values.softmax(dim=-1) * centers).sum(dim=-1).amin(dim=-1)
    selected = set(range(protein_tokens, protein_tokens + ligand_tokens))
    selected_protein: set[int] = set()
    for position in torch.argsort(distance, stable=True).tolist():
        left = max(0, position - neighborhood_size // 2)
        left = min(left, protein_tokens - neighborhood_size)
        window = set(range(left, left + neighborhood_size))
        new_protein = window - selected_protein
        if not new_protein:
            continue
        if (
            len(selected) + len(new_protein) > max_tokens
            or len(selected_protein) + len(new_protein) > max_protein_tokens
        ):
            break
        selected.update(new_protein)
        selected_protein.update(new_protein)
    return torch.tensor(sorted(selected))


@pytest.mark.parametrize("seed", [7, 19])
def test_online_query_window_matches_submitted_reference(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    protein_tokens = 20
    ligand_tokens = 4
    length = protein_tokens + ligand_tokens
    token_mask = torch.ones(length, dtype=torch.bool)
    chain_type = torch.full((length,), C.ChainType.PROTEIN.value, dtype=torch.long)
    chain_type[protein_tokens:] = C.ChainType.LIGAND.value
    s_inputs = torch.randn(length, 384, generator=generator)
    s_lm = torch.randn(length, 384, generator=generator)
    z = torch.randn(length, length, 256, generator=generator)
    logits = _symmetric_logits(length, generator=generator)
    config = PerQueryAffinityConfig(
        max_tokens=16,
        max_protein_tokens=12,
        neighborhood_size=4,
    )

    online = build_per_query_affinity_inputs(
        s_inputs=s_inputs,
        s_lm=s_lm,
        z=z,
        distogram_logits=logits,
        token_mask=token_mask,
        chain_type=chain_type,
        config=config,
    )
    expected_indices = _reference_crop(
        logits,
        protein_tokens=protein_tokens,
        ligand_tokens=ligand_tokens,
        max_tokens=config.max_tokens,
        max_protein_tokens=config.max_protein_tokens,
        neighborhood_size=config.neighborhood_size,
    )

    assert torch.equal(online.crop_indices.cpu(), expected_indices)
    assert torch.equal(
        online.s_inputs[0],
        s_inputs.to(torch.bfloat16).float()[expected_indices],
    )
    expected_protein_tokens = len(expected_indices) - ligand_tokens
    assert online.s_inputs.shape[:2] == (1, len(expected_indices))
    assert online.protein_mask.sum().item() == expected_protein_tokens
    assert online.ligand_mask.sum().item() == 4
    pp = online.z[0][online.protein_mask[0]][:, online.protein_mask[0]]
    assert torch.count_nonzero(pp).item() == 0


@pytest.mark.parametrize(
    "chain_types, message",
    [
        ([C.ChainType.PROTEIN] * 4, "at least one protein token and one ligand"),
        ([C.ChainType.LIGAND] * 4, "at least one protein token and one ligand"),
        (
            [C.ChainType.PROTEIN, C.ChainType.DNA, C.ChainType.LIGAND],
            "protein--ligand systems only",
        ),
    ],
)
def test_affinity_rejects_non_pl_systems(
    chain_types: list[C.ChainType], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_affinity_system(
            token_mask=torch.ones(len(chain_types), dtype=torch.bool),
            chain_type=torch.tensor(chain_types),
        )


def test_affinity_checkpoint_auto_resolution_prefers_backbone_sibling(
    tmp_path,
) -> None:
    backbone_dir = tmp_path / "release"
    backbone_dir.mkdir()
    backbone = backbone_dir / "kfold-92k.pth"
    backbone.touch()
    sibling = backbone_dir / DEFAULT_AFFINITY_HEAD_FILENAME
    sibling.touch()
    repository_head = tmp_path / "weights" / DEFAULT_AFFINITY_HEAD_FILENAME
    repository_head.parent.mkdir()
    repository_head.touch()

    resolved = resolve_affinity_head_checkpoint(
        AUTO_AFFINITY_HEAD_CHECKPOINT,
        backbone_checkpoint=backbone,
        repository_root=tmp_path,
    )

    assert resolved == sibling.resolve()


def test_affinity_checkpoint_auto_resolution_uses_repository_weights(tmp_path) -> None:
    backbone = tmp_path / "release" / "kfold-92k.pth"
    backbone.parent.mkdir()
    backbone.touch()
    repository_head = tmp_path / "weights" / DEFAULT_AFFINITY_HEAD_FILENAME
    repository_head.parent.mkdir()
    repository_head.touch()

    resolved = resolve_affinity_head_checkpoint(
        AUTO_AFFINITY_HEAD_CHECKPOINT,
        backbone_checkpoint=backbone,
        repository_root=tmp_path,
    )

    assert resolved == repository_head.resolve()


def test_affinity_checkpoint_auto_resolution_reports_missing_paths(tmp_path) -> None:
    backbone = tmp_path / "release" / "kfold-92k.pth"
    backbone.parent.mkdir()
    backbone.touch()

    with pytest.raises(FileNotFoundError, match=DEFAULT_AFFINITY_HEAD_FILENAME):
        resolve_affinity_head_checkpoint(
            AUTO_AFFINITY_HEAD_CHECKPOINT,
            backbone_checkpoint=backbone,
            repository_root=tmp_path,
        )


def test_online_query_window_rejects_noncontiguous_protein_tokens() -> None:
    token_mask = torch.ones(14, dtype=torch.bool)
    chain_type = torch.full((14,), C.ChainType.PROTEIN.value, dtype=torch.long)
    chain_type[5:7] = C.ChainType.LIGAND.value
    generator = torch.Generator().manual_seed(3)

    with pytest.raises(ValueError, match="contiguous monomer"):
        build_per_query_affinity_inputs(
            s_inputs=torch.randn(14, 384, generator=generator),
            s_lm=torch.randn(14, 384, generator=generator),
            z=torch.randn(14, 14, 256, generator=generator),
            distogram_logits=_symmetric_logits(14, generator=generator),
            token_mask=token_mask,
            chain_type=chain_type,
            config=PerQueryAffinityConfig(neighborhood_size=4),
        )


def test_affinity_crop_uses_only_ligand_of_interest_distogram() -> None:
    protein_tokens = 20
    selected_ligand = torch.arange(20, 23)
    spectator_ligand = torch.arange(23, 26)
    length = 26
    token_mask = torch.ones(length, dtype=torch.bool)
    chain_type = torch.full((length,), C.ChainType.PROTEIN.value, dtype=torch.long)
    chain_type[protein_tokens:] = C.ChainType.LIGAND.value
    asym_id = torch.ones(length, dtype=torch.long)
    asym_id[selected_ligand] = 2
    asym_id[spectator_ligand] = 3
    logits = torch.full((length, length, 64), -20.0)
    logits[..., -1] = 20.0
    for protein_index in range(4):
        logits[protein_index, selected_ligand, 0] = 40.0
        logits[selected_ligand, protein_index, 0] = 40.0
    for protein_index in range(14, 18):
        logits[protein_index, spectator_ligand, 0] = 40.0
        logits[spectator_ligand, protein_index, 0] = 40.0
    generator = torch.Generator().manual_seed(23)

    inputs = build_per_query_affinity_inputs(
        s_inputs=torch.randn(length, 384, generator=generator),
        s_lm=torch.randn(length, 384, generator=generator),
        z=torch.randn(length, length, 256, generator=generator),
        distogram_logits=logits,
        token_mask=token_mask,
        chain_type=chain_type,
        asym_id=asym_id,
        ligand_asym_id=2,
        config=PerQueryAffinityConfig(
            max_tokens=7,
            max_protein_tokens=4,
            neighborhood_size=4,
        ),
    )

    assert set(inputs.crop_indices.tolist()) == {0, 1, 2, 3, 20, 21, 22}
    assert not set(inputs.crop_indices.tolist()).intersection(spectator_ligand.tolist())
    assert inputs.ligand_mask.sum().item() == 3


def test_resolve_affinity_ligand_requires_selection_for_multiple_ligands() -> None:
    ref_struct = SimpleNamespace(
        metadata=SimpleNamespace(
            chains=[
                SimpleNamespace(name="A", ctype=C.ChainType.PROTEIN, asym_id=1),
                SimpleNamespace(name="D", ctype=C.ChainType.LIGAND, asym_id=2),
                SimpleNamespace(name="E", ctype=C.ChainType.LIGAND, asym_id=3),
            ]
        )
    )

    with pytest.raises(ValueError, match="exactly one ligand chain"):
        resolve_affinity_ligand_asym_id(ref_struct, None)  # type: ignore[arg-type]
    assert (
        resolve_affinity_ligand_asym_id(  # type: ignore[arg-type]
            ref_struct, "D"
        )
        == 2
    )


def test_affinity_state_dict_matches_structure_style() -> None:
    tensor = torch.arange(3)
    lightning = {"state_dict": {"model._orig_mod.readout.weight": tensor}}
    direct = {"readout.weight": tensor}

    for checkpoint in (lightning, direct):
        extracted = affinity_head_state_dict(checkpoint)
        assert set(extracted) == {"readout.weight"}
        assert torch.equal(extracted["readout.weight"], tensor)


def test_inference_client_reuses_trunk_for_affinity() -> None:
    generator = torch.Generator().manual_seed(11)
    protein_tokens = 20
    ligand_tokens = 4
    length = protein_tokens + ligand_tokens
    token_mask = torch.ones(length, dtype=torch.bool)
    chain_type = torch.full((length,), C.ChainType.PROTEIN.value, dtype=torch.long)
    chain_type[protein_tokens:] = C.ChainType.LIGAND.value

    class FakeKFold(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.return_embeddings = False

        def inference(self, _f_input, **kwargs):
            self.return_embeddings = bool(kwargs["return_embeddings"])
            output = {
                "distogram": {"logits": _symmetric_logits(length, generator=generator)},
                "diffusion": {"coordinates": torch.zeros(1, 1, 3)},
                "confidence": {},
            }
            if self.return_embeddings:
                output["trunk"] = {
                    "s_inputs": torch.randn(length, 384, generator=generator),
                    "s_lm": torch.randn(length, 384, generator=generator),
                    "z": torch.randn(length, length, 256, generator=generator),
                }
            return output, {}

    model = FakeKFold()
    head = AffinityPairformer(AffinityPairformer.Config(num_blocks=1)).eval()
    predictor = PerQueryAffinityPredictor(
        head,
        checkpoint_sha256="0" * 64,
        config=PerQueryAffinityConfig(
            max_tokens=16,
            max_protein_tokens=12,
            neighborhood_size=4,
        ),
    ).eval()
    client = KFoldInferenceClient(
        model,  # type: ignore[arg-type]
        InferenceConfig(num_recycles=3, num_steps=1, num_samples=1),
        affinity_predictor=predictor,
    ).eval()
    f_input = SimpleNamespace(
        token=SimpleNamespace(
            pad_mask=token_mask,
            chain_type=chain_type,
            asym_id=torch.cat(
                (
                    torch.ones(protein_tokens, dtype=torch.long),
                    torch.full((ligand_tokens,), 2, dtype=torch.long),
                )
            ),
        )
    )

    output = client.forward(f_input, [])  # type: ignore[arg-type]

    assert model.return_embeddings
    assert "trunk" not in output
    assert output["affinity"]["p_activity"].shape == (1,)
    assert output["affinity"]["crop_token_count"].item() == 16
