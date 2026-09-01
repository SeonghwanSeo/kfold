from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import kfold.constants as C
from kfold.inference.affinity import (
    PerQueryAffinityConfig,
    PerQueryAffinityPredictor,
    build_per_query_affinity_inputs,
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
        token=SimpleNamespace(pad_mask=token_mask, chain_type=chain_type)
    )

    output = client.forward(f_input, [])  # type: ignore[arg-type]

    assert model.return_embeddings
    assert "trunk" not in output
    assert output["affinity"]["p_activity"].shape == (1,)
    assert output["affinity"]["crop_token_count"].item() == 16
