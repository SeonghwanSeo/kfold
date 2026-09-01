from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import kfold.constants as C
from kfold.inference.affinity import (
    PerQueryAffinityConfig,
    PerQueryAffinityPredictor,
    build_per_query_affinity_inputs,
)
from kfold.inference.pl_client import InferenceConfig, KFoldInferenceClient
from kfold.model.modules.affinity_pairformer import AffinityPairformer
from kfold.training.affinity.crop import select_pocket_annotation_crop
from kfold.training.affinity.pair_storage import (
    crop_full_cross_payload,
    full_cross_pl_distogram_profile,
    pack_full_cross_pair_storage,
)


def _symmetric_logits(length: int, *, generator: torch.Generator) -> torch.Tensor:
    logits = torch.randn(length, length, 64, generator=generator)
    return (logits + logits.transpose(0, 1)) / 2


@pytest.mark.parametrize("seed", [7, 19])
def test_online_query_window_matches_submitted_cache_path(seed: int) -> None:
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

    packed = pack_full_cross_pair_storage(
        {
            "s_inputs": s_inputs.numpy(),
            "s_lm": s_lm.numpy(),
            "z": z.numpy(),
            "distogram_logits": logits.numpy(),
            "token_mask": token_mask.numpy(),
            "chain_type": chain_type.numpy(),
        }
    )
    profile = full_cross_pl_distogram_profile(packed.arrays)
    cached_indices = select_pocket_annotation_crop(
        token_mask=token_mask,
        chain_type=chain_type,
        protein_min_distance=profile.protein_min_expected_distance,
        max_tokens=config.max_tokens,
        max_protein_tokens=config.max_protein_tokens,
        neighborhood_size=config.neighborhood_size,
        require_contiguous_monomer=True,
    )
    cached = crop_full_cross_payload(packed.arrays, crop_indices=cached_indices)

    online = build_per_query_affinity_inputs(
        s_inputs=s_inputs,
        s_lm=s_lm,
        z=z,
        distogram_logits=logits,
        token_mask=token_mask,
        chain_type=chain_type,
        config=config,
    )

    assert torch.equal(online.crop_indices, cached["crop_indices"])
    assert torch.equal(online.s_inputs[0], cached["s_inputs"])
    assert torch.equal(online.s_lm[0], cached["s_lm"])
    assert torch.equal(online.z[0], cached["z"])
    assert torch.equal(online.distogram_features[0], cached["distogram_features"])
    assert torch.equal(online.token_mask[0], cached["token_mask"])
    assert torch.equal(
        online.protein_mask[0],
        cached["token_mask"] & (cached["chain_type"] == C.ChainType.PROTEIN.value),
    )
    assert torch.equal(
        online.ligand_mask[0],
        cached["token_mask"] & (cached["chain_type"] == C.ChainType.LIGAND.value),
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
            trunk = {
                "s_inputs": torch.randn(length, 384, generator=generator),
                "s_lm": torch.randn(length, 384, generator=generator),
                "z": torch.randn(length, length, 256, generator=generator),
            }
            output = {
                "distogram": {
                    "distogram": _symmetric_logits(length, generator=generator)
                },
                "diffusion": {"coordinates": torch.zeros(1, 1, 3)},
                "confidence": {},
            }
            if self.return_embeddings:
                output["trunk"] = trunk
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
