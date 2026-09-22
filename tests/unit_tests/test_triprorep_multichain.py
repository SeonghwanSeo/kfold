"""Native TriProRep chain identity, attention isolation, and apo-axis routing."""

from types import SimpleNamespace as NS

import pytest
import torch

from kfold.model.layers.struct_enc import nn as struct_nn
from kfold.model.layers.struct_enc.backbone import ProteinNetEncoder
from kfold.model.modules.prot_struct_encoder import StructureEncoder


def make_encoder(monkeypatch, layers=1):
    monkeypatch.setattr(struct_nn, "enable_init", True)
    torch.manual_seed(42)
    return ProteinNetEncoder(embed_dim=16, encoder_depth=layers, encoder_heads=2).eval()


def test_real_chain_embedding_and_object_attention(monkeypatch):
    model = make_encoder(monkeypatch)
    seq = torch.tensor([[4, 5, 6, 7, 8, 9]])
    bb = seq + 1
    fa = seq + 2
    # A/B can attend across their boundary, C is a separate object.
    groups = torch.tensor([[1, 1, 1, 1, 3, 3]])
    positions = torch.tensor([[0, 1, 0, 1, 0, 1]])
    chains = torch.tensor([[0, 0, 1, 1, 0, 0]])
    with torch.no_grad():
        legacy = model(seq, bb, fa, groups, positions)
        explicit_zero = model(seq, bb, fa, groups, positions, torch.zeros_like(chains))
        torch.testing.assert_close(legacy, explicit_zero, rtol=0, atol=0)
        actual = model(seq, bb, fa, groups, positions, chains)
        assert not torch.allclose(actual[:, 2:4], legacy[:, 2:4])
        # Prove learned row 1 reaches B, interacts with A, and cannot leak to C.
        model.chain_embedding.weight[1, :8] += 4
        changed = model(seq, bb, fa, groups, positions, chains)
        assert not torch.allclose(changed[:, :2], actual[:, :2])
        assert not torch.allclose(changed[:, 2:4], actual[:, 2:4])
        torch.testing.assert_close(changed[:, 4:], actual[:, 4:], rtol=0, atol=0)


@pytest.mark.parametrize("num_apo", [1, 5])
def test_encoder_wrapper_preserves_chain_ids_across_apo_axis(monkeypatch, num_apo):
    model = make_encoder(monkeypatch)
    wrapper = StructureEncoder.__new__(StructureEncoder)
    torch.nn.Module.__init__(wrapper)
    wrapper.encoder = model
    wrapper.offset = 4
    seq = torch.tensor([[4, 5, 6, 7, 8]])
    bb = torch.arange(1, 6).reshape(1, 5, 1) + torch.arange(num_apo)
    bb[:, -1, :] = -1
    fa = bb.clone()
    physical_ids = torch.tensor([[1, 1, 2, 2, 0]])
    groups = torch.tensor([[1, 1, 1, 1, 0]])
    positions = torch.tensor([[0, 1, 0, 1, 0]])
    chains = torch.tensor([[0, 0, 1, 1, 0]])
    valid = bb[:, :, 0] != -1
    features = NS(
        sequence=NS(
            seq_token_id=seq,
            bb_struct_token_id=bb,
            fa_struct_token_id=fa,
            asym_id=physical_ids,
            pos_id=positions,
            pad_mask=valid,
            is_protein=valid,
        ),
        token=NS(
            seq_token_index=torch.arange(5).unsqueeze(0), pad_mask=valid, is_protein=valid
        ),
    )
    captured = []

    def capture(_module, _args, kwargs):
        captured.append(kwargs)

    hook = model.register_forward_pre_hook(capture, with_kwargs=True)
    with torch.no_grad():
        actual = wrapper._forward(features, groups, positions, chains)
        hook.remove()
        assert captured[0]["chain_ids"].shape == (num_apo, 5)
        torch.testing.assert_close(captured[0]["chain_ids"], chains.expand(num_apo, -1))
        torch.testing.assert_close(captured[0]["pos_id"], positions.expand(num_apo, -1))
        expected = torch.stack(
            [
                model(
                    seq.masked_fill(~valid, 0) + 4,
                    bb[:, :, slot] + 4,
                    fa[:, :, slot] + 4,
                    groups.masked_fill(~valid, -1),
                    positions,
                    chains,
                )
                for slot in range(num_apo)
            ]
        ).mean(0)
        expected *= valid.unsqueeze(-1)
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="structure_chain_id"):
        wrapper._forward(features, groups, positions, torch.full_like(chains, 100))


def test_structure_only_group_ids_for_multiple_objects_and_batched_input():
    from kfold.inference.sequential_tokenization import apply_apo_structure_tokens

    # BOS, two residues, EOS for each of four chains; fifth chain is untouched.
    ids = torch.arange(1, 6).repeat_interleave(4)
    features = NS(
        is_batched=True,
        batch_size=1,
        sequence=NS(
            asym_id=ids.unsqueeze(0),
            pos_id=torch.tensor([0, 1, 2, 3] * 5).unsqueeze(0),
            bb_struct_token_id=torch.full((1, 20, 2), -1),
            fa_struct_token_id=torch.full((1, 20, 2), -1),
        ),
    )
    records = []
    for chain in range(1, 5):
        records.append(
            [
                {
                    "seq": "GG",
                    "coords": torch.zeros(2, 37, 3).numpy(),
                    "targets": [(chain, 0, 2)],
                    "structure_group": [1, 2] if chain <= 2 else [3, 4],
                }
            ]
            * 2
        )

    class Tokenizer:
        def tokenize(self, seq, xyz):
            assert len(seq) == 2  # Never concatenate chains for tokenization.
            return {
                "bb_token_id": torch.tensor([10, 11]),
                "fa_token_id": torch.tensor([12, 13]),
            }

    groups, positions, chains = apply_apo_structure_tokens(features, records, Tokenizer())
    assert groups.shape == positions.shape == chains.shape == (1, 20)
    assert groups[0, [1, 5, 9, 13, 17]].tolist() == [1, 1, 3, 3, 5]
    assert chains[0, [1, 5, 9, 13, 17]].tolist() == [0, 1, 0, 1, 0]
    for start in (1, 5, 9, 13):
        assert positions[0, start : start + 2].tolist() == [0, 1]
    assert positions[0, 17:19].tolist() == [1, 2]  # Untouched chain stays native.
    assert features.sequence.asym_id[0, [1, 5, 9, 13, 17]].tolist() == [1, 2, 3, 4, 5]
