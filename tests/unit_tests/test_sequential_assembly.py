import dataclasses
from types import SimpleNamespace as NS

import numpy as np
import pytest

from kfold.data.pipelines.prior_sampling import PriorSampler
from kfold.inference.assembly import (
    PriorObject,
    apply_prior_groups,
    atom_keys,
    select_top1,
    subset_query,
    subset_sources,
    validate_plan,
)
from kfold.inference.data_pipeline import InputDataPipeline, ResolvedStructureSources
from kfold.inference.query import Query


@dataclasses.dataclass
class Sequence:
    id: list

    @property
    def ids(self):
        return self.id


def query():
    return Query(
        name="test",
        sequences=[Sequence(["A", "B"]), Sequence(["L"])],
        yaml="name: test",
        assembly={"stages": [{"id": "pl", "chains": ["A", "L"]}]},
    )


def structure(names):
    chains, metadata = [], []
    for i, name in enumerate(names, 1):
        chains.append(
            NS(
                asym_id=i,
                num_residues=1,
                num_atoms=2,
                residue=NS(iter_residue_atoms=lambda r: range(2)),
                atom=NS(name=np.array(["C1", "N1"])),
            )
        )
        metadata.append(NS(asym_id=i, name=name))
    return NS(chains=chains, metadata=NS(chains=metadata), num_atoms=2 * len(names))


def test_plan_and_subset_preserve_physical_copies():
    q = query()
    assert validate_plan(q)[-1] == {"id": "final", "chains": ["A", "B", "L"]}
    sub = subset_query(q, ["A", "L"])
    assert sub.sequences[0].ids == ["A"]
    assert q.sequences[0].ids == ["A", "B"]
    assert sub.assembly is None


def test_explicit_direct_control_plan():
    from kfold.inference.sequential import execution_plan

    q = query()
    with pytest.raises(ValueError, match="without assembly"):
        execution_plan(q, direct=True)
    q.assembly = None
    assert execution_plan(q, direct=True) == [{"id": "final", "chains": ["A", "B", "L"]}]
    with pytest.raises(ValueError):
        execution_plan(q)


@pytest.mark.parametrize("chains", [["A", "X"], ["A", "A"], ["A"], ["A", "B", "L"]])
def test_bad_plan(chains):
    q = query()
    q.assembly["stages"][0]["chains"] = chains
    with pytest.raises(ValueError):
        validate_plan(q)


def test_object_split_and_covalent_boundary():
    q = query()
    q.assembly["stages"].append({"id": "other", "chains": ["B", "L"]})
    with pytest.raises(ValueError, match="splits"):
        validate_plan(q)
    q = query()
    q.bonds = [NS(atom1=("A", 1, "N"), atom2=("B", 1, "C"))]
    with pytest.raises(ValueError, match="covalent"):
        validate_plan(q)


def test_disjoint_and_nested_stages():
    q = query()
    q.sequences.append(Sequence(["D", "E"]))
    q.assembly["stages"] += [
        {"id": "de", "chains": ["D", "E"]},
        {"id": "merge", "chains": ["A", "L", "D", "E"]},
    ]
    assert len(validate_plan(q)) == 4


def test_shared_rigid_transform_and_reordered_atoms(tmp_path):
    sub = structure(["A", "L"])
    xyz = np.random.default_rng(1).normal(size=(4, 3)).astype(np.float32)
    obj = PriorObject(atom_keys(sub), xyz)
    obj.save(tmp_path / "object.npz")
    obj = PriorObject.load(tmp_path / "object.npz")
    full = structure(["L", "B", "A"])
    priors = np.random.default_rng(2).normal(size=(5, 6, 3)).astype(np.float32)
    unchanged = priors[:, 2:4].copy()
    apply_prior_groups(
        full, priors, [obj], PriorSampler.inference_mode(), np.random.default_rng(3)
    )
    indices = [atom_keys(full).index(k) for k in obj.keys]

    def distance(x):
        return np.linalg.norm(x[:, None] - x[None, :], axis=-1)

    for sample in priors:
        np.testing.assert_allclose(distance(sample[indices]), distance(xyz), atol=1e-5)
    np.testing.assert_array_equal(unchanged, priors[:, 2:4])
    assert not np.allclose(priors[0, indices], priors[1, indices])


def test_nonfinite_missing_duplicate_atoms():
    with pytest.raises(ValueError):
        PriorObject([("A", 1, "N")], [[np.nan, 0, 0]])
    with pytest.raises(ValueError):
        PriorObject([("A", 1, "N")] * 2, np.zeros((2, 3)))
    obj = PriorObject([("A", 1, "C1")], np.zeros((1, 3)))
    with pytest.raises(ValueError, match="mapping"):
        apply_prior_groups(
            structure(["A"]),
            np.zeros((1, 2, 3)),
            [obj],
            PriorSampler.inference_mode(),
            np.random.default_rng(0),
        )


def test_source_mapping_preserves_original_apo_and_tokens():
    full, sub = structure(["A", "B", "L"]), structure(["B", "L"])
    a, b = np.ones((2, 1, 37, 3)), np.ones((2, 1, 37, 3)) * 2
    sources = ResolvedStructureSources(
        2,
        {1: a, 2: b},
        [{1: a[0], 2: b[0]}],
        [[{"targets": [(1, 0, 1), (2, 0, 1)], "coords": a[0], "seq": "G"}]],
    )
    result = subset_sources(full, sub, sources)
    assert result.apo_coords[1] is b
    assert set(result.prior_sources[0]) == {1}
    assert result.struct_token_records[0][0]["targets"] == [(1, 0, 1)]
    assert sources.struct_token_records[0][0]["targets"] == [(1, 0, 1), (2, 0, 1)]


def test_selection_no_oracle_and_deterministic_ties():
    candidates = [
        dict(seed=2, sample=0, ranking_score=0.9),
        dict(seed=1, sample=1, ranking_score=0.9),
        dict(seed=1, sample=0, ranking_score=0.8),
    ]
    assert select_top1(candidates) == candidates[1]
    with pytest.raises(ValueError):
        select_top1([dict(seed=1, sample=0, ranking_score=float("nan"))])


def test_ordinary_pipeline_rejects_assembly():
    with pytest.raises(ValueError, match="inference_sequential"):
        InputDataPipeline(None).run(query())


def test_orchestration_full_trunk_input_top1_and_resume(tmp_path, monkeypatch):
    import torch

    from kfold.inference.dataset import InferenceDataset
    from kfold.inference.sequential import run_query

    monkeypatch.setattr(InferenceDataset, "pad_input", lambda self, f: f)
    calls = []

    class Pipeline:
        ccd = None
        num_apo = None

        def read_query(self, q):
            return structure([c for s in q.sequences for c in s.ids])

        def resolve_structure_sources(self, *args):
            return ResolvedStructureSources(1, {}, [{}, {}], [])

        def run(self, q, *, sources, prior_groups, stage_index):
            struct = self.read_query(q)
            calls.append((q.seed, struct.num_atoms, len(prior_groups)))
            p = np.zeros((2, struct.num_atoms, 3), dtype=np.float32)
            apply_prior_groups(
                struct,
                p,
                prior_groups,
                PriorSampler.inference_mode(),
                np.random.default_rng(1),
            )
            f = NS(
                atom=NS(prior_coords=torch.from_numpy(p.transpose(1, 0, 2))),
                num_tokens=struct.num_atoms,
            )
            return struct, None, f, []

    class Backend:
        def predict(self, q, struct, features, records, out):
            candidates = []
            for sample in range(2):
                stem = f"{q.name}_seed-{q.seed}_sample-{sample}"
                PriorObject(
                    atom_keys(struct), np.ones((struct.num_atoms, 3)) * (q.seed + sample)
                ).save(out / f"{stem}_atoms.npz")
                (out / f"{stem}.cif").write_text("test fixture\n")
                candidates.append(
                    dict(
                        stem=stem,
                        seed=q.seed,
                        sample=sample,
                        ranking_score=q.seed + sample,
                    )
                )
            return candidates

    best = run_query(query(), Pipeline(), [1, 2], 2, tmp_path, Backend())
    assert calls == [(1, 4, 0), (2, 4, 0), (1, 6, 1), (2, 6, 1)]
    assert (best["seed"], best["sample"]) == (2, 1)
    assert run_query(query(), Pipeline(), [1, 2], 2, tmp_path, Backend()) == best
    assert len(calls) == 4
    (tmp_path / "pl/seed-1/test_seed-1_sample-0.cif").write_text("corrupted")
    with pytest.raises(ValueError, match="Changed result"):
        run_query(query(), Pipeline(), [1, 2], 2, tmp_path, Backend())

    class FailingBackend(Backend):
        failed = False

        def predict(self, q, struct, features, records, out):
            result = super().predict(q, struct, features, records, out)
            if q.seed == 2 and not self.failed:
                self.failed = True
                raise RuntimeError("simulated interrupted seed")
            return result

    backend = FailingBackend()
    retry = tmp_path / "retry"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_query(query(), Pipeline(), [1, 2], 2, retry, backend)
    completed_before = (retry / "pl/seed-1/complete.json").read_bytes()
    assert not (retry / "pl/seed-2").exists()
    result = run_query(query(), Pipeline(), [1, 2], 2, retry, backend)
    assert result["stage"] == "final"
    assert (retry / "pl/seed-1/complete.json").read_bytes() == completed_before
    assert len(list((retry / "pl").glob(".seed-2-*/failure.json"))) == 1


def test_real_smiles_pipeline_preserves_prior_and_apo_features():
    import warnings

    import torch

    from kfold.data.types.ccd import CCD
    from kfold.inference.query import LigandSequence

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ccd = CCD({})
    q = Query(
        name="chemical",
        yaml="",
        seed=5,
        sequences=[
            LigandSequence(id=["A"], smiles="CCO"),
            LigandSequence(id=["B"], smiles="CCN"),
            LigandSequence(id=["L"], smiles="C1CC1"),
        ],
    )
    pipeline = InputDataPipeline(ccd, num_prior_samples=3)
    struct, _, baseline, _ = pipeline.run(q)
    partial = pipeline.read_query(subset_query(q, ["A", "L"]))
    xyz = np.random.default_rng(14).normal(size=(partial.num_atoms, 3)).astype(np.float32)
    obj = PriorObject(atom_keys(partial), xyz)
    _, _, grouped, _ = pipeline.run(q, prior_groups=[obj], stage_index=1)
    keys = atom_keys(struct)
    idx = [keys.index(k) for k in obj.keys]
    for prior in grouped.atom.prior_coords.permute(1, 0, 2):
        torch.testing.assert_close(
            torch.cdist(prior[idx], prior[idx]),
            torch.cdist(torch.from_numpy(xyz), torch.from_numpy(xyz)),
            atol=1e-5,
            rtol=1e-5,
        )
    torch.testing.assert_close(
        grouped.atom.apo_coords, baseline.atom.apo_coords, equal_nan=True
    )
    remaining = [i for i, k in enumerate(keys) if k[0] == "B"]
    torch.testing.assert_close(
        grouped.atom.prior_coords[remaining], baseline.atom.prior_coords[remaining]
    )
    _, _, repeat, _ = pipeline.run(q)
    torch.testing.assert_close(
        repeat.atom.prior_coords, baseline.atom.prior_coords, rtol=0, atol=0
    )


def test_ecsi_consumes_grouped_endpoint_without_chainwise_regeneration():
    import torch

    from kfold.model.modules.structure.ecsi import KFoldECSI

    # Call the actual endpoint sampler on a small CPU fixture. Its augmentation
    # is stubbed to identity so the source-coordinate assertion is exact.
    endpoint = torch.randn(1, 6, 3, 3)
    f = NS(atom=NS(prior_coords=endpoint, pad_mask=torch.ones(1, 6, dtype=torch.bool)))
    sampler = NS(random_augmentation=lambda x, **kw: x)
    actual = KFoldECSI.sample_prior(sampler, f, 3)
    torch.testing.assert_close(actual, endpoint.permute(0, 2, 1, 3), atol=0, rtol=0)


def test_real_protein_ligand_protein_preserves_structure_conditioning(tmp_path):
    import warnings

    import torch
    from rdkit import Chem

    from kfold.data.types.ccd import CCD, Component
    from kfold.inference.query import LigandSequence, ProteinSequence

    mol = Chem.MolFromSmiles("NCC(=O)O")
    for atom, name in zip(mol.GetAtoms(), ["N", "CA", "C", "O", "OXT"], strict=True):
        atom.SetProp("name", name)
    gly = Component.from_mol("GLY", mol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ccd = CCD({"GLY": gly.to_bytes()})
    pdb = tmp_path / "apo.pdb"
    pdb.write_text(
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 90.00           N\n"
        "ATOM      2  CA  GLY A   1       1.450   0.000   0.000  1.00 90.00           C\n"
        "ATOM      3  C   GLY A   1       1.900   1.400   0.000  1.00 90.00           C\n"
        "ATOM      4  O   GLY A   1       1.200   2.300   0.000  1.00 90.00           O\n"
        "END\n"
    )
    q = Query(
        name="plp",
        seed=4,
        yaml="",
        sequences=[
            ProteinSequence(id=["A", "B"], sequence="G", apo=[str(pdb)]),
            LigandSequence(id=["L"], smiles="CCO"),
        ],
    )
    pipeline = InputDataPipeline(ccd, num_prior_samples=2)
    full = pipeline.read_query(q)
    sources = pipeline.resolve_structure_sources(full, q, np.random.default_rng(0))
    sub = subset_query(q, ["A", "L"])
    partial, _, _, _ = pipeline.run(
        sub, sources=subset_sources(full, pipeline.read_query(sub), sources)
    )
    coords = (
        np.random.default_rng(42).normal(size=(partial.num_atoms, 3)).astype(np.float32)
    )
    obj = PriorObject(atom_keys(partial), coords)
    _, _, baseline, records = pipeline.run(q, sources=sources)
    _, _, grouped, grouped_records = pipeline.run(
        q, sources=sources, prior_groups=[obj], stage_index=1
    )
    torch.testing.assert_close(
        grouped.atom.apo_coords, baseline.atom.apo_coords, equal_nan=True
    )
    for before, after in zip(records, grouped_records, strict=True):
        for b, a in zip(before, after, strict=True):
            assert b["targets"] == a["targets"]
            np.testing.assert_array_equal(b["coords"], a["coords"])
    indices = [atom_keys(full).index(k) for k in obj.keys]
    for p in grouped.atom.prior_coords.permute(1, 0, 2):
        torch.testing.assert_close(
            torch.cdist(p[indices], p[indices]),
            torch.cdist(torch.from_numpy(coords), torch.from_numpy(coords)),
            atol=1e-5,
            rtol=1e-5,
        )

    # Re-encoding uses native independent molecule paths, not shared P-L apo.
    five = dataclasses.replace(
        sources,
        num_apo=5,
        apo_coords={k: np.repeat(v, 5, axis=0) for k, v in sources.apo_coords.items()},
        struct_token_records=[r * 5 for r in sources.struct_token_records],
    )
    _, _, prior_only, original_records = pipeline.run(
        q, sources=five, prior_groups=[obj], stage_index=1
    )
    _, _, reencoded, new_records = pipeline.run(
        q, sources=five, prior_groups=[obj], trunk_groups=[obj], stage_index=1
    )
    full_keys = atom_keys(full)
    ia, ib, il = [
        [i for i, k in enumerate(full_keys) if k[0] == n] for n in ["A", "B", "L"]
    ]
    obj_index = {k: i for i, k in enumerate(obj.keys)}
    expected = torch.from_numpy(np.stack([coords[obj_index[full_keys[i]]] for i in ia]))
    torch.testing.assert_close(
        reencoded.atom.apo_coords[ia], expected[:, None].expand(-1, 5, -1)
    )
    torch.testing.assert_close(
        reencoded.atom.apo_coords[ib], prior_only.atom.apo_coords[ib]
    )
    assert not reencoded.atom.apo_mask[il].any()
    torch.testing.assert_close(reencoded.token.apo_uid, prior_only.token.apo_uid)
    torch.testing.assert_close(
        reencoded.atom.ref_space_uid, prior_only.atom.ref_space_uid
    )
    torch.testing.assert_close(
        reencoded.atom.ref_pos[ia + ib], prior_only.atom.ref_pos[ia + ib]
    )
    ligand_xyz = torch.from_numpy(np.stack([coords[obj_index[full_keys[i]]] for i in il]))
    torch.testing.assert_close(
        torch.cdist(reencoded.atom.ref_pos[il], reencoded.atom.ref_pos[il]),
        torch.cdist(ligand_xyz, ligand_xyz),
        atol=1e-5,
        rtol=1e-5,
    )
    shifted = PriorObject(
        obj.keys, obj.coordinates + np.array([10, -4, 3], dtype=np.float32)
    )
    _, _, shifted_input, _ = pipeline.run(
        q, sources=five, trunk_groups=[shifted], stage_index=1
    )
    torch.testing.assert_close(
        shifted_input.atom.ref_pos[il], reencoded.atom.ref_pos[il], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(reencoded.atom.prior_coords, prior_only.atom.prior_coords)
    # Shared A/B source record was split; B is untouched, A gets predicted atom37.
    b_records = [rs for rs in new_records if rs[0]["targets"][0][0] == 2][0]
    a_records = [rs for rs in new_records if rs[0]["targets"][0][0] == 1][0]
    assert len(a_records) == len(b_records) == 5
    np.testing.assert_array_equal(
        b_records[0]["coords"], original_records[0][0]["coords"]
    )
    assert not np.array_equal(
        a_records[0]["coords"], b_records[0]["coords"], equal_nan=True
    )
    # Actual token insertion consumes the new records, using a coordinate-dependent spy.
    from kfold.inference.structure_tokenization import apply_apo_structure_tokens

    class Encoder:
        def __init__(self):
            self.seen = []

        def tokenize(self, seq, xyz):
            self.seen.append(xyz.copy())
            value = (
                12 if np.array_equal(xyz, a_records[0]["coords"], equal_nan=True) else 7
            )
            return {
                "bb_token_id": torch.full((len(seq),), value),
                "fa_token_id": torch.full((len(seq),), value + 1),
            }

    encoder = Encoder()
    apply_apo_structure_tokens(reencoded, new_records, encoder)
    assert len(encoder.seen) == 10
    assert (reencoded.sequence.bb_struct_token_id[1] == 12).all()
    assert (reencoded.sequence.bb_struct_token_id[4] == 7).all()
    # Refuse missing atoms and overlapping objects rather than partial replacement.
    with pytest.raises(ValueError, match="mapping"):
        pipeline.run(
            q,
            sources=five,
            trunk_groups=[PriorObject(obj.keys[:-1], obj.coordinates[:-1])],
        )
    with pytest.raises(ValueError, match="Overlapping"):
        pipeline.run(q, sources=five, trunk_groups=[obj, obj])
    # Deterministic replay, with the original sources still immutable.
    _, _, again, _ = pipeline.run(
        q, sources=five, prior_groups=[obj], trunk_groups=[obj], stage_index=1
    )
    torch.testing.assert_close(again.atom.ref_pos, reencoded.atom.ref_pos, rtol=0, atol=0)
