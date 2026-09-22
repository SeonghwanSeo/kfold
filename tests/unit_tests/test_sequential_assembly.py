import dataclasses
from types import SimpleNamespace as NS

import numpy as np
import pytest

from kfold.data.pipelines.prior_sampling import PriorSampler
from kfold.inference.assembly import (
    PriorObject,
    apply_prior_groups,
    atom_keys,
    select_apo_ensemble,
    select_top1,
    subset_query,
    subset_sources,
    validate_plan,
)
from kfold.inference.sequential_pipeline import (
    InputDataPipeline,
    ResolvedStructureSources,
)
from kfold.inference.sequential_query import Query


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
                is_protein=name != "L",
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


def test_seed_apo_selection_keeps_each_seed_and_independent_prior(tmp_path):
    from kfold.inference.assembly import select_seed_apo_ensemble

    keys = [("H", 1, "CA"), ("L", 1, "CA")]
    prior = PriorObject(keys, np.zeros((2, 3)))
    candidates = []
    for seed in [11, 12]:
        for sample in range(5):
            path = tmp_path / f"{seed}-{sample}.npz"
            xyz = np.array([[seed, sample, 0], [seed, sample, 7]], dtype=np.float32)
            PriorObject(keys[::-1], xyz[::-1]).save(path)
            candidates.append(
                dict(
                    seed=seed,
                    sample=sample,
                    ranking_score=100 * (seed == 12) + sample,
                    atoms=str(path),
                )
            )
    obj, selected = select_seed_apo_ensemble(candidates, [11, 12], prior)
    assert [(c["seed"], c["sample"]) for c in selected] == [(11, 4), (12, 4)]
    np.testing.assert_array_equal(obj.apo_coordinates[:, :, 2], [[0, 7], [0, 7]])
    np.testing.assert_array_equal(obj.apo_coordinates[:, 0, 0], [11, 12])
    np.testing.assert_array_equal(obj.coordinates, prior.coordinates)
    with pytest.raises(ValueError, match="generation seeds"):
        select_seed_apo_ensemble(candidates[:5], [11, 12], prior)


def test_apo_generation_seed_schedule_matches_native_and_avoids_collisions():
    from kfold.inference.sequential import apo_generation_seeds

    seeds = list(range(1, 11))
    schedule = apo_generation_seeds(seeds, dict.fromkeys(seeds, 5))
    assert schedule[1] == [11, 12, 13, 14, 15]
    assert schedule[10] == [101, 102, 103, 104, 105]
    assert sum(map(len, schedule.values())) * 5 == 250
    many = apo_generation_seeds([1, 2], {1: 12, 2: 11})
    assert len(set(many[1] + many[2])) == 23
    assert apo_generation_seeds([0], {0: 2})[0] == [1, 2]
    with pytest.raises(ValueError, match="unique nonnegative"):
        apo_generation_seeds([1, 1], {1: 5})


def test_ordinary_pipeline_rejects_assembly():
    with pytest.raises(ValueError, match="inference_sequential"):
        InputDataPipeline(None).run(query())


@pytest.mark.parametrize(
    "conditioning", ["prior_only", "prior_and_trunk", "prior_and_trunk_multichain"]
)
@pytest.mark.parametrize("apo_count", [2, 5])
def test_orchestration_full_trunk_input_top1_and_resume(
    tmp_path, monkeypatch, conditioning, apo_count
):
    import torch

    from kfold.inference.sequential import run_query as run

    def run_query(*args, **kwargs):
        return run(*args, conditioning=conditioning, **kwargs)

    from kfold.inference.sequential_dataset import InferenceDataset

    monkeypatch.setattr(InferenceDataset, "pad_input", lambda self, f: f)
    calls = []
    samples = apo_count

    class Pipeline:
        ccd = None
        num_apo = None

        def read_query(self, q):
            return structure([c for s in q.sequences for c in s.ids])

        def resolve_structure_sources(self, struct, q, rng):
            return ResolvedStructureSources(
                apo_count,
                {1: np.full((apo_count, 1, 37, 3), q.seed, dtype=np.float32)},
                [{}] * samples,
                [],
            )

        def run(
            self,
            q,
            *,
            sources,
            prior_groups,
            stage_index,
            trunk_groups=(),
            multichain_structure=False,
        ):
            parent_seed = int(sources.apo_coords[1][0, 0, 0, 0])
            expected_parent = (q.seed - 1) // 10 if q.seed > 10 else q.seed
            assert parent_seed == expected_parent
            if (
                stage_index
                and conditioning != "prior_only"
                and trunk_groups[0].apo_coordinates is not None
            ):
                assert trunk_groups is prior_groups
                assert trunk_groups[0].apo_coordinates.shape[0] == apo_count
                np.testing.assert_array_equal(
                    trunk_groups[0].apo_coordinates[:, 0, 0],
                    [
                        parent_seed * 10 + slot + samples - 1
                        for slot in range(1, apo_count + 1)
                    ],
                )
                # ECSI still receives global top-1, independent of apo slot 0.
                np.testing.assert_array_equal(
                    trunk_groups[0].coordinates, 20 + apo_count + samples - 1
                )
            struct = self.read_query(q)
            calls.append((q.seed, struct.num_atoms, len(prior_groups)))
            p = np.zeros((samples, struct.num_atoms, 3), dtype=np.float32)
            apply_prior_groups(
                struct,
                p,
                prior_groups,
                PriorSampler.inference_mode(),
                np.random.default_rng(1),
            )
            f = NS(
                atom=NS(
                    prior_coords=torch.from_numpy(p.transpose(1, 0, 2)),
                    apo_coords=torch.zeros((struct.num_atoms, apo_count, 3)),
                    apo_mask=torch.ones((struct.num_atoms, apo_count), dtype=torch.bool),
                    ref_pos=torch.zeros((struct.num_atoms, 3)),
                    ref_space_uid=torch.arange(struct.num_atoms),
                ),
                token=NS(
                    apo_uid=torch.arange(struct.num_atoms),
                    apo_repr_coords=torch.zeros((struct.num_atoms, apo_count, 3)),
                    apo_frame_coords=torch.zeros((struct.num_atoms, apo_count, 3, 3)),
                ),
                num_tokens=struct.num_atoms,
            )
            return struct, None, f, []

    class Backend:
        def predict(self, q, struct, features, records, out):
            candidates = []
            for sample in range(samples):
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

    best = run_query(query(), Pipeline(), [1, 2], samples, tmp_path, Backend())
    intermediate_seeds = (
        [1, 2]
        if conditioning == "prior_only"
        else [s * 10 + i for s in [1, 2] for i in range(1, apo_count + 1)]
    )
    assert calls == [(s, 4, 0) for s in intermediate_seeds] + [(1, 6, 1), (2, 6, 1)]
    assert (best["seed"], best["sample"]) == (2, samples - 1)
    assert run_query(query(), Pipeline(), [1, 2], samples, tmp_path, Backend()) == best
    assert len(calls) == len(intermediate_seeds) + 2
    if conditioning != "prior_only":
        import json

        for parent in [1, 2]:
            root = tmp_path / "pl" / f"parent-seed-{parent}"
            picked = json.loads((root / "apo_selection.json").read_text())
            assert [(c["seed"], c["sample"]) for c in picked] == [
                (parent * 10 + slot, samples - 1) for slot in range(1, apo_count + 1)
            ]
            assert {c["source_seed"] for c in picked} == {parent}
            assert PriorObject.load(
                root / "selected_ensemble.npz"
            ).apo_coordinates.shape == (apo_count, 4, 3)
        input_path = tmp_path / "final/seed-1/input.json"
        saved = input_path.read_text()
        old = json.loads(saved)
        old.pop("conditioning_schema")
        input_path.write_text(json.dumps(old))
        with pytest.raises(ValueError, match="schema changed"):
            run_query(query(), Pipeline(), [1, 2], samples, tmp_path, Backend())
        input_path.write_text(saved)
        with pytest.raises(ValueError, match="apo policy changed"):
            run_query(query(), Pipeline(), [1], samples, tmp_path, Backend())
        with pytest.raises(ValueError, match="apo policy changed"):
            run_query(query(), Pipeline(), [1, 2], samples + 1, tmp_path, Backend())
        legacy = tmp_path / "legacy"
        (legacy / "pl").mkdir(parents=True)
        before = len(calls)
        with pytest.raises(ValueError, match="Missing per-seed apo policy"):
            run_query(query(), Pipeline(), [1, 2], samples, legacy, Backend())
        assert len(calls) == before
    first_seed = intermediate_seeds[0]
    first_root = tmp_path / "pl"
    if conditioning != "prior_only":
        first_root /= "parent-seed-1"
    (first_root / f"seed-{first_seed}/test_seed-{first_seed}_sample-0.cif").write_text(
        "corrupted"
    )
    with pytest.raises(ValueError, match="Changed result"):
        run_query(query(), Pipeline(), [1, 2], samples, tmp_path, Backend())

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
        run_query(query(), Pipeline(), [1, 2], samples, retry, backend)
    completed_file = (
        retry / first_root.relative_to(tmp_path) / f"seed-{first_seed}/complete.json"
    )
    completed_before = completed_file.read_bytes()
    failed_stage = "pl" if conditioning == "prior_only" else "final"
    assert not (retry / failed_stage / "seed-2").exists()
    result = run_query(query(), Pipeline(), [1, 2], samples, retry, backend)
    assert result["stage"] == "final"
    assert completed_file.read_bytes() == completed_before
    assert len(list((retry / failed_stage).glob(".seed-2-*/failure.json"))) == 1

    # A later intermediate must receive the ensemble for its own parent seed.
    nested = query()
    nested.sequences.append(Sequence(["D", "E"]))
    nested.assembly["stages"].append({"id": "merge", "chains": ["A", "L", "D"]})
    nested_best = run_query(
        nested, Pipeline(), [1, 2], samples, tmp_path / "nested", Backend()
    )
    assert nested_best["stage"] == "final"

    # A provided GT/experimental intermediate bypasses binary prediction entirely.
    supplied = tmp_path / "provided.npz"
    PriorObject(atom_keys(structure(["A", "L"])), np.ones((4, 3))).save(supplied)
    calls.clear()
    oracle_out = tmp_path / "provided_run"
    result = run_query(
        query(),
        Pipeline(),
        [1, 2],
        samples,
        oracle_out,
        Backend(),
        provided_intermediates={"pl": supplied},
    )
    assert calls == [(1, 6, 1), (2, 6, 1)]
    assert result["stage"] == "final"
    assert not (oracle_out / "pl/selection.json").exists()
    assert (oracle_out / "pl/provided_structure.json").is_file()
    assert not list((oracle_out / "pl").glob("seed-*"))
    assert (
        run_query(
            query(),
            Pipeline(),
            [1, 2],
            samples,
            oracle_out,
            Backend(),
            provided_intermediates={"pl": supplied},
        )
        == result
    )
    assert len(calls) == 2
    PriorObject(atom_keys(structure(["A", "L"])), np.ones((4, 3)) * 2).save(supplied)
    with pytest.raises(ValueError, match="changed during resume"):
        run_query(
            query(),
            Pipeline(),
            [1, 2],
            samples,
            oracle_out,
            Backend(),
            provided_intermediates={"pl": supplied},
        )
    with pytest.raises(ValueError, match="non-final stage"):
        run_query(
            query(),
            Pipeline(),
            [1],
            2,
            tmp_path / "bad_provided",
            Backend(),
            provided_intermediates={"final": supplied},
        )


def test_real_smiles_pipeline_preserves_prior_and_apo_features():
    import warnings

    import torch

    from kfold.data.types.ccd import CCD
    from kfold.inference.sequential_query import LigandSequence

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

    from kfold.model.modules.ecsi import KFoldECSI

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
    from kfold.inference.sequential_query import LigandSequence, ProteinSequence

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
    from kfold.inference.sequential_tokenization import apply_apo_structure_tokens

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

    # Partial GT retains all sequence/atom identities. Missing protein atoms are
    # masked in the trunk and initialized ONLY in the diffusion endpoint.
    mask = np.ones(len(obj.keys), dtype=bool)
    mask[[i for i, k in enumerate(obj.keys) if k[0] == "A" and k[2] == "N"]] = False
    partial_gt = PriorObject(obj.keys, coords, mask)
    partial_gt.save(tmp_path / "partial_gt.npz")
    partial_gt = PriorObject.load(tmp_path / "partial_gt.npz")
    _, _, masked, masked_records = pipeline.run(
        q,
        sources=five,
        prior_groups=[partial_gt],
        trunk_groups=[partial_gt],
        stage_index=1,
    )
    assert masked.atom.pad_mask.sum() == reencoded.atom.pad_mask.sum()
    ni = full_keys.index(("A", 1, "N"))
    assert not masked.atom.apo_mask[ni].any()
    assert torch.isfinite(masked.atom.prior_coords).all()
    torch.testing.assert_close(masked.atom.apo_coords[ib], prior_only.atom.apo_coords[ib])
    apply_apo_structure_tokens(masked, masked_records, Encoder())
    assert (masked.sequence.bb_struct_token_id[1] == -1).all()
    assert (masked.sequence.fa_struct_token_id[1] == -1).all()
    assert (masked.sequence.bb_struct_token_id[4] == 7).all()
    observed_i = [full_keys.index(k) for i, k in enumerate(obj.keys) if mask[i]]
    expected_gt = torch.from_numpy(coords[mask])
    for endpoint in masked.atom.prior_coords.permute(1, 0, 2):
        torch.testing.assert_close(
            torch.cdist(endpoint[observed_i], endpoint[observed_i]),
            torch.cdist(expected_gt, expected_gt),
            atol=1e-5,
            rtol=1e-5,
        )


def test_multichain_structure_repr_jointly_tokenizes_and_groups_attention(tmp_path):
    import warnings

    import torch
    from rdkit import Chem

    from kfold.data.types.ccd import CCD, Component
    from kfold.inference.sequential_query import LigandSequence, ProteinSequence
    from kfold.inference.sequential_tokenization import apply_apo_structure_tokens

    mol = Chem.MolFromSmiles("NCC(=O)O")
    for atom, name in zip(mol.GetAtoms(), ["N", "CA", "C", "O", "OXT"], strict=True):
        atom.SetProp("name", name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ccd = CCD({"GLY": Component.from_mol("GLY", mol).to_bytes()})
    pdb = tmp_path / "apo.pdb"
    pdb.write_text(
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 90.00           N\n"
        "ATOM      2  CA  GLY A   1       1.450   0.000   0.000  1.00 90.00           C\n"
        "ATOM      3  C   GLY A   1       1.900   1.400   0.000  1.00 90.00           C\n"
        "ATOM      4  O   GLY A   1       1.200   2.300   0.000  1.00 90.00           O\n"
        "END\n"
    )
    q = Query(
        name="joint",
        seed=7,
        yaml="",
        sequences=[
            ProteinSequence(id=["A", "B"], sequence="G", apo=[str(pdb)]),
            LigandSequence(id=["L"], smiles="CCO"),
        ],
    )
    pipeline = InputDataPipeline(ccd, num_prior_samples=2)
    full = pipeline.read_query(q)
    sources = pipeline.resolve_structure_sources(full, q, np.random.default_rng(0))
    sources = dataclasses.replace(
        sources,
        num_apo=3,
        apo_coords={k: np.repeat(v, 3, axis=0) for k, v in sources.apo_coords.items()},
        struct_token_records=[records * 3 for records in sources.struct_token_records],
    )
    keys = atom_keys(full)
    ab_keys = [key for key in keys if key[0] in {"A", "B"}]
    ab_coords = np.random.default_rng(8).normal(size=(len(ab_keys), 3)).astype(np.float32)
    obj = PriorObject(ab_keys, ab_coords)

    _, _, chainwise, chainwise_records = pipeline.run(
        q, sources=sources, trunk_groups=[obj], stage_index=1
    )
    _, _, joint, joint_records = pipeline.run(
        q,
        sources=sources,
        trunk_groups=[obj],
        multichain_structure=True,
        stage_index=1,
    )
    # Only the structure-representation path changes between these modes.
    torch.testing.assert_close(joint.atom.apo_coords, chainwise.atom.apo_coords)
    torch.testing.assert_close(joint.atom.apo_mask, chainwise.atom.apo_mask)
    torch.testing.assert_close(joint.token.apo_uid, chainwise.token.apo_uid)
    torch.testing.assert_close(joint.atom.ref_pos, chainwise.atom.ref_pos)
    torch.testing.assert_close(
        joint.atom.prior_coords, chainwise.atom.prior_coords, equal_nan=True
    )
    assert len(chainwise_records) == 2
    assert len(joint_records) == 1
    assert len(joint_records[0]) == 3
    record = joint_records[0][0]
    assert record["seq"] == "GG"
    assert record["segments"] == [(1, 0, 1, 0, 1), (2, 0, 1, 1, 2)]
    assert record["structure_group"] == [1, 2]
    assert np.diff(record["residue_index"])[0] > 1
    expected = np.stack([ab_coords[ab_keys.index(key)] for key in keys if key[0] == "A"])
    expected_b = np.stack(
        [ab_coords[ab_keys.index(key)] for key in keys if key[0] == "B"]
    )
    mapped_a = full.chains[0].map_atom_coords_to_residue_coords(expected)
    mapped_b = full.chains[1].map_atom_coords_to_residue_coords(expected_b)
    np.testing.assert_array_equal(
        record["coords"], np.concatenate([mapped_a, mapped_b], axis=0)
    )

    class Encoder:
        def __init__(self):
            self.calls = []

        def tokenize(self, seq, xyz, residue_index=None):
            self.calls.append(
                (seq, xyz.copy(), None if residue_index is None else residue_index.copy())
            )
            values = torch.arange(len(seq), dtype=torch.long) + 20
            return {"bb_token_id": values, "fa_token_id": values + 100}

    encoder = Encoder()
    structure_seq_id, structure_pos_id = apply_apo_structure_tokens(
        joint, joint_records, encoder
    )
    assert len(encoder.calls) == 3
    assert all(call[0] == "GG" for call in encoder.calls)
    assert structure_seq_id is not None
    assert structure_pos_id is not None
    # Physical identities stay unchanged; only the structure encoder override joins A/B.
    assert set(joint.sequence.asym_id.tolist()) == {1, 2, 3}
    assert (structure_seq_id[joint.sequence.asym_id == 1] == 1).all()
    assert (structure_seq_id[joint.sequence.asym_id == 2] == 1).all()
    assert (structure_seq_id[joint.sequence.asym_id == 3] == 3).all()
    # Physical positions restart per chain, while structure-only positions retain
    # the same chain gap used by the joint structure tokenizers.
    assert joint.sequence.pos_id[1] == joint.sequence.pos_id[4] == 1
    assert structure_pos_id[1] == 1
    assert structure_pos_id[4] > structure_pos_id[1] + 1
    assert torch.equal(
        structure_pos_id[joint.sequence.asym_id == 3],
        joint.sequence.pos_id[joint.sequence.asym_id == 3],
    )
    assert (joint.sequence.bb_struct_token_id[1] == 20).all()
    assert (joint.sequence.bb_struct_token_id[4] == 21).all()

    class ChainEncoder:
        def tokenize(self, seq, xyz):
            return {
                "bb_token_id": torch.zeros(len(seq), dtype=torch.long),
                "fa_token_id": torch.zeros(len(seq), dtype=torch.long),
            }

    # Distinct intermediate samples keep paired H/L poses in every apo slot;
    # antigen C retains both its original ensemble and a separate UID.
    q3 = q.copy(
        sequences=[*q.sequences, ProteinSequence(id=["C"], sequence="G", apo=[str(pdb)])]
    )
    full3 = pipeline.read_query(q3)
    source3 = pipeline.resolve_structure_sources(full3, q3, np.random.default_rng(0))
    source3 = dataclasses.replace(
        source3,
        num_apo=3,
        apo_coords={k: np.repeat(v, 3, axis=0) for k, v in source3.apo_coords.items()},
        struct_token_records=[r * 3 for r in source3.struct_token_records],
    )
    ensemble = np.stack([ab_coords.copy() for _ in range(3)])
    # Change B relative to A (not just the global rigid frame).
    ensemble[1, [k[0] == "B" for k in ab_keys]] += 5
    ensemble[2, [k[0] == "B" for k in ab_keys]] -= 3
    grouped = PriorObject(ab_keys, ab_coords, apo_coordinates=ensemble)
    grouped.save(tmp_path / "ensemble.npz")
    grouped = PriorObject.load(tmp_path / "ensemble.npz")
    np.testing.assert_array_equal(grouped.apo_coordinates, ensemble)
    _, _, baseline, _ = pipeline.run(q3, sources=source3)
    for multichain in (False, True):
        _, tok, features, recs = pipeline.run(
            q3,
            sources=source3,
            trunk_groups=[grouped],
            multichain_structure=multichain,
        )
        a_uid = features.token.apo_uid[features.token.asym_id == 1]
        b_uid = features.token.apo_uid[features.token.asym_id == 2]
        assert (a_uid == b_uid[0]).all()
        assert (b_uid == a_uid[0]).all()
        other = ~torch.isin(features.token.asym_id, torch.tensor([1, 2]))
        assert not (features.token.apo_uid[other] == a_uid[0]).any()
        torch.testing.assert_close(
            features.token.apo_uid[other], baseline.token.apo_uid[other]
        )
        assert tok.chain.apo_uid[0] == tok.chain.apo_uid[1]
        all_keys = atom_keys(full3)
        indices = [all_keys.index(k) for k in ab_keys]
        torch.testing.assert_close(
            features.atom.apo_coords[indices],
            torch.from_numpy(ensemble.transpose(1, 0, 2)),
        )
        antigen = [i for i, k in enumerate(all_keys) if k[0] == "C"]
        torch.testing.assert_close(
            features.atom.apo_coords[antigen], baseline.atom.apo_coords[antigen]
        )
        changed_records = [r for r in recs if any(t[0] == 2 for t in r[0]["targets"])]
        assert len(changed_records) == 1
        assert not np.array_equal(
            changed_records[0][0]["coords"],
            changed_records[0][1]["coords"],
            equal_nan=True,
        )
        apply_apo_structure_tokens(
            features, recs, Encoder() if multichain else ChainEncoder()
        )


def test_training_model_accepts_structure_encoder_overrides():
    import inspect

    from kfold.model.model_train import KFoldForTrain

    for method in (KFoldForTrain._encode_lm_single, KFoldForTrain.run_trunk):
        parameters = inspect.signature(method).parameters
        assert "structure_seq_id" in parameters
        assert "structure_pos_id" in parameters


def test_multichain_resume_settings_have_a_schema_version():
    from kfold.cli.predict_sequential import _settings

    args = NS(
        seeds=[1],
        share_apo_seeds=None,
        num_samples=5,
        num_recycles=10,
        num_steps=100,
        disable_struct_encoder=False,
        disable_rna_encoder=False,
        cpu_offload=False,
        conditioning="prior_and_trunk_multichain",
        provided_intermediates=None,
    )
    assert _settings(args)["conditioning_schema"] == 3
    assert _settings(args)["apo_policy"] == "per_final_seed_top1_per_generation_seed"
    args.conditioning = "prior_and_trunk"
    assert _settings(args)["conditioning_schema"] == 3
    args.conditioning = "prior_only"
    assert "conditioning_schema" not in _settings(args)


def test_structure_source_expands_all_pdb_models(tmp_path):
    atom_lines = (
        "ATOM      1  N   GLY A   1       {x:6.3f}   0.000   0.000  1.00 90.00           N\n"  # noqa: E501
        "ATOM      2  CA  GLY A   1       {ca:6.3f}   0.000   0.000  1.00 90.00           C\n"  # noqa: E501
        "ATOM      3  C   GLY A   1       {c:6.3f}   1.400   0.000  1.00 90.00           C\n"  # noqa: E501
    )
    ensemble = tmp_path / "ensemble.pdb"
    ensemble.write_text(
        "MODEL        1\n"
        + atom_lines.format(x=0, ca=1.45, c=1.9)
        + "ENDMDL\nMODEL        2\n"
        + atom_lines.format(x=10, ca=11.45, c=11.9)
        + "ENDMDL\nEND\n"
    )
    pipeline = InputDataPipeline(None, num_prior_samples=2)
    apos, priors, records = pipeline._load_monomer_sources(
        [1], "G", [str(ensemble)], [str(ensemble)], "ensemble"
    )
    assert len(apos) == len(priors) == len(records[0]) == 2
    assert apos[0][1][0, 0, 0] == pytest.approx(0)
    assert apos[1][1][0, 0, 0] == pytest.approx(10)


def test_release_protein_pair_example_preserves_assembly_group(tmp_path):
    import argparse
    import copy
    import warnings
    from pathlib import Path

    import yaml

    from kfold.data.types.ccd import CCD
    from kfold.inference.query import ProteinPair
    from kfold.inference.query import Query as NativeQuery
    from kfold.inference.sequential_query import parse_single_file

    root = Path(__file__).resolve().parents[2]
    data = yaml.safe_load((root / "examples/8jeo_sequential.yaml").read_text())
    automatic = copy.deepcopy(data)
    apo = tmp_path / "apo.pdb"
    apo.write_text("END\n")  # This test validates parsing/planning, not coordinates.
    for entry in data["sequences"]:
        next(iter(entry.values()))["apo"] = [str(apo)]
    path = tmp_path / "8jeo.yaml"
    path.write_text(yaml.safe_dump(data))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = parse_single_file(path, CCD({}))
    assert validate_plan(parsed) == [
        {"id": "assemble_BC", "chains": ["B", "C"]},
        {"id": "final", "chains": ["A", "B", "C"]},
    ]
    intermediate = subset_query(parsed, ["B", "C"])
    assert not intermediate.sequences
    assert intermediate.multimer_sequences[0].ids == [("B", "C")]
    assert (
        intermediate.multimer_sequences[0].sequence1
        == data["sequences"][1]["protein_pair"]["sequence1"]
    )
    with pytest.raises(ValueError, match="splits an existing object"):
        validate_plan(
            parsed.copy(assembly={"stages": [{"id": "AB", "chains": ["A", "B"]}]})
        )
    # The release parser preserves the plan through standard apo preparation.
    native = NativeQuery.load(path)
    assert isinstance(native.sequences[1], ProteinPair)
    assert native.assembly == data["assembly"]
    saved = tmp_path / "prepared.json"
    native.save(saved)
    assert NativeQuery.load(saved).assembly == data["assembly"]
    native_intermediate = subset_query(native, ["B", "C"])
    assert isinstance(native_intermediate.sequences[0], ProteinPair)
    assert native_intermediate.sequences[0].id == [["B", "C"]]

    # The standard CLI accepts the same assembly plan without explicit apo paths.
    from kfold.cli.predict import _load_queries, add_arguments

    automatic_path = tmp_path / "automatic.yaml"
    automatic_path.write_text(yaml.safe_dump(automatic))
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["-i", str(automatic_path), "-o", str(tmp_path / "out")])
    loaded = _load_queries(args)
    assert loaded[0].assembly == automatic["assembly"]
    assert all(entry.apo is None for entry in loaded[0].protein_entries)
    multichain_args = parser.parse_args(
        [
            "-i",
            str(automatic_path),
            "-o",
            str(tmp_path / "joint-out"),
            "--conditioning",
            "prior_and_trunk_multichain",
        ]
    )
    assert _load_queries(multichain_args)[0].assembly == automatic["assembly"]
    multichain_args.disable_struct_encoder = True
    with pytest.raises(ValueError, match="requires the protein structure encoder"):
        _load_queries(multichain_args)


def test_ranked_apo_ensemble_identity_and_shortage(tmp_path):
    keys = [("H", 1, "CA"), ("L", 1, "CA")]
    candidates = []
    for sample, score in enumerate([0.5, 0.9, 0.9]):
        coords = np.array([[sample, 0, 0], [sample, 2, 0]], dtype=np.float32)
        path = tmp_path / f"{sample}.npz"
        # Deliberately different atom order for the highest ranked structure.
        PriorObject(
            keys[::-1] if sample == 1 else keys, coords[::-1] if sample == 1 else coords
        ).save(path)
        candidates.append(
            dict(seed=1, sample=sample, ranking_score=score, atoms=str(path))
        )
    obj, selected = select_apo_ensemble(candidates, 2)
    assert [c["sample"] for c in selected] == [1, 2]
    assert obj.keys == keys[::-1]
    np.testing.assert_array_equal(obj.apo_coordinates[:, :, 0], [[1, 1], [2, 2]])
    np.testing.assert_array_equal(obj.apo_coordinates[:, :, 1], [[2, 0], [2, 0]])
    with pytest.raises(ValueError, match="needs 4"):
        select_apo_ensemble(candidates, 4)
    with pytest.raises(ValueError, match="Duplicate"):
        select_apo_ensemble([candidates[0], candidates[0]], 2)
