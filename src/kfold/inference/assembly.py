"""Identity-preserving assembly plans and lossless rigid prior objects.

This module does not alter molecular topology or model chain/interface IDs.
"""

import dataclasses
import json
import re

import numpy as np


def chain_names(query):
    return [c for s in query.sequences for c in s.ids] + [
        c for s in query.multimer_sequences for pair in s.ids for c in pair
    ]


def validate_plan(query):
    assembly = query.assembly
    if not isinstance(assembly, dict) or set(assembly) - {"stages", "selection"}:
        raise ValueError("assembly must contain stages and optional selection")
    if assembly.get("selection", "confidence_top1") != "confidence_top1":
        raise ValueError("Only confidence_top1 selection is supported")
    stages = assembly.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("assembly.stages must be a nonempty list")
    names = set(chain_names(query))
    groups = [set(pair) for s in query.multimer_sequences for pair in s.ids]
    ids = {"final"}
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {"id", "chains"}:
            raise ValueError("Each stage requires exactly id and chains")
        sid = stage["id"]
        if (
            not isinstance(sid, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sid)
            or sid in ids
        ):
            raise ValueError(f"Invalid or duplicate stage id: {sid}")
        ids.add(sid)
        chains = stage["chains"]
        if not isinstance(chains, list) or not all(isinstance(c, str) for c in chains):
            raise ValueError("stage chains must be a list of chain IDs")
        chosen = set(chains)
        if len(chosen) < 2 or len(chosen) != len(chains) or not chosen <= names:
            raise ValueError("stage requires >=2 unique, known chains")
        if chosen == names:
            raise ValueError("Full-system final stage is appended automatically")
        for group in groups:
            if chosen & group and not group <= chosen:
                raise ValueError(
                    f"Stage {sid} splits an existing object: {sorted(group)}"
                )
        for bond in query.bonds:
            if (bond.atom1[0] in chosen) != (bond.atom2[0] in chosen):
                raise ValueError(f"Stage {sid} cuts a covalent bond")
        groups = [g for g in groups if not g <= chosen] + [chosen]
    return stages + [{"id": "final", "chains": chain_names(query)}]


def subset_query(query, names):
    chosen = set(names)
    sequences = [
        dataclasses.replace(s, id=[c for c in s.ids if c in chosen])
        for s in query.sequences
        if chosen.intersection(s.ids)
    ]
    multimers = []
    for seq in query.multimer_sequences:
        if any(bool(chosen.intersection(p)) and not set(p) <= chosen for p in seq.ids):
            raise ValueError("Cannot split a multimer")
        pairs = [p for p in seq.ids if set(p) <= chosen]
        if pairs:
            multimers.append(dataclasses.replace(seq, id=pairs))
    bonds = [b for b in query.bonds if b.atom1[0] in chosen and b.atom2[0] in chosen]
    return query.copy(
        sequences=sequences, multimer_sequences=multimers, bonds=bonds, assembly=None
    )


def atom_keys(struct):
    """Stable identity in RefStructure atom order (not stage-local asym ID)."""
    metadata = {c.asym_id: c for c in struct.metadata.chains}
    keys = []
    for chain in struct.chains:
        name = metadata[chain.asym_id].name
        for ri in range(chain.num_residues):
            for ai in chain.residue.iter_residue_atoms(ri + 1):
                keys.append((name, ri + 1, str(chain.atom.name[ai])))
    if len(keys) != struct.num_atoms or len(set(keys)) != len(keys):
        raise ValueError("Invalid/duplicate atom identity")
    return keys


@dataclasses.dataclass
class PriorObject:
    keys: list
    coordinates: np.ndarray

    def __post_init__(self):
        self.keys = [tuple(k) for k in self.keys]
        self.coordinates = np.asarray(self.coordinates, dtype=np.float32)
        if (
            not self.keys
            or len(set(self.keys)) != len(self.keys)
            or self.coordinates.shape != (len(self.keys), 3)
            or not np.isfinite(self.coordinates).all()
        ):
            raise ValueError(
                "Prior object requires unique atoms and finite complete coordinates"
            )

    @property
    def chains(self):
        return {k[0] for k in self.keys}

    def save(self, path):
        np.savez_compressed(
            path, keys=json.dumps(self.keys), coordinates=self.coordinates
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(json.loads(str(data["keys"])), data["coordinates"].copy())


def apply_prior_groups(struct, priors, groups, sampler, rng):
    keys = atom_keys(struct)
    index = {key: i for i, key in enumerate(keys)}
    used = set()
    for group in groups:
        if group.chains & used:
            raise ValueError("Overlapping prior objects")
        used.update(group.chains)
        expected = {k for k in keys if k[0] in group.chains}
        if set(group.keys) != expected:
            raise ValueError("Prior object atom mapping is incomplete or mismatched")
        indices = [index[k] for k in group.keys]
        for sample in priors:
            sample[indices] = sampler.apply_random_augmentation(group.coordinates, rng)


def subset_sources(full_struct, sub_struct, sources):
    """Keep the original full-query apo/prior choices, remapping local IDs."""
    sub_ids = {c.name: c.asym_id for c in sub_struct.metadata.chains}
    mapping = {
        c.asym_id: sub_ids[c.name]
        for c in full_struct.metadata.chains
        if c.name in sub_ids
    }

    def remap(coords):
        return {mapping[k]: v for k, v in coords.items() if k in mapping}

    records = []
    for record_group in sources.struct_token_records:
        filtered = []
        for record in record_group:
            targets = [
                (mapping[a], start, end)
                for a, start, end in record["targets"]
                if a in mapping
            ]
            if targets:
                filtered.append({**record, "targets": targets})
        if filtered:
            records.append(filtered)
    return dataclasses.replace(
        sources,
        apo_coords=remap(sources.apo_coords),
        prior_sources=[remap(p) for p in sources.prior_sources],
        struct_token_records=records,
    )


def select_top1(candidates):
    """No ground truth; tie break by ascending seed then sample."""
    if not candidates or any(not np.isfinite(c["ranking_score"]) for c in candidates):
        raise ValueError("Missing or nonfinite candidate ranking score")
    return min(candidates, key=lambda c: (-c["ranking_score"], c["seed"], c["sample"]))
