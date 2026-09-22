"""Identity-preserving assembly plans and lossless rigid prior objects.

This module does not alter molecular topology or model chain/interface IDs.
"""

import dataclasses
import json
import re

import numpy as np


def _is_pair(sequence):
    """Return whether a native or experimental entry is a protein pair."""
    return hasattr(sequence, "sequence1") and hasattr(sequence, "sequence2")


def _ids(sequence):
    """Return normalized IDs from native and experimental sequence objects."""
    return getattr(sequence, "ids", sequence.id)


def chain_names(query):
    """Physical chain names for native and experimental query objects."""
    names = []
    for sequence in query.sequences:
        if _is_pair(sequence):
            names.extend(chain for pair in _ids(sequence) for chain in pair)
        else:
            names.extend(_ids(sequence))
    for sequence in getattr(query, "multimer_sequences", []):
        names.extend(chain for pair in _ids(sequence) for chain in pair)
    return names


def pair_groups(query):
    """Pre-existing two-chain objects that an assembly stage cannot split."""
    groups = [
        set(pair)
        for sequence in query.sequences
        if _is_pair(sequence)
        for pair in _ids(sequence)
    ]
    groups.extend(
        set(pair)
        for sequence in getattr(query, "multimer_sequences", [])
        for pair in _ids(sequence)
    )
    return groups


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
    groups = pair_groups(query)
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
    sequences = []
    for sequence in query.sequences:
        if _is_pair(sequence):
            if any(
                bool(chosen.intersection(pair)) and not set(pair) <= chosen
                for pair in _ids(sequence)
            ):
                raise ValueError("Cannot split a protein pair")
            pairs = [pair for pair in _ids(sequence) if set(pair) <= chosen]
            if pairs:
                sequences.append(dataclasses.replace(sequence, id=pairs))
        else:
            ids = [chain for chain in _ids(sequence) if chain in chosen]
            if ids:
                sequences.append(dataclasses.replace(sequence, id=ids))
    multimers = []
    for seq in getattr(query, "multimer_sequences", []):
        if any(bool(chosen.intersection(p)) and not set(p) <= chosen for p in _ids(seq)):
            raise ValueError("Cannot split a multimer")
        pairs = [p for p in _ids(seq) if set(p) <= chosen]
        if pairs:
            multimers.append(dataclasses.replace(seq, id=pairs))
    bonds = [b for b in query.bonds if b.atom1[0] in chosen and b.atom2[0] in chosen]
    updates = {"sequences": sequences, "bonds": bonds, "assembly": None}
    if hasattr(query, "multimer_sequences"):
        updates["multimer_sequences"] = multimers
    if hasattr(query, "copy"):
        return query.copy(**updates)
    return dataclasses.replace(query, **updates)


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
    observed_mask: np.ndarray | None = None
    apo_coordinates: np.ndarray | None = None

    def __post_init__(self):
        self.keys = [tuple(k) for k in self.keys]
        self.coordinates = np.asarray(self.coordinates, dtype=np.float32)
        explicit_mask = self.observed_mask is not None
        self.observed_mask = (
            np.asarray(self.observed_mask, dtype=bool)
            if explicit_mask
            else np.ones(len(self.keys), dtype=bool)
        )
        if (
            not self.keys
            or len(set(self.keys)) != len(self.keys)
            or self.coordinates.shape != (len(self.keys), 3)
            or self.observed_mask.shape != (len(self.keys),)
            or not self.observed_mask.any()
            or not np.isfinite(self.coordinates[self.observed_mask]).all()
        ):
            raise ValueError(
                "Prior object requires unique atoms and finite complete coordinates"
            )
        self.coordinates = self.coordinates.copy()
        self.coordinates[~self.observed_mask] = np.nan

        if self.apo_coordinates is not None:
            self.apo_coordinates = np.asarray(
                self.apo_coordinates, dtype=np.float32
            ).copy()
            if (
                self.apo_coordinates.ndim != 3
                or self.apo_coordinates.shape[1:] != self.coordinates.shape
                or len(self.apo_coordinates) == 0
                or not np.isfinite(self.apo_coordinates[:, self.observed_mask]).all()
            ):
                raise ValueError("Invalid intermediate apo ensemble")
            self.apo_coordinates[:, ~self.observed_mask] = np.nan

    @property
    def chains(self):
        return {k[0] for k in self.keys}

    def save(self, path):
        np.savez_compressed(
            path,
            keys=json.dumps(self.keys),
            coordinates=self.coordinates,
            observed_mask=self.observed_mask,
            **(
                {"apo_coordinates": self.apo_coordinates}
                if self.apo_coordinates is not None
                else {}
            ),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                json.loads(str(data["keys"])),
                data["coordinates"].copy(),
                data["observed_mask"].copy() if "observed_mask" in data else None,
                data["apo_coordinates"].copy() if "apo_coordinates" in data else None,
            )


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
            coords = group.coordinates
            if not group.observed_mask.all():
                # Native missing-atom prior initialization only; never pass its
                # synthetic coordinates to the trunk as observed structure.
                coords = coords.copy()
                coords -= coords[group.observed_mask].mean(axis=0)
                by_name = {c.name: c.asym_id for c in struct.metadata.chains}
                by_id = {c.asym_id: c for c in struct.chains}
                for name in sorted(group.chains):
                    rows = [i for i, k in enumerate(group.keys) if k[0] == name]
                    chain = by_id[by_name[name]]
                    canonical = [k for k in keys if k[0] == name]
                    row_by_key = {group.keys[i]: i for i in rows}
                    ordered = [row_by_key[k] for k in canonical]
                    if not np.isfinite(coords[ordered]).all():
                        if not chain.is_protein:
                            raise ValueError("Missing ligand coordinates are unsupported")
                        before = coords[ordered].copy()
                        resolved = np.isfinite(before).all(axis=-1)
                        relaxed = sampler.langevin_relaxation(before, chain, rng)
                        relaxed[resolved] = before[resolved]
                        if not np.isfinite(relaxed).all():
                            raise ValueError("Missing-atom prior initialization failed")
                        coords[ordered] = relaxed
            sample[indices] = sampler.apply_random_augmentation(coords, rng)


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


def select_apo_ensemble(candidates, count):
    """Select distinct predicted samples, keeping each complex in its own frame."""
    select_top1(candidates)  # Validate every score with the same ranking policy.
    if count < 1 or len(candidates) < count:
        raise ValueError(
            f"Intermediate apo ensemble needs {count} predictions; "
            f"got {len(candidates)}. "
            "Increase seeds/samples or reduce --num-apos."
        )
    if len({(c["seed"], c["sample"]) for c in candidates}) != len(candidates):
        raise ValueError("Duplicate intermediate candidates")
    selected = sorted(
        candidates, key=lambda c: (-c["ranking_score"], c["seed"], c["sample"])
    )[:count]
    objects = [PriorObject.load(c["atoms"]) for c in selected]
    first = objects[0]
    coordinates = []
    for obj in objects:
        if set(obj.keys) != set(first.keys) or not obj.observed_mask.all():
            raise ValueError(
                "Intermediate ensemble atom mapping is incomplete or mismatched"
            )
        index = {k: i for i, k in enumerate(obj.keys)}
        coordinates.append(obj.coordinates[[index[k] for k in first.keys]])
    return PriorObject(
        first.keys, first.coordinates, apo_coordinates=np.stack(coordinates)
    ), selected


def select_seed_apo_ensemble(candidates, generation_seeds, prior):
    """Keep one winner per generation seed, in slot order, with a separate prior.

    The apo policy matches AtlasFold-M's per-generation-seed selection. ECSI
    retains its existing global top-1 policy, independent of apo slot zero.
    """
    if not generation_seeds or len(set(generation_seeds)) != len(generation_seeds):
        raise ValueError("Expected distinct apo generation seeds")
    if {c["seed"] for c in candidates} != set(generation_seeds):
        raise ValueError("Apo candidates do not match the generation seeds")
    if len({(c["seed"], c["sample"]) for c in candidates}) != len(candidates):
        raise ValueError("Duplicate intermediate candidates")
    selected = [
        select_top1([c for c in candidates if c["seed"] == seed])
        for seed in generation_seeds
    ]
    coordinates = []
    for candidate in selected:
        obj = PriorObject.load(candidate["atoms"])
        if set(obj.keys) != set(prior.keys) or not obj.observed_mask.all():
            raise ValueError(
                "Intermediate ensemble atom mapping is incomplete or mismatched"
            )
        index = {k: i for i, k in enumerate(obj.keys)}
        coordinates.append(obj.coordinates[[index[k] for k in prior.keys]])
    return PriorObject(
        prior.keys,
        prior.coordinates,
        prior.observed_mask,
        apo_coordinates=np.stack(coordinates),
    ), selected
