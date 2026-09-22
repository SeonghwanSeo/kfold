"""Re-encode predicted objects as structural inputs, without changing topology.

Ranked intermediate samples fill the apo axis with paired protein coordinates.
Proteins in each assembled object share an apo UID. Unassembled chains keep
their original apo ensemble. Protein tokens are recomputed on the model device;
ligands update only their residue-local reference conformers. No P-L apo pair
geometry or cross-molecule reference space is introduced.
"""

import numpy as np

from .assembly import atom_keys

_CHAIN_RESIDUE_INDEX_GAP = 512


def apply_trunk_groups(
    struct,
    tokenized,
    records,
    groups,
    rng,
    *,
    multichain_structure=False,
):
    from kfold.data.pipelines.tokenization import refresh_apo_geometry
    from kfold.utils.geometry.random_augment import center_random_augmentation

    keys = atom_keys(struct)
    index = {k: i for i, k in enumerate(keys)}
    metadata = {c.name: c.asym_id for c in struct.metadata.chains}
    chains = {c.asym_id: c for c in struct.chains}
    replacement = {}
    apo_replacement = {}
    num_apo = tokenized.atom.apo_coords.shape[-2]
    used = set()
    # Validate the entire update before mutating any input.
    for group in groups:
        if group.chains & used:
            raise ValueError("Overlapping trunk objects")
        if set(group.keys) != {k for k in keys if k[0] in group.chains}:
            raise ValueError("Trunk object atom mapping is incomplete or mismatched")
        used.update(group.chains)
        replacement.update(zip(group.keys, group.coordinates, strict=True))
        ensemble = group.apo_coordinates
        if ensemble is None:
            # Explicit single structures (e.g. GT) remain valid inputs.
            ensemble = np.repeat(group.coordinates[None], num_apo, axis=0)
        if len(ensemble) < num_apo:
            raise ValueError("Intermediate apo ensemble has fewer samples than apo slots")
        apo_replacement.update(
            zip(group.keys, ensemble[:num_apo].transpose(1, 0, 2), strict=True)
        )
    if not groups:
        return records
    affected = {metadata[n] for n in used}
    if any(not (chains[a].is_protein or chains[a].is_ligand) for a in affected):
        raise ValueError("Trunk re-encoding currently supports protein and ligand only")
    # Split shared records: a physical copy not in this object must stay apo.
    updated = []
    for record_group in records:
        kept = []
        for record in record_group:
            keep_indices = [
                i
                for i, target in enumerate(record["targets"])
                if target[0] not in affected
            ]
            targets = [record["targets"][i] for i in keep_indices]
            if targets:
                retained = {**record, "targets": targets}
                if "segments" in retained:
                    retained["segments"] = [retained["segments"][i] for i in keep_indices]
                kept.append(retained)
        if kept:
            updated.append(kept)

    def protein_component(name, slot):
        chain = chains[metadata[name]]
        chain_keys = [k for k in keys if k[0] == name]
        coords = chain.map_atom_coords_to_residue_coords(
            np.stack([apo_replacement[k][slot] for k in chain_keys])
        )
        return chain, coords

    if multichain_structure:
        structure_order = [chain.name for chain in struct.metadata.chains]
        for group in groups:
            protein_names = [
                name
                for name in structure_order
                if name in group.chains and chains[metadata[name]].is_protein
            ]
            if not protein_names:
                continue
            ensemble_records = []
            for slot in range(num_apo):
                components = [protein_component(name, slot) for name in protein_names]
                sequences = [
                    chain.get_sequence(map_to_standard=True) for chain, _ in components
                ]
                coordinates = [coords for _, coords in components]
                source_start = 0
                residue_offset = 0
                segments = []
                residue_indices = []
                for (chain, _coords), sequence in zip(components, sequences, strict=True):
                    source_end = source_start + len(sequence)
                    segments.append(
                        (chain.asym_id, 0, chain.num_residues, source_start, source_end)
                    )
                    residue_indices.append(
                        np.arange(1, len(sequence) + 1, dtype=np.int64) + residue_offset
                    )
                    source_start = source_end
                    residue_offset += len(sequence) + _CHAIN_RESIDUE_INDEX_GAP
                record = {
                    "seq": "".join(sequences),
                    "coords": np.concatenate(coordinates, axis=0),
                    "targets": [segment[:3] for segment in segments],
                    "segments": segments,
                    "residue_index": np.concatenate(residue_indices),
                    "structure_group": [chain.asym_id for chain, _ in components],
                    "mask_missing_structure": True,
                }
                ensemble_records.append(record)
            updated.append(ensemble_records)
    else:
        for name in sorted(used):
            chain = chains[metadata[name]]
            if chain.is_protein:
                ensemble_records = []
                for slot in range(num_apo):
                    _, coords = protein_component(name, slot)
                    record = {
                        "seq": chain.get_sequence(map_to_standard=True),
                        "coords": coords,
                        "targets": [(chain.asym_id, 0, chain.num_residues)],
                        "mask_missing_structure": True,
                    }
                    ensemble_records.append(record)
                updated.append(ensemble_records)
    slots = np.argwhere(tokenized.atom.pad_mask)
    if len(slots) != len(keys):
        raise ValueError("Token/atom identity count differs")
    for name in sorted(used):
        chain = chains[metadata[name]]
        chain_keys = [k for k in keys if k[0] == name]
        if chain.is_protein:
            for key in chain_keys:
                ti, ai = slots[index[key]]
                tokenized.atom.apo_coords[ti, ai] = apo_replacement[key]
        else:
            # Match the native residue-local reference-space contract, including
            # independent centering/rotation: do not leak the P-L rigid pose.
            for ri in range(1, chain.num_residues + 1):
                residue_keys = [k for k in chain_keys if k[1] == ri]
                xyz = np.stack([replacement[k] for k in residue_keys])
                if not np.isfinite(xyz).all():
                    raise ValueError("Missing ligand coordinates are unsupported")
                xyz = center_random_augmentation(
                    xyz, np.ones(len(xyz), dtype=bool), rng=rng
                )
                residue_slots = np.array([slots[index[k]] for k in residue_keys])
                ti, ai = residue_slots.T
                tokenized.atom.ref_pos[ti, ai] = xyz
                tokenized.atom.ref_mask[ti, ai] = True
                _refresh_ligand_frames(tokenized, residue_slots, xyz)
    # Use fresh IDs to avoid merging an unrelated original apo group. Ligands
    # keep their original IDs and never participate in protein apo geometry.
    next_uid = int(tokenized.chain.apo_uid.max()) + 1
    for group in groups:
        protein_ids = [
            metadata[n] for n in group.chains if chains[metadata[n]].is_protein
        ]
        if len(protein_ids) < 2:
            continue
        tokenized.chain.apo_uid[np.isin(tokenized.chain.asym_id, protein_ids)] = next_uid
        tokenized.token.apo_uid[np.isin(tokenized.token.asym_id, protein_ids)] = next_uid
        next_uid += 1
    refresh_apo_geometry(tokenized)
    tokenized.validate()
    return updated


def _refresh_ligand_frames(tok, slots, xyz):
    """Native nearest-neighbor/25-degree frame rule on the new conformer."""
    from kfold.data.pipelines.tokenization import COLLISION_ANGLE_CUTOFF

    for i, (ti, _) in enumerate(slots):
        tok.token.frame_token_index[ti] = -1
        tok.token.frame_atom_index[ti] = -1
        if len(xyz) < 3:
            continue
        distances = np.linalg.norm(xyz - xyz[i], axis=-1)
        distances[i] = np.inf
        a, c = np.argsort(distances)[:2]
        v, w = xyz[a] - xyz[i], xyz[c] - xyz[i]
        denom = np.linalg.norm(v) * np.linalg.norm(w)
        if denom < 1e-8 or abs(np.dot(v, w) / denom) > COLLISION_ANGLE_CUTOFF:
            continue
        tok.token.frame_token_index[ti] = slots[[a, i, c], 0]
        tok.token.frame_atom_index[ti] = slots[[a, i, c], 1]
