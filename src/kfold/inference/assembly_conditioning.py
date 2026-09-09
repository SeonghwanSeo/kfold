"""Re-encode predicted objects as structural inputs, without changing topology.

One selected object is broadcast over the apo axis. Unassembled chains keep
their original apo ensemble. Protein tokens are recomputed on the model device;
ligands update only their residue-local reference conformers. No P-L apo pair
geometry, shared apo UID, or cross-molecule reference space is introduced.
"""

import numpy as np

from .assembly import atom_keys


def apply_trunk_groups(struct, tokenized, records, groups, rng):
    from kfold.data.pipelines.tokenization import refresh_apo_geometry
    from kfold.utils.geometry.random_augment import center_random_augmentation

    keys = atom_keys(struct)
    index = {k: i for i, k in enumerate(keys)}
    metadata = {c.name: c.asym_id for c in struct.metadata.chains}
    chains = {c.asym_id: c for c in struct.chains}
    replacement = {}
    used = set()
    # Validate the entire update before mutating any input.
    for group in groups:
        if group.chains & used:
            raise ValueError("Overlapping trunk objects")
        if set(group.keys) != {k for k in keys if k[0] in group.chains}:
            raise ValueError("Trunk object atom mapping is incomplete or mismatched")
        used.update(group.chains)
        replacement.update(zip(group.keys, group.coordinates, strict=True))
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
            targets = [t for t in record["targets"] if t[0] not in affected]
            if targets:
                kept.append({**record, "targets": targets})
        if kept:
            updated.append(kept)
    num_apo = tokenized.atom.apo_coords.shape[-2]
    for name in sorted(used):
        chain = chains[metadata[name]]
        if chain.is_protein:
            chain_keys = [k for k in keys if k[0] == name]
            coords = chain.map_atom_coords_to_residue_coords(
                np.stack([replacement[k] for k in chain_keys])
            )
            record = {
                "seq": chain.get_sequence(map_to_standard=True),
                "coords": coords,
                "targets": [(chain.asym_id, 0, chain.num_residues)],
            }
            updated.append([record.copy() for _ in range(num_apo)])
    slots = np.argwhere(tokenized.atom.pad_mask)
    if len(slots) != len(keys):
        raise ValueError("Token/atom identity count differs")
    for name in sorted(used):
        chain = chains[metadata[name]]
        chain_keys = [k for k in keys if k[0] == name]
        if chain.is_protein:
            for key in chain_keys:
                ti, ai = slots[index[key]]
                tokenized.atom.apo_coords[ti, ai] = replacement[key]
        else:
            # Match the native residue-local reference-space contract, including
            # independent centering/rotation: do not leak the P-L rigid pose.
            for ri in range(1, chain.num_residues + 1):
                residue_keys = [k for k in chain_keys if k[1] == ri]
                xyz = np.stack([replacement[k] for k in residue_keys])
                xyz = center_random_augmentation(
                    xyz, np.ones(len(xyz), dtype=bool), rng=rng
                )
                residue_slots = np.array([slots[index[k]] for k in residue_keys])
                ti, ai = residue_slots.T
                tokenized.atom.ref_pos[ti, ai] = xyz
                tokenized.atom.ref_mask[ti, ai] = True
                _refresh_ligand_frames(tokenized, residue_slots, xyz)
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
