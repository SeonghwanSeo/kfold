import msgpack
import numpy as np

import kfold.constants as C
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import BondLayout, Chain, CovalentConnection, RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.training.dataset.utils.apo_io import (
    unpack_apo_multimer_record,
    unpack_apo_multimer_token_record,
    unpack_prior_multimer_stack_record,
)

from .base import _open_lmdb
from .train_dataset import TrainingDataset

StructInfo = dict
MAX_BOND_LENGTH = 2.4


class RCSBTrainingDataset(TrainingDataset):
    """Training dataset for RCSB structures with experimental metadata."""

    def setup(self) -> None:
        """Load multimer apo/prior lookup products."""
        self.prob_use_complex_apo = self.config.prob_use_complex_apo
        self.prob_use_complex_prior = self.config.prob_use_complex_prior
        self.apo_multimer_lookup_table = self.load_apo_multimer_lookup_table()

    def load_apo_multimer_lookup_table(self) -> dict:
        lookup_path = self.data_root / "apo_multimer_lookup.msgpack"
        if not lookup_path.exists():
            return {}
        with open(lookup_path, "rb") as f:
            return msgpack.unpack(f, raw=False)

    def _get_apo_multimer_source_lmdb_env(self, chain_type: str, source: str):
        return self._get_source_lmdb_env(
            "_apo_multimer_source_lmdb_envs", "apo_multimer_lmdb", chain_type, source
        )

    def _get_multimer_prior_lmdb_env(self):
        if not hasattr(self, "_protein_multimer_prior_lmdb_env"):
            self._protein_multimer_prior_lmdb_env = _open_lmdb(
                self.data_root / "prior_multimer_lmdb" / "protein.lmdb"
            )
        return self._protein_multimer_prior_lmdb_env

    def __del__(self):
        if hasattr(self, "_apo_multimer_source_lmdb_envs"):
            for env in self._apo_multimer_source_lmdb_envs.values():
                env.close()
        if hasattr(self, "_protein_multimer_prior_lmdb_env"):
            self._protein_multimer_prior_lmdb_env.close()
        super().__del__()

    def load_ref_structure(self, metadata: Metadata) -> RefStructure:
        """Get the structure for the given index."""
        ref_struct = super().load_ref_structure(metadata)
        # Clean up the structure (e.g., filter out unrealistic bonds)
        ref_struct = clean_up_ref_structure(ref_struct)
        return ref_struct

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        cfg = self.config
        # Check if perturbation is enabled for training set, and validate files.
        if cfg.apo_perturb is None:
            self.logger.warning("Protein perturbation is disabled.")
        else:
            if cfg.apo_perturb.rieprody is None:
                self.logger.info("RieProDy perturbation is disabled.")
            else:
                rieprody_lmdb_path = self.data_root / "rieprody_metric.lmdb"
                if not rieprody_lmdb_path.exists():
                    raise FileNotFoundError(
                        f"RieProDy LMDB path {rieprody_lmdb_path} not found "
                        f"while rieprody is enabled."
                    )
                # If rieprody perturbation is enabled, we need to provide the LMDB path
                cfg.apo_perturb.rieprody.metric_lmdb_path = rieprody_lmdb_path

    def _load_apo_multimer_info_from_lmdb(
        self,
        apo_info: dict,
        *,
        context: str,
    ) -> dict:
        """Attach multimer chain payloads from source-specific apo multimer LMDB."""
        loaded = apo_info.copy()
        source = loaded["source"]
        chain_type = loaded["chain_type"]
        lmdb_key = loaded["name"]

        env = self._get_apo_multimer_source_lmdb_env(chain_type, source)
        with env.begin(write=False) as txn:
            value_bytes = txn.get(lmdb_key.encode("utf-8"))
        if value_bytes is None:
            raise KeyError(f"Apo multimer '{source}:{lmdb_key}' not found for {context}")

        loaded["key"] = f"{source}:{lmdb_key}"
        loaded["lmdb_key"] = lmdb_key
        loaded["chains"] = unpack_apo_multimer_record(value_bytes)
        return loaded

    def _load_multimer_prior_stack_info_from_lmdb(
        self,
        group: dict,
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray] | None:
        """Load and sample one protein multimer prior stack record."""
        assert group["chain_type"] == "protein"
        name = group["name"]
        prior_lmdb_path = self.data_root / "prior_multimer_lmdb" / "protein.lmdb"
        if not prior_lmdb_path.exists():
            return None

        env = self._get_multimer_prior_lmdb_env()
        with env.begin(write=False) as txn:
            value_bytes = txn.get(name.encode("utf-8"))
        if value_bytes is None:
            return None

        record = unpack_prior_multimer_stack_record(value_bytes)
        first_chain = next(iter(record["chains"].values()))
        num_samples = int(first_chain["coords"].shape[0])
        sample_i = int(rng.integers(0, num_samples))

        out: dict[int, np.ndarray] = {}
        for asym_id, chain in record["chains"].items():
            coords = chain["coords"]
            if coords.ndim != 4:
                raise ValueError(
                    f"Prior multimer stack {name}/{asym_id} has shape "
                    f"{coords.shape}; expected (N, L, A, 3)."
                )
            if coords.shape[0] != num_samples:
                raise ValueError(
                    f"Prior multimer stack {name} has inconsistent sample counts."
                )
            out[int(asym_id)] = coords[sample_i].copy()
        return out

    def get_apo_lookup(
        self, ref_struct: RefStructure, rng: np.random.Generator
    ) -> dict[int, list[dict | None]]:
        """Get monomer apo lookup, then replace selected entities with multimer apo."""
        num_apo = self.sample_num_apo(rng)
        apo_lookup = self.get_monomer_apo_lookup(ref_struct, rng, num_apo)
        entry_id = ref_struct.id
        multimer_groups: list[dict] = self.apo_multimer_lookup_table.get(entry_id, [])
        if not multimer_groups or rng.random() >= self.prob_use_complex_apo:
            return apo_lookup

        metadata_by_asym_id = {c.asym_id: c for c in ref_struct.metadata.chains}
        protein_asym_ids = {c.asym_id for c in ref_struct.chains if c.ctype.is_protein}

        # Multiple records with the same identity are alternative sources for one
        # multimer. Select one identity first so a chain never mixes monomer and
        # multimer coordinates across apo slots.
        candidates_by_identity: dict[tuple[str, int, tuple[int, ...]], list[dict]] = {}
        for group in multimer_groups:
            identity = (
                group["name"],
                int(group["apo_uid"]),
                tuple(int(asym_id) for asym_id in group["asym_ids"]),
            )
            candidates_by_identity.setdefault(identity, []).append(group)

        valid_candidates: list[tuple[tuple[int, ...], int, list[dict]]] = []
        for (_, apo_uid, group_asym_ids), candidates in candidates_by_identity.items():
            active_asym_ids = [
                asym_id for asym_id in group_asym_ids if asym_id in protein_asym_ids
            ]
            if not active_asym_ids:
                continue

            if len(set(active_asym_ids)) != len(active_asym_ids):
                self.logger.warning(
                    f"Skipping multimer apo group {entry_id}:{apo_uid} because it "
                    f"contains duplicate active asym_ids: {active_asym_ids}."
                )
                continue

            active_entity_ids = [
                metadata_by_asym_id[asym_id].entity_id for asym_id in active_asym_ids
            ]
            if len(set(active_entity_ids)) != len(active_entity_ids):
                self.logger.warning(
                    f"Skipping multimer apo group {entry_id}:{apo_uid} because its "
                    f"active chains share an entity: asym_ids={active_asym_ids}, "
                    f"entity_ids={active_entity_ids}."
                )
                continue

            valid_candidates.append((tuple(active_asym_ids), apo_uid, candidates))

        if not valid_candidates:
            return apo_lookup

        candidate_i = (
            int(rng.integers(len(valid_candidates))) if len(valid_candidates) > 1 else 0
        )
        active_asym_ids, apo_uid, candidates = valid_candidates[candidate_i]
        num_multimer = min(num_apo, len(candidates))
        selected_source_indices = rng.choice(
            len(candidates), size=num_multimer, replace=False
        )

        active_entity_ids = {
            metadata_by_asym_id[asym_id].entity_id for asym_id in active_asym_ids
        }
        # Reuse the selected coordinates across entity copies while preserving
        # each physical multimer pair's rigid-group ID from the lookup.
        apo_uid_by_asym_id = {asym_id: apo_uid for asym_id in active_asym_ids}
        for candidate_asym_ids, candidate_apo_uid, _ in valid_candidates:
            candidate_entity_ids = {
                metadata_by_asym_id[asym_id].entity_id for asym_id in candidate_asym_ids
            }
            if candidate_entity_ids != active_entity_ids:
                continue
            # Keep the selected identity when an alternative record assigns a
            # different UID to any of the same physical chains.
            if any(
                asym_id in apo_uid_by_asym_id
                and apo_uid_by_asym_id[asym_id] != candidate_apo_uid
                for asym_id in candidate_asym_ids
            ):
                continue
            apo_uid_by_asym_id.update(
                dict.fromkeys(candidate_asym_ids, candidate_apo_uid)
            )

        selected_by_entity: dict[int, list[dict | None]] = {
            entity_id: [None] * self.max_apo for entity_id in active_entity_ids
        }

        for apo_i, source_i in enumerate(selected_source_indices):
            group = candidates[int(source_i)]
            context = f"multimer group {entry_id}:{apo_uid}"
            multimer_info = self._load_apo_multimer_info_from_lmdb(group, context=context)
            multimer_chains: dict[int, dict] = multimer_info["chains"]
            for asym_id in active_asym_ids:
                if asym_id not in multimer_chains:
                    raise KeyError(
                        f"Apo multimer {multimer_info['key']} for entry {entry_id} "
                        f"does not contain asym_id {asym_id}."
                    )
                entity_id = metadata_by_asym_id[asym_id].entity_id
                loaded = group.copy()
                loaded.update(multimer_chains[asym_id])
                loaded["key"] = multimer_info["key"]
                loaded["lmdb_key"] = multimer_info["lmdb_key"]
                loaded["multimer_key"] = multimer_info["key"]
                loaded["source_asym_id"] = asym_id
                loaded["apo_uid"] = apo_uid
                loaded["is_multimer_apo"] = True
                selected_by_entity[entity_id][apo_i] = loaded

        for chain in ref_struct.chains:
            if not chain.ctype.is_protein or chain.entity_id not in selected_by_entity:
                continue
            chain_metadata = metadata_by_asym_id[chain.asym_id]
            chain_apo_uid = apo_uid_by_asym_id.get(
                chain.asym_id, int(chain_metadata.apo_uid)
            )
            apo_lookup[chain.asym_id] = [
                None
                if loaded is None
                else {
                    **loaded,
                    "chain_key": f"{entry_id}_{chain.asym_id}",
                    "apo_uid": chain_apo_uid,
                }
                for loaded in selected_by_entity[chain.entity_id]
            ]
            chain_metadata.apo_uid = chain_apo_uid

        return apo_lookup

    def get_prior_coords(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray]:
        """Sample monomer priors with optional RCSB multimer prior overlay."""
        prior_coords = super().get_prior_coords(ref_struct, rng)
        entry_id = ref_struct.id
        multimer_groups: list[dict] = self.apo_multimer_lookup_table.get(entry_id, [])
        if not multimer_groups or rng.random() >= self.prob_use_complex_prior:
            return prior_coords

        selected_multimer_by_name: dict[str, dict[int, np.ndarray] | None] = {}
        metadata_by_asym_id = {c.asym_id: c for c in ref_struct.metadata.chains}
        protein_asym_ids = {c.asym_id for c in ref_struct.chains if c.ctype.is_protein}

        for group in multimer_groups:
            name = group["name"]
            if name in selected_multimer_by_name:
                continue
            prior_uid = int(group["apo_uid"])
            group_asym_ids = [int(aid) for aid in group["asym_ids"]]
            active_asym_ids = [
                asym_id for asym_id in group_asym_ids if asym_id in protein_asym_ids
            ]
            if not active_asym_ids:
                continue

            selected_multimer_by_name[name] = (
                self._load_multimer_prior_stack_info_from_lmdb(group, rng)
            )
            multimer_prior = selected_multimer_by_name[name]
            if multimer_prior is None:
                continue

            for asym_id in active_asym_ids:
                if asym_id not in multimer_prior:
                    raise KeyError(
                        f"Prior multimer {name} for "
                        f"entry {entry_id} does not contain asym_id {asym_id}."
                    )
                prior_coords[asym_id] = multimer_prior[asym_id]
                if asym_id in metadata_by_asym_id:
                    metadata_by_asym_id[asym_id].prior_uid = prior_uid

        return prior_coords

    def populate_structure_tokens(
        self,
        tokenized: TokenizedStructure,
        apo_lookup: dict[int, list[dict | None]],
    ) -> None:
        """Populate monomer and RCSB multimer apo structure tokens."""
        monomer_apo_lookup = {
            asym_id: [
                None
                if apo_info is not None and apo_info.get("is_multimer_apo", False)
                else apo_info
                for apo_info in apo_infos
            ]
            for asym_id, apo_infos in apo_lookup.items()
        }
        super().populate_structure_tokens(tokenized, monomer_apo_lookup)

        multimer_token_cache: dict[tuple[str, str], dict[int, np.ndarray]] = {}
        missing_token_sources: set[str] = set()
        missing_token_records: set[tuple[str, str]] = set()
        for c_i in range(tokenized.num_chains):
            if tokenized.chain.chain_type[c_i] != C.ChainType.PROTEIN.value:
                continue

            asym_id = int(tokenized.chain.asym_id[c_i])
            ek = f"{tokenized.id}:{asym_id}"
            for apo_i, apo_info in enumerate(apo_lookup.get(asym_id, [])):
                if apo_info is None or not apo_info.get("is_multimer_apo", False):
                    continue

                source = apo_info["source"]
                key = apo_info["name"]
                cache_key = (source, key)
                if cache_key not in multimer_token_cache:
                    if (
                        source in missing_token_sources
                        or cache_key in missing_token_records
                    ):
                        continue
                    token_lmdb_path = (
                        self.data_root
                        / "apo_tok_lmdb"
                        / "protein_multimer"
                        / f"{source}.lmdb"
                    )
                    if not token_lmdb_path.exists():
                        self.logger.warning(
                            f"Apo multimer structure-token LMDB for {source} "
                            "does not exist. Skipping structure tokens for this source."
                        )
                        missing_token_sources.add(source)
                        continue
                    env = self._get_apo_tok_source_lmdb_env("protein_multimer", source)
                    with env.begin(write=False) as txn:
                        value = txn.get(key.encode("utf-8"))
                    if value is None:
                        self.logger.warning(
                            f"Apo multimer structure tokens {source}:{key} not found "
                            f"in LMDB for chain `{ek}`. Skipping this entry"
                        )
                        missing_token_records.add(cache_key)
                        continue
                    multimer_token_cache[cache_key] = unpack_apo_multimer_token_record(
                        value
                    )
                source_asym_id = int(apo_info["source_asym_id"])
                if source_asym_id not in multimer_token_cache[cache_key]:
                    self.logger.warning(
                        f"Apo multimer structure tokens {source}:{key} do not contain "
                        f"asym_id {source_asym_id}. Skipping chain `{ek}`."
                    )
                    continue
                self._insert_structure_tokens(
                    tokenized,
                    c_i,
                    apo_i,
                    multimer_token_cache[cache_key][source_asym_id],
                    key=f"{source}:{key}",
                )

    def determine_confidence_train_data(self, metadata: Metadata) -> bool:
        # For RCSB training dataset, we only train confidence head on the
        # high-resolution experimental structures.
        assert metadata.source == "rcsb", (
            f"Expected metadata source to be 'rcsb' for RCSBTrainingDataset,"
            f" but got '{metadata.source}'."
        )
        assert metadata.exp is not None, (
            "Experimental metadata must be available for RCSBTrainingDataset."
        )
        # Train the confidence head only on experimental structures.
        resolution = metadata.exp.resolution
        if resolution is not None and 0.1 <= resolution <= 4.0:
            return True
        return False


class DisorderedPDBTrainingDataset(RCSBTrainingDataset):
    """Training dataset for disordered PDB structures predicted by
    AlphaFold-multimer"""

    def setup(self) -> None:
        """Disable RCSB multimer apo policy for disordered PDB."""
        self.prob_use_complex_apo = 0.0
        self.prob_use_complex_prior = 0.0
        self.apo_multimer_lookup_table = {}

    def determine_confidence_train_data(self, metadata: Metadata) -> bool:
        return False


def clean_up_ref_structure(
    struct: RefStructure,
) -> RefStructure:
    """Prepare the reference structure from chains.

    Parameters
    ----------
    struct : RefStructure
        The reference structure to be cleaned.

    Returns
    -------
    clean_struct : RefStructure
        The cleaned reference structure.
    """
    # clean up each chain
    struct = RefStructure(
        chains=[clean_up_chain(chain) for chain in struct.chains],
        connections=struct.connections,
        metadata=struct.metadata,
    )
    # clean up connections
    struct = clean_up_connections(struct)
    return struct


def clean_up_chain(
    chain: Chain,
) -> Chain:
    """Clean up the reference chain.

    Parameters
    ----------
    chain : RefStructure.Chain
        The reference chain to be cleaned.

    Returns
    -------
    clean_chain : Chain
        The cleaned reference chain.
    """

    # For now, we only remove bonds with unrealistic bond lengths.
    bonds = chain.bond
    is_valid = np.zeros(len(bonds), dtype=bool)
    for bond_i in range(len(bonds)):
        res_idx1, res_idx2 = bonds.residue_index[bond_i]
        atom1, atom2 = bonds.atom_name[bond_i]
        aidx1 = chain.find_atom_index(res_idx1, atom1)
        aidx2 = chain.find_atom_index(res_idx2, atom2)

        coord1 = chain.atom.coords[aidx1]
        coord2 = chain.atom.coords[aidx2]
        dsq = ((coord1 - coord2) ** 2).sum()
        if dsq < MAX_BOND_LENGTH**2 or np.isnan(dsq):
            # Keep the bond if the distance is less than the threshold
            # or if the distance is NaN since we can't ensure this is
            # an unrealistic bond
            is_valid[bond_i] = True

    clean_bond = BondLayout(
        residue_index=bonds.residue_index[is_valid],
        atom_name=bonds.atom_name[is_valid],
        bond_type=bonds.bond_type[is_valid],
    )

    return chain.copy_with(bond=clean_bond)


def clean_up_connections(
    struct: RefStructure,
) -> RefStructure:
    """Clean up the covalent connections by removing unrealistic connections.

    Parameters
    ----------
    struct : RefStructure
        The reference structure containing the connections to be cleaned.

    Returns
    -------
    clean_struct : RefStructure
        The reference structure with cleaned connections.
    """
    # For training, remove connections with unrealistic bond lengths.
    connections: list[CovalentConnection] = struct.connections
    if len(connections) == 0:
        return struct

    asym_id_to_chain = {c.asym_id: c for c in struct.chains}
    clean_connections: list[CovalentConnection] = []
    for conn in connections:
        asym_id1, asym_id2 = conn.asym_id
        res_idx1, res_idx2 = conn.residue_index
        atom1, atom2 = conn.atom_names
        chain1 = asym_id_to_chain[asym_id1]
        chain2 = asym_id_to_chain[asym_id2]

        # Get the coordinates of the connected atoms
        aidx1 = chain1.find_atom_index(res_idx1, atom1)
        aidx2 = chain2.find_atom_index(res_idx2, atom2)

        coord1 = chain1.atom.coords[aidx1]
        coord2 = chain2.atom.coords[aidx2]
        dsq = ((coord1 - coord2) ** 2).sum()

        if dsq < MAX_BOND_LENGTH**2 or np.isnan(dsq):
            # Keep the bond if the distance is less than the threshold
            # or if the distance is NaN since we can't ensure this is
            # an unrealistic bond
            clean_connections.append(conn)

    return RefStructure(
        chains=struct.chains,
        connections=clean_connections,
        metadata=struct.metadata,
    )
