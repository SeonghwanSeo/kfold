import itertools
import logging

import numpy as np
from scipy.spatial.distance import cdist

from kfold.data.types.constraint import Constraint
from kfold.data.types.metadata import InterfaceInfo
from kfold.data.types.structure import Chain, RefStructure
from kfold.utils.misc import spawn_rng

INTRA_PROTEIN_CONTACT_DISTANCE = 8.0
INTRA_NUCLEIC_ACID_CONTACT_DISTANCE = 12.0
INTERFACE_DISTANCE = 15.0


logger = logging.getLogger(__name__)


class ConstraintSampling:
    """Constraint sampling module"""

    def __init__(
        self,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        prob_constraint: float = 0.1,
        max_constraints: int = 5,
        prob_contact: float = 0.8,
        prob_intra_chain: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        min_dist : float
            The minimum distance for sampling constraints.
        max_dist : float
            The maximum distance for sampling constraints.
        prob_constraint : float
            The probability of sampling constraints.
        max_constraints : int
            The maximum number of constraints to sample.
        prob_contact : float
            The probability of sampling constraints from contact regions.
        prob_intra_chain : float
            The probability of sampling intra-chain constraints versus inter-chain.
        """
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist
        self.prob_constraint: float = prob_constraint
        self.max_constraints: int = max_constraints
        self.prob_contact: float = prob_contact
        self.prob_intra_chain: float = prob_intra_chain

    def __call__(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator | None = None,
    ) -> list[Constraint]:
        """Sample constraints from the reference structure.

        Parameters
        ----------
        ref_struct : RefStructure
            The reference structure to sample constraints from.
        rng : numpy.random.Generator, optional
            Random number generator for stochastic processes, by default None.

        Returns
        -------
        list[Constraint]
            A list of sampled constraints.
        """
        return self.sample_constraints(ref_struct, rng)

    def sample_constraints(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator | None = None,
    ) -> list[Constraint]:
        rng = spawn_rng(rng)

        # Sample whether to sample constraints
        if rng.random() >= self.prob_constraint:
            return []

        # Sample the number of constraints to sample
        p = np.linspace(1.0, 0.0, self.max_constraints + 1)
        p /= p.sum()
        num_constraints = int(rng.choice(np.arange(1, self.max_constraints + 2), p=p))

        # Sample constraints
        constraints: list[Constraint] = []
        for _ in range(num_constraints):
            try:
                if (
                    len(ref_struct.chains) <= 1
                    or len(ref_struct.metadata.interfaces) == 0
                    or rng.random() <= self.prob_intra_chain
                ):
                    # Sample from polymer chains
                    polymer_chains = [c for c in ref_struct.chains if c.is_polymer]
                    chain = polymer_chains[rng.integers(len(polymer_chains))]
                    cond = self.sample_intra_chain_constraint(chain, rng)
                else:
                    # Sample from interfaces
                    cond = self.sample_inter_chain_constraint(ref_struct, rng)

            except Exception as e:
                # In case of any error during sampling, skip this constraint
                logger.warning(f"Error during constraint sampling: {e}")
                continue

            if cond is not None:
                constraints.append(cond)

        return constraints

    def sample_intra_chain_constraint(
        self,
        chain: Chain,
        rng: np.random.Generator,
    ) -> Constraint | None:
        """Sample an intra-chain constraint."""
        if not chain.is_polymer:
            return None

        sample_contact = rng.random() < self.prob_contact

        if chain.is_protein:
            threshold = INTRA_PROTEIN_CONTACT_DISTANCE
            center_atom = "CA"
            min_sequence_separation = 24
            prob_dynamic = 0.5
        else:
            threshold = INTRA_NUCLEIC_ACID_CONTACT_DISTANCE
            center_atom = "C1'"
            min_sequence_separation = 12
            prob_dynamic = 0.0
        d_min = 0 if sample_contact else threshold
        d_max = threshold if sample_contact else self.max_dist + 5.0
        return self._sample_intra_chain_constraint(
            chain,
            center_atom,
            min_sequence_separation,
            min_distance=d_min,
            max_distance=d_max,
            prob_dynamic=prob_dynamic,
            rng=rng,
        )

    def _sample_intra_chain_constraint(
        self,
        chain: Chain,
        center_atom_name: str,
        min_sequence_separation: int,
        min_distance: float,
        max_distance: float,
        prob_dynamic: float,
        rng: np.random.Generator,
    ) -> Constraint | None:
        """Sample an intra-chain constraint from a polymer chain."""
        assert chain.is_polymer, (
            "Intra-chain constraints can only be sampled from polymer chains."
        )
        if len(chain.residue) <= min_sequence_separation:
            return None

        num_residues = len(chain.residue)
        is_center = chain.atom.name == center_atom_name  # [L,]

        if is_center.sum() == num_residues:
            res_indices = np.arange(1, num_residues + 1)  # [L,]
        else:
            atom_end = chain.residue.atom_ends
            atom_indices = np.where(is_center)[0]
            res_indices = np.sum(atom_indices[:, None] >= atom_end[None, :], axis=1) + 1

        x_holo = chain.atom.coords[is_center]  # [L, 3]

        # Create a mask for valid residue pairs based on sequence separation
        # NOTE: Here, we only allow pairs with i > j to avoid redundant pairs.
        mask = res_indices[:, None] - res_indices[None, :] >= min_sequence_separation

        # Mask out pairs where either residue has unresolved CA coordinates
        resolved_mask = np.isfinite(x_holo).all(axis=-1)
        mask &= resolved_mask[:, None] & resolved_mask[None, :]

        if not mask.any():
            return None

        d_holo = cdist(x_holo, x_holo).astype(np.float32)
        mask &= d_holo <= max_distance
        mask &= d_holo >= min_distance
        if not mask.any():
            return None

        x_apo = chain.atom.apo_coords[is_center]
        mask_apo = np.isfinite(x_apo).all(axis=-1)
        is_apo_available = mask_apo.any()

        if is_apo_available and rng.random() < prob_dynamic:
            # Use the difference between holo and apo distances as a proxy for
            # dynamic regions.
            d_apo = cdist(x_apo, x_apo).astype(np.float32, copy=False)
            diff = np.abs(d_holo - d_apo)
            mask &= np.isfinite(diff)
            if not mask.any():
                return None
            # Normalize the difference based on the apo distance
            diff /= d_apo + 1e-6
            # Smoothing the probability
            p_sample = np.sqrt(diff)
        else:
            p_sample = np.ones_like(d_holo, dtype=np.float32)

        p_sample[~mask] = 0.0
        if p_sample.sum() == 0.0:
            return None

        # Sample a residue pair.
        L = res_indices.size
        p_sample /= p_sample.sum()
        idx = rng.choice(np.arange(L * L), p=p_sample.flatten())
        i, j = divmod(idx, L)

        dist = d_holo[i, j].clip(1e-3)
        lower_bound = float(rng.triangular(0, dist, dist))
        upper_bound = (
            float(rng.triangular(dist, dist, self.max_dist))
            if dist < self.max_dist
            else -1.0
        )
        # Drop one of the bounds to create more diverse constraints.
        drop_prob = rng.random()
        if drop_prob < 0.05:
            lower_bound = -1.0
        elif drop_prob < 0.1:
            upper_bound = -1.0

        return Constraint(
            asym_id=(chain.asym_id, chain.asym_id),
            residue_index=(int(res_indices[i]), int(res_indices[j])),
            atom_name=(center_atom_name, center_atom_name),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def sample_inter_chain_constraint(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
    ) -> Constraint | None:
        """Sample a constraint between two chains in an interface."""
        chains: list[Chain] = ref_struct.chains
        all_asym_ids: list[int] = [c.asym_id for c in chains]
        asym_id_to_chain: dict[int, Chain] = {c.asym_id: c for c in chains}
        assert len(chains) > 1, (
            "At least two chains are required for sampling inter-chain constraints."
        )

        interfaces: list[InterfaceInfo] = ref_struct.metadata.interfaces
        assert len(interfaces) > 0, "No interfaces available for sampling."

        contact_pairs: list[tuple[int, int]] = [iface.asym_ids for iface in interfaces]
        _contact_pairs_set = set(tuple(sorted(p)) for p in contact_pairs)
        noncontact_pairs: list[tuple[int, int]] = sorted(
            set(itertools.combinations(all_asym_ids, 2)) - _contact_pairs_set
        )

        # Filter out ligand-ligand interfaces
        contact_pairs = [
            p for p in contact_pairs if any(asym_id_to_chain[i].is_polymer for i in p)
        ]
        noncontact_pairs = [
            p for p in noncontact_pairs if any(asym_id_to_chain[i].is_polymer for i in p)
        ]

        if rng.random() < self.prob_contact:
            # Sample from the interface
            if len(contact_pairs) == 0:
                return None
            asym_id1, asym_id2 = contact_pairs[rng.integers(len(contact_pairs))]
            chain1, chain2 = asym_id_to_chain[asym_id1], asym_id_to_chain[asym_id2]
            return self._sample_interface_constraint(chain1, chain2, rng)
        else:
            # Sample from non-interface regions
            if len(noncontact_pairs) == 0:
                return None
            asym_id1, asym_id2 = noncontact_pairs[rng.integers(len(noncontact_pairs))]
            chain1, chain2 = asym_id_to_chain[asym_id1], asym_id_to_chain[asym_id2]
            return self._sample_non_interface_constraint(chain1, chain2, rng)

    def _get_repr_atoms(self, chain: Chain) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get representative atoms for different chain types."""
        if chain.is_protein:
            is_center = chain.atom.name == "CA"
        elif chain.is_dna or chain.is_rna:
            # Use C1' for nucleic acids
            is_center = chain.atom.name == "C1'"
        else:
            # Use all atoms for ligands
            is_center = np.ones(len(chain.atom), dtype=bool)

        center_coords = chain.atom.coords[is_center]
        atom_names = chain.atom.name[is_center]
        if is_center.sum() == len(chain.residue):
            res_indices = np.arange(1, len(chain.residue) + 1)
        else:
            atom_end = chain.residue.atom_ends
            atom_indices = np.where(is_center)[0]
            res_indices = np.sum(atom_indices[:, None] >= atom_end[None, :], axis=1) + 1
        return center_coords, atom_names, res_indices

    def _sample_interface_constraint(
        self,
        chain1: Chain,
        chain2: Chain,
        rng: np.random.Generator,
    ) -> Constraint | None:
        x1, names1, res_indices1 = self._get_repr_atoms(chain1)
        x2, names2, res_indices2 = self._get_repr_atoms(chain2)

        if len(x1) == 0 or len(x2) == 0:
            return None

        dists = cdist(x1, x2).astype(np.float32)
        mask = dists <= INTERFACE_DISTANCE
        if not mask.any():
            return None

        # Sample from the interface pairs, with preference for closer pairs
        d = dists.clip(min=self.min_dist, max=self.max_dist)
        p_sample = mask.astype(np.float32)
        p_sample /= d  # prefer closer pairs
        p_sample[~mask] = 0.0
        p_sample /= p_sample.sum()  # normalize to get probabilities
        idx = rng.choice(np.arange(dists.size), p=p_sample.flatten())
        i, j = divmod(idx, dists.shape[1])

        dist = dists[i, j].clip(1e-3)
        lower_bound = float(rng.triangular(0, dist, dist))
        upper_bound = (
            float(rng.triangular(dist, dist, self.max_dist))
            if dist < self.max_dist
            else -1.0
        )
        # In general, interface constraints are more likely to upper bound only,
        # so randomly drop the lower bound.
        if upper_bound > 0.0 and rng.random() < 0.5:
            lower_bound = -1.0

        return Constraint(
            asym_id=(chain1.asym_id, chain2.asym_id),
            residue_index=(int(res_indices1[i]), int(res_indices2[j])),
            atom_name=(str(names1[i]), str(names2[j])),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def _sample_non_interface_constraint(
        self,
        chain1: Chain,
        chain2: Chain,
        rng: np.random.Generator,
    ) -> Constraint | None:
        """Sample a constraint between two chains that are not in contact."""
        x1, names1, res_indices1 = self._get_repr_atoms(chain1)
        x2, names2, res_indices2 = self._get_repr_atoms(chain2)

        if len(x1) == 0 or len(x2) == 0:
            return None

        dists = cdist(x1, x2).astype(np.float32)
        mask = dists > self.max_dist
        if not mask.any():
            return None

        # Sample
        p_sample = mask.astype(np.float32)
        p_sample /= p_sample.sum()
        idx = rng.choice(np.arange(dists.size), p=p_sample.flatten())
        i, j = divmod(idx, dists.shape[1])

        return Constraint(
            asym_id=(chain1.asym_id, chain2.asym_id),
            residue_index=(int(res_indices1[i]), int(res_indices2[j])),
            atom_name=(str(names1[i]), str(names2[j])),
            lower_bound=self.max_dist,
            upper_bound=-1.0,
        )
