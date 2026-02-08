"""Tokenization pipeline for structures."""

from functools import lru_cache

import numpy as np
from rdkit import Chem

import kfold.constants as C
from kfold.constants.interaction import get_residue_interaction_type
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.utils.geometry.random_augment import center_random_augmentation, do_centering
from kfold.utils.geometry.rigid_align import compute_rmsd

from .apo_initialization import (
    get_ambiguous_atoms_in_residue,
    get_molecule_symmetries,
)
from .prior_sampling import PriorSampler


class Tokenizer:
    def __init__(self, prior_sampler: PriorSampler, ccd: CCD):
        """Tokenizer for structures.

        Parameters
        ----------
        prior_sampler : PriorSampler
            The prior sampler.
        ccd : CCD
            The chemical component dictionary.
        """
        self.prior_sampler: PriorSampler = prior_sampler
        self.ccd: CCD = ccd

    def __call__(
        self,
        input: RefStructure,
        rng: np.random.Generator | None = None,
        use_only_cached_conformers: bool = False,
        ref_pos_permutation: bool = False,
    ) -> TokenizedStructure:
        """Tokenize structure.

        Parameters
        ----------
        input : RefStructure
            The input structure.
        rng : np.random.Generator, optional
            Random number generator for stochastic processes, by default None.
        use_only_cached_conformers : bool, optional
            if True, only the cached conformers in the CCD will be used.
        ref_pos_permutation : bool, optional
            If True, apply permutation to reference positions to match label structure.

        Returns
        -------
        struct: TokenizedStructure
            The parsed tokenized structure.
        """
        return self.tokenize(input, rng, use_only_cached_conformers, ref_pos_permutation)

    def tokenize(
        self,
        input: RefStructure,
        rng: np.random.Generator | None = None,
        use_only_cached_conformers: bool = False,
        ref_pos_permutation: bool = False,
    ) -> TokenizedStructure:
        """Tokenize structure.

        Parameters
        ----------
        input : RefStructure
            The input structure.
        rng : np.random.Generator, optional
            Random number generator for stochastic processes, by default None.
        use_only_cached_conformers : bool, optional
            if True, only the cached conformers in the CCD will be used.
        ref_pos_permutation : bool, optional
            If True, apply permutation to reference positions to match label structure.

        Returns
        -------
        struct: TokenizedStructure
            The parsed tokenized structure.
        """
        return tokenize_structure(
            input,
            self.prior_sampler,
            self.ccd,
            rng,
            use_only_cached_conformers,
            ref_pos_permutation=ref_pos_permutation,
        )


def tokenize_structure(
    input: RefStructure,
    prior_sampler: PriorSampler,
    ccd: CCD,
    rng: np.random.Generator | None = None,
    use_only_cached_conformers: bool = False,
    ref_pos_permutation: bool = False,
) -> TokenizedStructure:
    """Tokenize structure.

    Parameters
    ----------
    input : RefStructure
        The input structure.
    ccd : CCD
        The chemical component dictionary.
    rng : np.random.Generator, optional
        Random number generator for stochastic processes, by default None.
    use_only_cached_conformers : bool, optional
        if True, only the cached conformers in the CCD will be used.
          - EKTDG-cached (up to 10 conformers by default with `ccd-train.pkl`)
          - Ideal
          - Model (Experimental)
    ref_pos_permutation : bool, optional
        If True, apply permutation to reference positions to match label structure.

    Returns
    -------
    struct: TokenizedStructure
        The parsed tokenized structure.
    """

    @lru_cache  # No cache limit within a single function call
    def get_ccd_component(ccd_name: str) -> Component:
        """Get CCD component with caching.

        NOTE (SeonghwanSeo): CCD.__getitem__ deserialize the Component,
        which is time-consuming. Therefore, we cache the Component objects here.
        The cache is removed when the function is terminated.
        """
        return ccd[ccd_name]

    rng = rng or np.random.default_rng()

    conformer_mode = "auto"
    if use_only_cached_conformers:
        conformer_mode = "train"

    # ==================================================
    # Estimate sizes
    # ==================================================
    num_chains = len(input.chains)
    chain_token_st: dict[int, int] = {}
    chain_atom_st: dict[int, int] = {}
    token_offset = 0
    atom_offset = 0
    for chain in input.chains:
        chain_token_st[chain.asym_id] = token_offset
        token_offset += chain.num_tokens
        chain_atom_st[chain.asym_id] = atom_offset
        atom_offset += chain.num_atoms

    # ==================================================
    # Create empty tokenized structure
    # ==================================================
    struct = TokenizedStructure.get_empty(
        num_chains=len(input.chains),
        num_residues=input.num_residues,
        num_tokens=input.num_tokens,
        num_bonds=input.num_bonds + input.num_connections,
        num_priors=prior_sampler.num_samples,
    )

    # ==================================================
    # Collect all component in the structure
    # ==================================================
    # (asym_id, residue_index) -> Component
    ccd_components: dict[tuple[int, int], Component] = {}
    for chain in input.chains:
        asym_id = chain.asym_id
        ccd_sequence: list[str] = chain.get_ccd_sequence()
        for res_idx, ccd_name in enumerate(ccd_sequence, start=1):
            if ccd_name.startswith("LIG"):
                # This residue is from a smiles string, load smiles from metadata
                smiles = chain.smiles
                assert smiles is not None, (
                    "Smiles string not found in metadata for LIG residue."
                )
                assert chain.num_residues == 1, (
                    "Residue with LIG prefix found in chain with multiple residues."
                )
                comp: Component = Component.from_smiles(ccd_name, smiles, num_confs=1)
            else:
                assert ccd_name in ccd, f"Residue name {ccd_name} not found in CCD."
                comp = get_ccd_component(ccd_name)
            ccd_components[(asym_id, res_idx)] = comp

    # ==================================================
    # Fill chain structures
    # ==================================================
    for chain_i in range(num_chains):
        chain = input.chains[chain_i]
        # Insert chain info
        struct.chain.chain_type[chain_i] = chain.chain_type
        struct.chain.entity_id[chain_i] = chain.entity_id
        struct.chain.asym_id[chain_i] = chain.asym_id
        struct.chain.sym_id[chain_i] = chain.sym_id
        struct.chain.num_residues[chain_i] = chain.num_residues
        struct.chain.num_atoms[chain_i] = chain.num_atoms
        struct.chain.num_tokens[chain_i] = chain.num_tokens

    # ==================================================
    # Fill residue and token structures
    # ==================================================
    g_res_i = 0
    g_tok_i = 0
    g_atom_i = 0
    # Map from global atom index to (token_index, local_atom_index)
    g_atom_to_token_map: dict[int, tuple[int, int]] = {}

    for chain in input.chains:
        ctype = chain.ctype
        asym_id = chain.asym_id
        ccd_sequence: list[str] = chain.get_ccd_sequence()
        all_atom_names: list[str] = chain.atom.name.tolist()

        # Iterate residues in the chain and fill token and some atom info
        for res_i in range(chain.num_residues):
            res_idx = res_i + 1  # 1-based index

            # Get residue info
            ccd_name: str = ccd_sequence[res_i]
            res_name: C.ResidueName = C.residue.get_residue_name_with_unk(ccd_name, ctype)
            restype: int = res_name.value
            is_res_standard = chain.residue.is_standard[res_i]
            comp: Component = ccd_components[(asym_id, res_idx)]

            # Get atom info
            atom_names = all_atom_names[chain.residue.get_atom_slice(res_idx)]
            natoms = len(atom_names)

            # Determine number of tokens
            ntokens = 1 if is_res_standard else natoms

            # Insert additional residue info
            struct.residue.residue_index[g_res_i] = res_idx
            struct.residue.name[g_res_i] = ccd_name
            struct.residue.res_type[g_res_i] = restype
            struct.residue.num_atoms[g_res_i] = natoms
            struct.residue.num_tokens[g_res_i] = ntokens
            struct.residue.is_standard[g_res_i] = is_res_standard

            if is_res_standard:
                # Standard protein/dna/rna residues (including ambiguous residues)
                ref_atom_idx: int = C.atom.REF_ATOM_INDEX[res_name]
                disto_atom_idx: int = C.atom.PSEUDO_BETA_ATOM_INDEX[res_name]
                struct.token.num_atoms[g_tok_i] = natoms
                struct.token.center_index[g_tok_i] = ref_atom_idx
                struct.token.disto_index[g_tok_i] = disto_atom_idx
                struct.token.is_standard[g_tok_i] = True

                # Insert pre-defined non-covalent interaction types
                nci_indices = get_residue_interaction_type(res_name)
                if nci_indices:
                    struct.token.interaction_type[g_tok_i, nci_indices] = True

                # Update atom existence mask
                struct.atom.pad_mask[g_tok_i, :natoms] = True

                # Map global atom index to (token_index, local_index)
                for _i in range(natoms):
                    g_atom_to_token_map[g_atom_i] = (g_tok_i, _i)
                    g_atom_i += 1

                # Update global token index
                g_tok_i += 1
            else:
                # Ligands, Modifications, Covalent inhibitors
                st = g_tok_i
                end = g_tok_i + natoms
                struct.token.num_atoms[st:end] = 1
                struct.token.center_index[st:end] = 0
                struct.token.disto_index[st:end] = 0
                struct.token.is_standard[st:end] = False

                # Insert interaction types from CCD component
                atom_indices = comp.get_atom_indices(atom_names)
                nci_types = comp.interaction_types[atom_indices]
                struct.token.interaction_type[st:end, :] = nci_types

                # Update atom existence mask
                struct.atom.pad_mask[st:end, 0] = True

                for _ in range(natoms):
                    # Map global atom index to (token_index, local_atom_index)
                    g_atom_to_token_map[g_atom_i] = (g_tok_i, 0)
                    # Update global token index
                    g_atom_i += 1
                    g_tok_i += 1

            # Update global residue index
            g_res_i += 1

    assert g_res_i == input.num_residues, "Global residue index does not match."
    assert g_tok_i == input.num_tokens, "Global token index does not match."
    assert g_atom_i == input.num_atoms, "Global atom index does not match."

    # Propagate chain features to residue and token levels
    for k in ["chain_type", "entity_id", "asym_id", "sym_id"]:
        chain_feat = getattr(struct.chain, k)
        residue_feat = getattr(struct.residue, k)
        token_feat = getattr(struct.token, k)
        residue_feat[:] = np.repeat(chain_feat, struct.chain.num_residues, axis=0)
        token_feat[:] = np.repeat(chain_feat, struct.chain.num_tokens, axis=0)

    # Propagate residue features to token levels
    for k in ["res_type", "residue_index"]:
        residue_feat = getattr(struct.residue, k)
        token_feat = getattr(struct.token, k)
        token_feat[:] = np.repeat(residue_feat, struct.residue.num_tokens, axis=0)

    # Set default token index
    struct.token.token_index[:] = np.arange(input.num_tokens, dtype=np.int64)

    # ==================================================
    # Fill atom structures
    # ==================================================
    g_tok_i = 0
    for chain in input.chains:
        ctype = chain.ctype
        asym_id = chain.asym_id
        ccd_sequence: list[str] = chain.get_ccd_sequence()
        all_atom_names: list[str] = chain.atom.name.tolist()

        token_st: int = chain_token_st[asym_id]
        token_end: int = token_st + chain.num_tokens
        assert g_tok_i == token_st, "Global token index does not match."

        # Valid atom mask for the chain
        pad_mask = struct.atom.pad_mask[token_st:token_end]  # (chain_tokens, 24)
        assert pad_mask.sum() == chain.num_atoms, "Number of valid atoms does not match"

        # Insert ground-truth coordinates
        struct.atom.label_coords[token_st:token_end][pad_mask] = chain.atom.coords

        # Insert apo coordinates
        struct.atom.apo_coords[token_st:token_end][pad_mask] = chain.atom.apo_coords

        # Insert reference atom info except reference conformers
        struct.atom.ref_atom_name_chars[token_st:token_end][pad_mask] = np.array(
            [C.atom.encode_atom_name(name) for name in chain.atom.name.tolist()]
        )  # (num_atoms, 4)
        struct.atom.ref_element[token_st:token_end][pad_mask] = chain.atom.element
        struct.atom.ref_charge[token_st:token_end][pad_mask] = chain.atom.charge.astype(
            np.float32
        )

        # Insert reference molecular conformers
        for res_i in range(chain.num_residues):
            res_idx = res_i + 1  # 1-based index
            is_standard = chain.residue.is_standard[res_i]
            ref_comp: Component = ccd_components[(asym_id, res_idx)]

            # Get reference conformer positions with random augmentation
            ref_pos: np.ndarray = ref_comp.get_conformer(conformer_mode, rng)  # type: ignore
            assert ref_pos is not None, "Auto mode always provides a conformer."

            # Insert coordinates based on atom names
            atom_slices = chain.residue.get_atom_slice(res_idx)
            atom_names = all_atom_names[atom_slices]
            atom_indices: list[int] = ref_comp.get_atom_indices(atom_names)
            natoms = len(atom_names)
            ref_pos = ref_pos[atom_indices, :]
            ref_mask = np.isfinite(ref_pos).all(axis=-1)

            if ref_mask.any():
                # Match residue permutation to label structure
                if ref_pos_permutation:
                    label_pos = chain.atom.coords[atom_slices]
                    align_mask = np.isfinite(label_pos).all(axis=-1) & ref_mask
                    perm = find_best_residue_permutation(
                        ref_pos, label_pos, ref_comp, atom_names, align_mask, is_standard
                    )
                    ref_pos, ref_mask = ref_pos[perm], ref_mask[perm]

                # Apply random augmentation to reference positions
                ref_pos = center_random_augmentation(ref_pos, ref_mask, rng=rng)
                ref_pos[~ref_mask] = np.nan

            if is_standard:
                # Standard residue (one token)
                assert np.all(struct.atom.pad_mask[g_tok_i, :natoms]), (
                    "Atom pad mask mismatch for standard residue."
                    f" (g_tok_i={g_tok_i}, natoms={natoms})"
                )
                struct.atom.ref_pos[g_tok_i, :natoms, :] = ref_pos
                g_tok_i += 1
            else:
                # Non-standard residue (multiple tokens, one per atom)
                assert np.all(struct.atom.pad_mask[g_tok_i : g_tok_i + natoms, 0]), (
                    "Atom pad mask mismatch for non-standard residue."
                )
                st = g_tok_i
                end = g_tok_i + natoms
                struct.atom.ref_pos[st:end, 0, :] = ref_pos
                g_tok_i += natoms

    # Sample prior coordinates (xT)
    pad_mask = struct.atom.pad_mask
    if prior_sampler.num_samples > 0:
        prior_coords = prior_sampler(input, rng=rng)  # (num_samples, num_atoms, 3)
        struct.atom.prior_coords[pad_mask] = prior_coords.transpose(1, 0, 2)

    # Update atom masks at once
    struct.atom.ref_mask[:] = np.isfinite(struct.atom.ref_pos).all(axis=-1)
    struct.atom.resolved_mask[:] = np.isfinite(struct.atom.label_coords).all(axis=-1)
    struct.atom.apo_mask[:] = np.isfinite(struct.atom.apo_coords).all(axis=-1)

    # Update NaN to zero
    struct.atom.ref_charge[pad_mask] = np.nan_to_num(
        struct.atom.ref_charge[pad_mask], nan=0.0
    )

    # Update apo/holo coordinates (centering & NaN to zero)
    struct.atom.apo_coords[:] = do_centering(
        struct.atom.apo_coords.reshape(-1, 3),
        struct.atom.apo_mask.reshape(-1),
        mask_to_zero=False,
    ).reshape(struct.atom.apo_coords.shape)
    struct.atom.label_coords[:] = do_centering(
        struct.atom.label_coords.reshape(-1, 3),
        struct.atom.resolved_mask.reshape(-1),
        mask_to_zero=False,
    ).reshape(struct.atom.label_coords.shape)

    # ==================================================
    # Fill bond structures
    # ==================================================
    # First iterate intra-chain bonds
    g_bond_i = 0
    for chain_i in range(num_chains):
        chain = input.chains[chain_i]
        asym_id = chain.asym_id
        for bond_i in range(chain.num_bonds):
            ridx1, ridx2 = chain.bond.residue_index[bond_i]

            atom1, atom2 = chain.bond.atom_name[bond_i].tolist()
            bondtype: C.ConnectionType = C.bond.bond_type_to_connection_type(
                Chem.BondType.values[int(chain.bond.bond_type[bond_i])]
            )
            # Find atom index
            atom_i1 = chain.find_atom_index(ridx1, atom1)
            atom_i2 = chain.find_atom_index(ridx2, atom2)

            # Map chain-local atom indices to global atom indices
            g_atom_i1 = chain_atom_st[asym_id] + atom_i1
            g_atom_i2 = chain_atom_st[asym_id] + atom_i2

            # Map atom indices to token and atom indices
            g_tok_i1, local_atom1 = g_atom_to_token_map[g_atom_i1]
            g_tok_i2, local_atom2 = g_atom_to_token_map[g_atom_i2]

            assert (
                struct.token.is_ligand[g_tok_i1] and struct.token.is_ligand[g_tok_i2]
            ), "Intra-chain bonds should only exist within ligand chains."

            # Insert bond info
            struct.bond.asym_id[g_bond_i, :] = asym_id
            struct.bond.token_index[g_bond_i] = (g_tok_i1, g_tok_i2)
            struct.bond.atom_index[g_bond_i] = (local_atom1, local_atom2)
            struct.bond.bond_type[g_bond_i] = bondtype.value

            # Update global bond index
            g_bond_i += 1

    # Then iterate cross-chain bonds
    for conn_i in range(input.num_connections):
        asym_id1, asym_id2 = input.connections[conn_i].asym_id
        ridx1, ridx2 = input.connections[conn_i].residue_index
        atom1, atom2 = input.connections[conn_i].atom_names
        bondtype = C.ConnectionType.INTERMOLECULAR

        # Find atom index
        chain1 = input.get_chain_by_asym_id(asym_id1)
        chain2 = input.get_chain_by_asym_id(asym_id2)

        # Find atom index
        aidx1 = chain1.find_atom_index(ridx1, atom1)
        aidx2 = chain2.find_atom_index(ridx2, atom2)

        # Map chain-local atom indices to global atom indices
        g_aidx1 = chain_atom_st[asym_id1] + int(aidx1)
        g_aidx2 = chain_atom_st[asym_id2] + int(aidx2)

        # Map atom indices to token and atom indices
        g_tok_i1, local_atom1 = g_atom_to_token_map[g_aidx1]
        g_tok_i2, local_atom2 = g_atom_to_token_map[g_aidx2]

        # Insert bond info
        struct.bond.asym_id[g_bond_i] = (asym_id1, asym_id2)
        struct.bond.token_index[g_bond_i] = (g_tok_i1, g_tok_i2)
        struct.bond.atom_index[g_bond_i] = (local_atom1, local_atom2)
        struct.bond.bond_type[g_bond_i] = bondtype.value

        # Update global bond index
        g_bond_i += 1

    # Sanity check
    struct.validate()

    return struct


def find_best_residue_permutation(
    ref_pos: np.ndarray,
    label_pos: np.ndarray,
    ref_comp: Component,
    atom_names: list[str],
    mask: np.ndarray,
    is_standard: bool,
) -> np.ndarray:
    """Find the best permutation of reference positions to match label positions.

    Parameters
    ----------
    ref_pos : np.ndarray
        Reference positions. Shape: (N, 3)
    label_pos : np.ndarray
        Label positions. Shape: (N, 3)
    ref_comp : Component
        Reference component from CCD.
    atom_names : list[str]
        List of atom names in the residue.
    mask : np.ndarray
        Boolean mask indicating valid atoms. Shape: (N,)
    is_standard : bool
        Whether the residue is standard.

    Returns
    -------
    np.ndarray
        The best permutation of reference positions. Shape: (N,)
    """
    if not mask.any():
        return np.arange(ref_pos.shape[0])

    if is_standard:
        # Standard residue: use predefined ambiguous atom groups
        perms = get_ambiguous_atoms_in_residue(ref_comp.code)
    else:
        # Non-standard residue: use molecular symmetries from CCD
        perms = get_molecule_symmetries(ref_comp, atom_names)

    if perms is None or len(perms) == 0:
        return np.arange(ref_pos.shape[0])

    best_rmsd = np.inf
    best_perm = list(range(ref_pos.shape[0]))
    for perm in perms:
        permuted_pos = ref_pos[perm, :]
        rmsd = compute_rmsd(
            permuted_pos[mask], label_pos[mask], mask=None, align=True, no_svd=True
        )
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm = perm
    return np.array(best_perm)
