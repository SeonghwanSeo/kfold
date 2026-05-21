"""Tokenization pipeline for structures."""

import math

import numpy as np
from rdkit import Chem

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.constraint import Constraint
from kfold.data.types.structure import Chain, RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.utils.geometry.random_augment import center_random_augmentation, do_centering
from kfold.utils.misc import spawn_rng

COLLISION_ANGLE_CUTOFF = math.cos(math.radians(25))  # 25 degree
DETERMINISTIC_FRAME_SEED = 20000106

frame_atoms = C.atom.CHAIN_FRAME_ATOMS[C.ChainType.PROTEIN]
PROTEIN_FRAME_ATOM_INDICES: dict[C.ResidueName, tuple[int, int, int]] = {
    res: tuple(C.atom.get_residue_atom_index(res, a) for a in frame_atoms)
    for res in C.residue.PROTEIN_RESIDUES
}

sequence_encode_fn = {
    C.ChainType.PROTEIN: C.sequence.encode_protein_sequence,
    C.ChainType.DNA: C.sequence.encode_dna_sequence,
    C.ChainType.RNA: C.sequence.encode_rna_sequence,
}


def get_mask(coords: np.ndarray) -> np.ndarray:
    """Get mask for valid coordinates: [*, 3] -> [*]."""
    return np.isfinite(coords).all(axis=-1)


class Tokenizer:
    def __init__(self, ccd: CCD, mode: str = "inference"):
        """Tokenizer for structures.

        Parameters
        ----------
        ccd : CCD
            The chemical component dictionary.
        """
        self.ccd: CCD = ccd
        match mode:
            case "train" | "val":
                self.train = True
            case "inference":
                self.train = False
            case _:
                raise ValueError(f"Invalid mode: {mode}")

    def __call__(
        self,
        struct: RefStructure,
        rng: np.random.Generator | None = None,
        *,
        apo_coords: dict[int, np.ndarray] | None = None,
        prior_coords: np.ndarray | None = None,
        constraints: list[Constraint] | None = None,
    ) -> TokenizedStructure:
        """Tokenize structure.

        Parameters
        ----------
        struct : RefStructure
            The input structure.
        rng : np.random.Generator, optional
            Random number generator for stochastic processes, by default None.
        apo_coords : dict[int, np.ndarray]
            A dictionary mapping entity id to apo coordinates for that chain.
        prior_coords : np.ndarray | None, optional
            Prior coordinates, by shape (num_priors, num_atoms, 3).

        Returns
        -------
        struct: TokenizedStructure
            The parsed tokenized structure.
        """
        return self.tokenize(
            struct,
            rng,
            apo_coords=apo_coords,
            prior_coords=prior_coords,
            constraints=constraints,
        )

    def tokenize(
        self,
        struct: RefStructure,
        rng: np.random.Generator | None = None,
        *,
        apo_coords: dict[int, np.ndarray] | None = None,
        prior_coords: np.ndarray | None = None,
        constraints: list[Constraint] | None = None,
    ) -> TokenizedStructure:
        """Tokenize structure.

        Parameters
        ----------
        struct : RefStructure
            The input structure.
        apo_coords : dict[int, np.ndarray]
            A dictionary mapping entity id to apo coordinates for that chain.
        rng : np.random.Generator, optional
            Random number generator for stochastic processes, by default None.
        num_priors : int, optional
            Number of prior conformers to include, by default 0.

        Returns
        -------
        tokenized: TokenizedStructure
            The parsed tokenized structure.
        """
        return tokenize_structure(
            struct,
            self.ccd,
            rng,
            train=self.train,
            apo_coords_dict=apo_coords,
            prior_coords=prior_coords,
            constraints=constraints,
        )


def tokenize_structure(
    struct: RefStructure,
    ccd: CCD,
    rng: np.random.Generator | None = None,
    *,
    train: bool = False,
    apo_coords_dict: dict[int, np.ndarray] | None = None,
    prior_coords: np.ndarray | None = None,
    constraints: list[Constraint] | None = None,
) -> TokenizedStructure:
    """Tokenize structure.

    Parameters
    ----------
    struct : RefStructure
        The input structure.
    ccd : CCD
        The chemical component dictionary.
    rng : np.random.Generator, optional
        Random number generator for stochastic processes, by default None.
    train : bool, optional
        Whether in training mode, by default False.
    apo_coords_dict : dict[int, np.ndarray], optional
        A dictionary mapping entity id to apo coordinates for that chain, by default None.
    prior_coords : np.ndarray | None, optional
        Prior coordinates, by shape (num_priors, num_atoms, 3), by default
    constraints : list[Constraint] | None, optional
        List of additional constraints to include, by default None.

    Returns
    -------
    struct: TokenizedStructure
        The parsed tokenized structure.
    """
    # Create new rng for this sampling to avoid affecting global state
    rng = spawn_rng(rng)

    # ==================================================
    # Collect all component in the structure
    # ==================================================
    ccd_dict: dict[str, Component] = {}
    ccd_smi_dict: dict[str, Component] = {}

    def get_ccd_component(ccd_name: str) -> Component:
        """Get CCD component with caching."""
        if ccd_name not in ccd_dict:
            ccd_dict[ccd_name] = ccd[ccd_name]
        return ccd_dict[ccd_name]

    ccd_components: dict[tuple[int, int], Component] = {}  # key: (asym_id, res_idx)
    ccd_sequence_dict: dict[int, list[str]] = {}
    chain_atom_dict: dict[int, list[str]] = {}
    for c in struct.chains:
        asym_id: int = c.asym_id
        smiles: str | None = c.smiles
        ccd_sequence: list[str] = c.get_ccd_sequence()
        ccd_sequence_dict[asym_id] = ccd_sequence
        chain_atom_dict[asym_id] = c.atom.name.tolist()

        if smiles is not None:
            assert len(ccd_sequence) == 1, (
                "Chain with SMILES should have exactly one residue."
            )

        for res_idx, ccd_name in enumerate(ccd_sequence, start=1):
            if smiles is not None:
                if smiles in ccd_smi_dict:
                    comp = ccd_smi_dict[smiles]
                else:
                    comp: Component = Component.from_smiles(ccd_name, smiles)
                    ccd_smi_dict[smiles] = comp
            else:
                assert ccd_name in ccd, f"Residue name {ccd_name} not found in CCD."
                comp = get_ccd_component(ccd_name)
            ccd_components[(asym_id, res_idx)] = comp

    # ==================================================
    # Create mappings
    # ==================================================
    g_tok_i, g_atom_i = 0, 0
    g_atom_to_token_map: dict[int, tuple[int, int]] = {}
    chain_atom_st: dict[int, int] = {}
    for c in struct.chains:
        chain_atom_st[c.asym_id] = g_atom_i
        for res_i in range(c.num_residues):
            n_atoms = int(c.residue.num_atoms[res_i])
            if c.residue.is_standard[res_i]:
                # Map global atom index to (token_index, local_index)
                g_atom_to_token_map.update(
                    {g_atom_i + _i: (g_tok_i, _i) for _i in range(n_atoms)}
                )
                g_tok_i += 1
                g_atom_i += n_atoms
            else:
                # Ligands, Modifications, Covalent inhibitors
                g_atom_to_token_map.update(
                    {g_atom_i + _i: (g_tok_i + _i, 0) for _i in range(n_atoms)}
                )
                g_tok_i += n_atoms
                g_atom_i += n_atoms
    del g_tok_i, g_atom_i, c, res_i, n_atoms

    # ==================================================
    # Create empty tokenized structure
    # ==================================================
    num_token_seq_tokens = sum(c.num_residues + 2 for c in struct.chains)
    num_priors = prior_coords.shape[0] if prior_coords is not None else 0
    num_constraints = len(constraints) if constraints is not None else 0
    tok = TokenizedStructure.get_empty(
        id=struct.id,
        num_chains=len(struct.chains),
        num_tokens=struct.num_tokens,
        num_bonds=struct.num_bonds + struct.num_connections,
        num_sequence_tokens=num_token_seq_tokens,
        num_constraints=num_constraints,
        num_priors=num_priors,
    )

    # ==================================================
    # Fill tokenized structure from input structure
    # ==================================================
    # Fill chain structures
    _insert_chain_structures(tok, struct)
    # Fill sequence structures
    _insert_sequence_structures(tok, struct)
    # Fill token structures
    _insert_token_structures(tok, struct, ccd_sequence_dict, chain_atom_dict)
    # Fill atom structures
    _insert_atom_structures(tok, struct, chain_atom_dict, ccd_components, rng, train)
    # Fill bond structures
    _insert_bond_structures(tok, struct, chain_atom_st, g_atom_to_token_map)
    # Fill frame structures
    _insert_frame_structures(tok, struct, chain_atom_dict, ccd_components, train)
    if apo_coords_dict:
        # Fill apo coordinates
        _insert_apo_coordinates(
            tok, struct, apo_coords_dict, ccd_sequence_dict, chain_atom_dict
        )
    if prior_coords is not None and prior_coords.shape[0] > 0:
        # Fill prior coordinates
        _insert_prior_coordinates(tok, prior_coords)
    if constraints:
        # Fill constraint structures
        _insert_constraint_structures(
            tok, struct, constraints, chain_atom_st, g_atom_to_token_map
        )

    # Sanity check
    tok.validate()

    return tok


def _insert_chain_structures(tok: TokenizedStructure, struct: RefStructure):
    """Insert chain info into tokenized structure."""
    tok.chain.chain_type[:] = [c.chain_type for c in struct.chains]
    tok.chain.entity_id[:] = [c.entity_id for c in struct.chains]
    tok.chain.asym_id[:] = [c.asym_id for c in struct.chains]
    tok.chain.sym_id[:] = [c.sym_id for c in struct.chains]
    tok.chain.num_residues[:] = [c.num_residues for c in struct.chains]
    tok.chain.num_atoms[:] = [c.num_atoms for c in struct.chains]
    tok.chain.num_tokens[:] = [c.num_tokens for c in struct.chains]


def _insert_sequence_structures(tok: TokenizedStructure, struct: RefStructure):
    """Insert sequence info into tokenized structure."""
    g_seq_i = 0
    for c in struct.chains:
        n_res = c.num_residues
        n_seq = n_res + 2  # +2 for start and end tokens
        st, end = g_seq_i, g_seq_i + n_seq
        g_seq_i += n_seq

        tok.sequence.pos_id[st:end] = np.arange(n_seq)
        tok.sequence.entity_id[st:end] = c.entity_id
        tok.sequence.chain_type[st:end] = c.chain_type

        # Add bos/eos tokens
        tok.sequence.seq_token_id[st] = C.sequence.BOS_TOKEN_INDEX
        tok.sequence.seq_token_id[end - 1] = C.sequence.EOS_TOKEN_INDEX
        # Insert sequence tokens
        if c.is_polymer:
            _fn = sequence_encode_fn[c.ctype]
            seq = c.get_sequence(map_to_standard=True)
            tok.sequence.seq_token_id[st + 1 : end - 1] = _fn(seq)
        else:
            # For non-polymer chains, set sequence tokens to UNK
            tok.sequence.seq_token_id[st + 1 : end - 1] = C.sequence.UNK_TOKEN_INDEX


def _insert_token_structures(
    tok: TokenizedStructure,
    struct: RefStructure,
    ccd_sequence_dict: dict[int, list[str]],
    all_atom_dict: dict[int, list[str]],
):
    """Insert token info into tokenized structure."""
    # Set token index (0-based)
    tok.token.token_index[:] = np.arange(struct.num_tokens, dtype=np.int64)

    g_seq_i, g_tok_i = 0, 0
    for c in struct.chains:
        asym_id = c.asym_id
        ccd_sequence: list[str] = ccd_sequence_dict[asym_id]
        all_atom_names: list[str] = all_atom_dict[asym_id]

        # Update sequence token index for the chain
        seq_token_st: int = g_seq_i
        g_seq_i += c.num_residues + 2  # +2 for bos/eos

        # Insert chain-level token info
        ntokens = c.num_tokens
        _st, _end = g_tok_i, g_tok_i + ntokens
        tok.token.chain_type[_st:_end] = c.chain_type
        tok.token.entity_id[_st:_end] = c.entity_id
        tok.token.asym_id[_st:_end] = c.asym_id
        tok.token.sym_id[_st:_end] = c.sym_id
        del _st, _end, ntokens

        # Iterate residues in the chain and fill token and some atom info
        for res_i in range(c.num_residues):
            res_idx = res_i + 1  # 1-based index
            seq_token_idx = seq_token_st + 1 + res_i  # +1 for start token

            # Get residue info
            ccd_name = ccd_sequence[res_i]
            res_name = C.residue.get_residue_name_with_unk(ccd_name, c.ctype)
            restype: int = res_name.value

            center_atom_idx: int = C.atom.CENTER_ATOM_INDEX[res_name]
            repr_atom_idx: int = C.atom.PSEUDO_BETA_ATOM_INDEX[res_name]

            # Get atom info
            atom_names = all_atom_names[c.residue.get_atom_slice(res_idx)]
            natoms = len(atom_names)

            if c.residue.is_standard[res_i]:
                # Standard protein/dna/rna residues
                assert c.ctype.is_polymer, "Only polymer residues can be standard."
                tok.token.res_type[g_tok_i] = restype
                tok.token.is_standard[g_tok_i] = True
                tok.token.residue_index[g_tok_i] = res_idx
                tok.token.seq_token_index[g_tok_i] = seq_token_idx
                tok.token.num_atoms[g_tok_i] = natoms
                tok.token.center_index[g_tok_i] = C.atom.CENTER_ATOM_INDEX[res_name]
                tok.token.repr_index[g_tok_i] = C.atom.PSEUDO_BETA_ATOM_INDEX[res_name]

                # Update atom existence mask
                tok.atom.pad_mask[g_tok_i, :natoms] = True

                # Update global token index
                g_tok_i += 1
            else:
                # Ligands, Modifications, Covalent inhibitors
                st = g_tok_i
                end = g_tok_i + natoms
                tok.token.res_type[st:end] = restype
                tok.token.is_standard[st:end] = False
                tok.token.residue_index[st:end] = res_idx
                tok.token.seq_token_index[st:end] = seq_token_idx
                tok.token.num_atoms[st:end] = 1
                # For non-standard residues, we set center and repr indices to itself.
                tok.token.center_index[st:end] = 0
                tok.token.repr_index[st:end] = 0

                # Update atom existence mask
                tok.atom.pad_mask[st:end, 0] = True

                # Update global token index
                g_tok_i += natoms


def _insert_atom_structures(
    tok: TokenizedStructure,
    struct: RefStructure,
    chain_atom_dict: dict[int, list[str]],
    ccd_components: dict[tuple[int, int], Component],
    rng: np.random.Generator,
    train: bool,
):
    g_tok_i = 0
    for c in struct.chains:
        asym_id = c.asym_id
        st, end = g_tok_i, g_tok_i + c.num_tokens

        # Valid atom mask for the chain
        m = tok.atom.pad_mask[st:end]  # (chain_tokens, 24)
        assert m.sum() == c.num_atoms, "Number of valid atoms does not match"

        # Insert atom index
        tok.atom.atom_index[st:end][m] = np.arange(c.num_atoms)

        # Insert ground-truth coordinates
        tok.atom.label_coords[st:end][m] = c.atom.coords

        # Insert reference atom info except conformers
        tok.atom.ref_atom_name_chars[st:end][m] = [
            C.atom.encode_atom_name(name) for name in c.atom.name.tolist()
        ]  # (num_atoms, 4)
        tok.atom.ref_element[st:end][m] = c.atom.element
        tok.atom.ref_charge[st:end][m] = np.nan_to_num(c.atom.charge, nan=0.0)

        # Insert reference molecular conformers
        all_atom_names: list[str] = chain_atom_dict[asym_id]
        for res_i in range(c.num_residues):
            res_idx = res_i + 1  # 1-based index
            is_standard = c.residue.is_standard[res_i]
            ref_comp: Component = ccd_components[(asym_id, res_idx)]

            # Get reference conformer positions with random augmentation
            ref_pos: np.ndarray = ref_comp.get_ref_conformer(rng, train)

            # Insert coordinates based on atom names
            atom_slices = c.residue.get_atom_slice(res_idx)
            atom_names = all_atom_names[atom_slices]
            atom_indices: list[int] = ref_comp.get_atom_indices(atom_names)
            natoms = len(atom_names)
            ref_pos = ref_pos[atom_indices, :]
            ref_mask = get_mask(ref_pos)

            if ref_mask.any():
                # Apply random augmentation to reference positions
                ref_pos = center_random_augmentation(
                    ref_pos, ref_mask, rng=rng, mask_to_zero=False
                )

            if is_standard:
                # Standard residue (one token)
                tok.atom.ref_pos[g_tok_i, :natoms, :] = ref_pos
                tok.atom.atom_type[g_tok_i, :natoms] = [
                    C.atom.atom_name_str_to_index[an] for an in atom_names
                ]
                g_tok_i += 1
            else:
                # Non-standard residue (multiple tokens, one per atom)
                _st, _end = g_tok_i, g_tok_i + natoms
                tok.atom.ref_pos[_st:_end, 0, :] = ref_pos
                # Set single-token atom types for non-standard residues
                tok.atom.atom_type[_st:_end, 0] = C.atom.num_atom_types
                g_tok_i += natoms
                del _st, _end

    # Update atom masks at once
    tok.atom.ref_mask[:] = get_mask(tok.atom.ref_pos)
    tok.atom.resolved_mask[:] = get_mask(tok.atom.label_coords)

    # Update holo coordinates (centering while keeping NaN for unresolved atoms)
    tok.atom.label_coords[:] = do_centering(
        tok.atom.label_coords.reshape(-1, 3),
        tok.atom.resolved_mask.reshape(-1),
        mask_to_zero=False,
    ).reshape(tok.atom.label_coords.shape)


def _insert_bond_structures(
    tok: TokenizedStructure,
    struct: RefStructure,
    chain_atom_st: dict[int, int],
    g_atom_to_token_map: dict[int, tuple[int, int]],
):
    asym_id_to_chain: dict[int, Chain] = {c.asym_id: c for c in struct.chains}

    # First iterate intra-chain bonds
    g_bond_i = 0
    for chain in struct.chains:
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

            assert tok.token.is_ligand[g_tok_i1] and tok.token.is_ligand[g_tok_i2], (
                "Intra-chain bonds should only exist within ligand chains."
            )

            # Insert bond info
            tok.bond.asym_id[g_bond_i, :] = asym_id
            tok.bond.token_index[g_bond_i] = (g_tok_i1, g_tok_i2)
            tok.bond.atom_index[g_bond_i] = (local_atom1, local_atom2)
            tok.bond.bond_type[g_bond_i] = bondtype.value

            # Update global bond index
            g_bond_i += 1

    # Then iterate cross-chain bonds
    for conn_i in range(struct.num_connections):
        asym_id1, asym_id2 = struct.connections[conn_i].asym_id
        ridx1, ridx2 = struct.connections[conn_i].residue_index
        atom1, atom2 = struct.connections[conn_i].atom_names
        bondtype = C.ConnectionType.INTERMOLECULAR

        # Find chain
        chain1 = asym_id_to_chain[asym_id1]
        chain2 = asym_id_to_chain[asym_id2]

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
        tok.bond.asym_id[g_bond_i] = (asym_id1, asym_id2)
        tok.bond.token_index[g_bond_i] = (g_tok_i1, g_tok_i2)
        tok.bond.atom_index[g_bond_i] = (local_atom1, local_atom2)
        tok.bond.bond_type[g_bond_i] = bondtype.value

        # Update global bond index
        g_bond_i += 1


def _insert_frame_structures(
    tok: TokenizedStructure,
    struct: RefStructure,
    chain_atom_dict: dict[int, list[str]],
    ccd_components: dict[tuple[int, int], Component],
    train: bool,
):
    g_tok_i = 0
    for chain in struct.chains:
        ctype = chain.ctype
        asym_id = chain.asym_id
        all_atom_names: list[str] = chain_atom_dict[asym_id]

        if ctype.is_polymer:
            a_n, b_n, c_n = map(str, C.atom.CHAIN_FRAME_ATOMS[ctype])

        # Iterate residues in the chain and fill token and some atom info
        for res_i in range(chain.num_residues):
            res_idx = res_i + 1  # 1-based index
            atom_names = all_atom_names[chain.residue.get_atom_slice(res_idx)]
            natoms = len(atom_names)
            if ctype.is_polymer:
                if not all(an in atom_names for an in (a_n, b_n, c_n)):
                    # If any of the frame atoms are missing, skip frame assignment
                    a_i = b_i = c_i = -1
                else:
                    # Set frame token/atom index
                    a_i, b_i, c_i = map(lambda v: atom_names.index(v), (a_n, b_n, c_n))

                if chain.residue.is_standard[res_i]:
                    if a_i == -1:
                        a_ti = b_ti = c_ti = -1
                        a_ai = b_ai = c_ai = -1
                    else:
                        a_ti, b_ti, c_ti = g_tok_i, g_tok_i, g_tok_i
                        a_ai, b_ai, c_ai = a_i, b_i, c_i
                    tok.token.frame_token_index[g_tok_i] = (a_ti, b_ti, c_ti)
                    tok.token.frame_atom_index[g_tok_i] = (a_ai, b_ai, c_ai)
                    g_tok_i += 1
                else:
                    # For non-standard polymer residues, one atoms per token.
                    st, end = g_tok_i, g_tok_i + natoms
                    if a_i == -1:
                        a_ti = b_ti = c_ti = -1
                        a_ai = b_ai = c_ai = -1
                    else:
                        a_ti, b_ti, c_ti = st + a_i, st + b_i, st + c_i
                        a_ai, b_ai, c_ai = 0, 0, 0
                    tok.token.frame_token_index[st:end] = (a_ti, b_ti, c_ti)
                    tok.token.frame_atom_index[st:end] = (a_ai, b_ai, c_ai)
                    g_tok_i += natoms
            else:
                # For ligand, use the closest atoms for each atom (=token)
                st, end = g_tok_i, g_tok_i + natoms

                ref_comp: Component = ccd_components[(asym_id, res_idx)]
                atom_indices: list[int] = ref_comp.get_atom_indices(atom_names)
                _rng = np.random.default_rng(DETERMINISTIC_FRAME_SEED)
                ref_pos: np.ndarray = ref_comp.get_ref_conformer(_rng, train)
                ref_pos = ref_pos[atom_indices, :]

                ref_mask = tok.atom.ref_mask[st:end, 0]
                if ref_mask.sum() < 3:
                    # If less than 3 valid ref atoms, skip frame assignment.
                    tok.token.frame_token_index[st:end] = -1
                    tok.token.frame_atom_index[st:end] = -1
                    g_tok_i += natoms
                    continue

                for i in range(natoms):
                    if not ref_mask[i]:
                        # If the atom itself is invalid, skip frame assignment.
                        a_ti = b_ti = c_ti = -1
                        a_ai = b_ai = c_ai = -1
                    else:
                        # Find the closest atoms to the current atom.
                        b_x = ref_pos[i]  # (3,)
                        dists = np.linalg.norm(ref_pos - b_x, axis=-1)
                        dists[i] = np.inf  # Mask out itself
                        dists[~ref_mask] = np.inf  # Mask out invalid atoms

                        # Get the indices of the two closest atoms (a and c)
                        a_i, c_i = np.argsort(dists)[:2]
                        a_x, c_x = ref_pos[a_i], ref_pos[c_i]

                        # Check these three atoms are not collinear (<25 degree)
                        v_ab = a_x - b_x
                        v_cb = c_x - b_x
                        cos_angle = np.dot(v_ab, v_cb) / (
                            np.linalg.norm(v_ab) * np.linalg.norm(v_cb) + 1e-8
                        )
                        if abs(cos_angle) > COLLISION_ANGLE_CUTOFF:
                            # If collinear, skip frame assignment for this atom.
                            a_ti = b_ti = c_ti = -1
                            a_ai = b_ai = c_ai = -1
                        else:
                            # Otherwise, assign frame tokens/atoms.
                            a_ti, b_ti, c_ti = st + a_i, g_tok_i, st + c_i
                            a_ai, b_ai, c_ai = 0, 0, 0
                    tok.token.frame_token_index[g_tok_i] = (a_ti, b_ti, c_ti)
                    tok.token.frame_atom_index[g_tok_i] = (a_ai, b_ai, c_ai)
                    g_tok_i += 1


def _insert_apo_coordinates(
    tok: TokenizedStructure,
    struct: RefStructure,
    apo_coords_dict: dict[int, np.ndarray],
    ccd_sequence_dict: dict[int, list[str]],
    chain_atom_dict: dict[int, list[str]],
):
    g_tok_i = 0
    for c in struct.chains:
        if not c.is_protein:
            g_tok_i += c.num_tokens
            continue

        ccd_sequence: list[str] = ccd_sequence_dict[c.asym_id]
        all_atom_names: list[str] = chain_atom_dict[c.asym_id]
        st, end = g_tok_i, g_tok_i + c.num_tokens
        m = tok.atom.pad_mask[st:end]  # (chain_tokens, 24)

        # Insert apo coordinates if available
        if c.is_protein and c.entity_id in apo_coords_dict:
            _coords = apo_coords_dict[c.entity_id]
            tok.atom.apo_coords[st:end][m] = _coords

        # Iterate residues in the chain and fill token and some atom info
        for res_i in range(c.num_residues):
            res_idx = res_i + 1  # 1-based index
            # Get residue info
            ccd_name = ccd_sequence[res_i]
            res_name = C.residue.get_residue_name_with_unk(ccd_name, c.ctype)
            is_standard = c.residue.is_standard[res_i]

            if is_standard:
                center_idx = C.atom.CENTER_ATOM_INDEX[res_name]
                repr_idx = C.atom.PSEUDO_BETA_ATOM_INDEX[res_name]
                frame_indices = PROTEIN_FRAME_ATOM_INDICES[res_name]
                center_coords = tok.atom.apo_coords[g_tok_i, center_idx]
                repr_coords = tok.atom.apo_coords[g_tok_i, repr_idx]
                frame_coords = tok.atom.apo_coords[g_tok_i, frame_indices]

                tok.token.apo_center_coords[g_tok_i] = center_coords
                tok.token.apo_repr_coords[g_tok_i] = repr_coords
                tok.token.apo_frame_coords[g_tok_i] = frame_coords
                g_tok_i += 1

            else:
                atom_names = all_atom_names[c.residue.get_atom_slice(res_idx)]
                atom_index = {n: i for i, n in enumerate(atom_names)}
                atom_coords = {
                    an: tok.atom.apo_coords[g_tok_i + atom_index[an], 0]
                    if an in atom_names
                    else np.full(3, np.nan)
                    for an in ["N", "CA", "C", "CB"]
                }
                center_coords = atom_coords["CA"]
                repr_coords = atom_coords["CB"]
                frame_coords = np.stack([atom_coords[an] for an in ["N", "CA", "C"]])

                # Modifications
                _st = g_tok_i
                _end = g_tok_i + len(atom_names)
                tok.token.apo_center_coords[_st:_end] = center_coords
                tok.token.apo_repr_coords[_st:_end] = repr_coords
                tok.token.apo_frame_coords[_st:_end] = frame_coords
                g_tok_i = _end

    # Update apo masks at once
    tok.atom.apo_mask[:] = get_mask(tok.atom.apo_coords)
    tok.token.apo_center_mask[:] = get_mask(tok.token.apo_center_coords)
    tok.token.apo_repr_mask[:] = get_mask(tok.token.apo_repr_coords)
    tok.token.apo_frame_mask[:] = get_mask(tok.token.apo_frame_coords).all(-1)


def _insert_prior_coordinates(tok: TokenizedStructure, prior_coords: np.ndarray):
    m = tok.atom.pad_mask
    if prior_coords.shape[1] != m.sum():
        raise ValueError(
            f"Prior coordinates have {prior_coords.shape[1]} atoms, "
            f"but expected {m.sum()}."
        )
    if not np.isfinite(prior_coords).all():
        raise ValueError("Prior coordinates contain non-finite values.")

    prior_coords = prior_coords.transpose(1, 0, 2)  # [num_atoms, num_priors, 3]
    tok.atom.prior_coords[tok.atom.pad_mask] = prior_coords


def _insert_constraint_structures(
    tok: TokenizedStructure,
    struct: RefStructure,
    constraints: list[Constraint],
    chain_atom_st: dict[int, int],
    g_atom_to_token_map: dict[int, tuple[int, int]],
):
    asym_id_to_chain: dict[int, Chain] = {c.asym_id: c for c in struct.chains}

    for cond_i, _cond in enumerate(constraints):
        asym_id1, asym_id2 = _cond.asym_id
        ridx1, ridx2 = _cond.residue_index
        atom1, atom2 = _cond.atom_name

        # Find chain
        chain1 = asym_id_to_chain[asym_id1]
        chain2 = asym_id_to_chain[asym_id2]

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
        tok.constraint.asym_id[cond_i] = (asym_id1, asym_id2)
        tok.constraint.token_index[cond_i] = (g_tok_i1, g_tok_i2)
        tok.constraint.atom_index[cond_i] = (local_atom1, local_atom2)
        tok.constraint.lower_bound[cond_i] = _cond.lower_bound
        tok.constraint.upper_bound[cond_i] = _cond.upper_bound
