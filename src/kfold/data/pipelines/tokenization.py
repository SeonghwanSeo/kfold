"""Tokenization pipeline for structures."""

import numpy as np
from rdkit import Chem

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.utils.geometry.random_augment import center_random_augmentation, do_centering
from kfold.utils.misc import spawn_rng


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
        input: RefStructure,
        rng: np.random.Generator | None = None,
        *,
        num_priors: int = 0,
    ) -> TokenizedStructure:
        """Tokenize structure.

        Parameters
        ----------
        input : RefStructure
            The input structure.
        rng : np.random.Generator, optional
            Random number generator for stochastic processes, by default None.
        num_priors : int, optional
            Number of prior conformers to include, by default 0.

        Returns
        -------
        struct: TokenizedStructure
            The parsed tokenized structure.
        """
        return self.tokenize(input, rng, num_priors=num_priors)

    def tokenize(
        self,
        input: RefStructure,
        rng: np.random.Generator | None = None,
        *,
        num_priors: int = 0,
    ) -> TokenizedStructure:
        """Tokenize structure.

        Parameters
        ----------
        input : RefStructure
            The input structure.
        rng : np.random.Generator, optional
            Random number generator for stochastic processes, by default None.
        num_priors : int, optional
            Number of prior conformers to include, by default 0.

        Returns
        -------
        struct: TokenizedStructure
            The parsed tokenized structure.
        """
        return tokenize_structure(
            input, self.ccd, rng, train=self.train, num_priors=num_priors
        )


def tokenize_structure(
    input: RefStructure,
    ccd: CCD,
    rng: np.random.Generator | None = None,
    *,
    train: bool = False,
    num_priors: int = 0,
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
    train : bool, optional
        Whether in training mode, by default False.
    num_priors : int, optional
        Number of prior conformers to include, by default 0.

    Returns
    -------
    struct: TokenizedStructure
        The parsed tokenized structure.
    """
    # Create new rng for this sampling to avoid affecting global state
    rng = spawn_rng(rng)

    ccd_dict: dict[str, Component] = {}
    ccd_smi_dict: dict[str, Component] = {}

    def get_ccd_component(ccd_name: str) -> Component:
        """Get CCD component with caching.

        NOTE (SeonghwanSeo): CCD.__getitem__ deserialize the Component,
        which is time-consuming. Therefore, we cache the Component objects here.
        The cache is removed when the function is terminated.
        """
        if ccd_name not in ccd_dict:
            ccd_dict[ccd_name] = ccd[ccd_name]
        return ccd_dict[ccd_name]

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

    visited_entity_ids = set()
    num_seq_tokens = 0
    for chain in input.chains:
        entity_id = chain.entity_id
        if entity_id not in visited_entity_ids:
            visited_entity_ids.add(entity_id)
            num_seq_tokens += chain.num_residues + 2  # add start and end tokens
    del visited_entity_ids

    # ==================================================
    # Create empty tokenized structure
    # ==================================================
    num_bonds = input.num_bonds + input.num_connections
    struct = TokenizedStructure.get_empty(
        id=input.id,
        num_chains=len(input.chains),
        num_tokens=input.num_tokens,
        num_bonds=num_bonds,
        num_sequence_tokens=num_seq_tokens,
        num_priors=num_priors,
    )

    # ==================================================
    # Collect all component in the structure
    # ==================================================
    ccd_components: dict[tuple[int, int], Component] = {}  # key: (asym_id, res_idx)
    ccd_sequence_dict: dict[int, list[str]] = {}
    all_atom_dict: dict[int, list[str]] = {}
    for chain in input.chains:
        asym_id: int = chain.asym_id
        smiles: str | None = chain.smiles
        ccd_sequence: list[str] = chain.get_ccd_sequence()
        ccd_sequence_dict[asym_id] = ccd_sequence
        all_atom_dict[asym_id] = chain.atom.name.tolist()

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
    # Fill sequence structures
    # ==================================================
    bos_token = C.sequence.BOS_TOKEN_INDEX
    eos_token = C.sequence.EOS_TOKEN_INDEX
    unk_token = C.sequence.UNK_TOKEN_INDEX
    entity_sequence_start: dict[int, int] = {}
    g_seq_i = 0
    for chain in input.chains:
        ctype = chain.ctype
        entity_id = chain.entity_id
        if entity_id in entity_sequence_start:
            continue
        entity_sequence_start[entity_id] = g_seq_i

        # Insert sequence info
        n_res = chain.num_residues
        st = g_seq_i
        end = g_seq_i + n_res + 2  # +2 for start and
        g_seq_i = end

        struct.sequence.pos_id[st:end] = np.arange(n_res + 2)
        struct.sequence.entity_id[st:end] = entity_id
        struct.sequence.chain_type[st:end] = chain.chain_type

        # Add bos/eos tokens
        struct.sequence.seq_token_id[st] = bos_token
        struct.sequence.seq_token_id[end - 1] = eos_token

        # Insert sequence tokens
        if ctype.is_polymer:
            if ctype.is_protein:
                encode_fn = C.sequence.encode_protein_sequence
            elif ctype.is_dna:
                encode_fn = C.sequence.encode_dna_sequence
            elif ctype.is_rna:
                encode_fn = C.sequence.encode_rna_sequence
            seq = chain.get_sequence(map_to_standard=True)
            struct.sequence.seq_token_id[st + 1 : end - 1] = encode_fn(seq)
        else:
            # For non-polymer chains, set sequence tokens to UNK
            struct.sequence.seq_token_id[st + 1 : end - 1] = unk_token

    # ==================================================
    # Fill token structures
    # ==================================================
    g_tok_i = 0
    g_atom_i = 0
    # Map from global atom index to (token_index, local_atom_index)
    g_atom_to_token_map: dict[int, tuple[int, int]] = {}

    for chain in input.chains:
        ctype = chain.ctype
        chain_type_i = chain.chain_type
        entity_id = chain.entity_id
        asym_id = chain.asym_id
        sym_id = chain.sym_id
        ccd_sequence: list[str] = ccd_sequence_dict[asym_id]
        all_atom_names: list[str] = all_atom_dict[asym_id]

        seq_token_st: int = entity_sequence_start[chain.entity_id]

        # Iterate residues in the chain and fill token and some atom info
        for res_i in range(chain.num_residues):
            res_idx = res_i + 1  # 1-based index
            seq_token_idx = seq_token_st + 1 + res_i  # +1 for start token

            # Get residue info
            ccd_name: str = ccd_sequence[res_i]
            res_name: C.ResidueName = C.residue.get_residue_name_with_unk(ccd_name, ctype)
            restype: int = res_name.value
            is_res_standard = chain.residue.is_standard[res_i]
            comp: Component = ccd_components[(asym_id, res_idx)]

            # Get atom info
            atom_names = all_atom_names[chain.residue.get_atom_slice(res_idx)]
            natoms = len(atom_names)

            if is_res_standard:
                # Standard protein/dna/rna residues (including ambiguous residues)
                center_atom_idx: int = C.atom.CENTER_ATOM_INDEX[res_name]
                repr_atom_index: int = C.atom.PSEUDO_BETA_ATOM_INDEX[res_name]
                struct.token.chain_type[g_tok_i] = chain_type_i
                struct.token.entity_id[g_tok_i] = entity_id
                struct.token.asym_id[g_tok_i] = asym_id
                struct.token.sym_id[g_tok_i] = sym_id
                struct.token.res_type[g_tok_i] = restype
                struct.token.is_standard[g_tok_i] = True
                struct.token.residue_index[g_tok_i] = res_idx
                struct.token.seq_token_index[g_tok_i] = seq_token_idx
                struct.token.num_atoms[g_tok_i] = natoms
                struct.token.center_index[g_tok_i] = center_atom_idx
                struct.token.repr_index[g_tok_i] = repr_atom_index

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
                struct.token.chain_type[st:end] = chain_type_i
                struct.token.entity_id[st:end] = entity_id
                struct.token.asym_id[st:end] = asym_id
                struct.token.sym_id[st:end] = sym_id
                struct.token.res_type[st:end] = restype
                struct.token.is_standard[st:end] = False
                struct.token.residue_index[st:end] = res_idx
                struct.token.seq_token_index[st:end] = seq_token_idx
                struct.token.num_atoms[st:end] = 1
                struct.token.center_index[st:end] = 0
                struct.token.repr_index[st:end] = 0

                # Update atom existence mask
                struct.atom.pad_mask[st:end, 0] = True

                for _ in range(natoms):
                    # Map global atom index to (token_index, local_atom_index)
                    g_atom_to_token_map[g_atom_i] = (g_tok_i, 0)
                    # Update global token index
                    g_atom_i += 1
                    g_tok_i += 1

    assert g_tok_i == input.num_tokens, "Global token index does not match."
    assert g_atom_i == input.num_atoms, "Global atom index does not match."

    # Set default token index
    struct.token.token_index[:] = np.arange(input.num_tokens, dtype=np.int64)

    # ==================================================
    # Fill atom structures
    # ==================================================
    g_tok_i = 0
    for chain in input.chains:
        ctype = chain.ctype
        asym_id = chain.asym_id
        ccd_sequence: list[str] = ccd_sequence_dict[asym_id]
        all_atom_names: list[str] = all_atom_dict[asym_id]

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
            ref_pos: np.ndarray = ref_comp.get_ref_conformer(rng, train)
            assert ref_pos is not None, "Auto mode always provides a conformer."

            # Insert coordinates based on atom names
            atom_slices = chain.residue.get_atom_slice(res_idx)
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

    # Update atom masks at once
    struct.atom.ref_mask[:] = get_mask(struct.atom.ref_pos)
    struct.atom.resolved_mask[:] = get_mask(struct.atom.label_coords)
    struct.atom.apo_mask[:] = get_mask(struct.atom.apo_coords)
    # Update NaN to zero
    struct.atom.ref_charge[struct.atom.pad_mask] = np.nan_to_num(
        struct.atom.ref_charge[struct.atom.pad_mask], nan=0.0
    )

    # Update holo coordinates (centering while keeping NaN for unresolved atoms)
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
