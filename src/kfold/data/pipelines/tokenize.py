"""Tokenization pipeline for structures."""

import numpy as np
from rdkit import Chem

import kfold.constants as C
from kfold.data import schema, structure, tokenized
from kfold.data.ccd import CCD, Component


def tokenize_structure(
    input: structure.Structure,
    ccd: CCD,
    rng: np.random.Generator | None = None,
) -> tokenized.TokenizedStructure:
    """Tokenize structure.

    Parameters
    ----------
    input : Structure
        The input structure.
    ccd : CCD
        The chemical component dictionary.
    rng : np.random.Generator, optional
        Random number generator for stochastic processes, by default None.

    Returns
    -------
    struct: TokenizedStructure
        The parsed tokenized structure.
    """
    rng = rng or np.random.default_rng()

    # Get metadata
    _metadata: schema.Metadata = input.metadata
    assert len(_metadata.chains) == len(input.chains), (
        "Number of chains in metadata does not match number of chains in structure."
        f" ({len(_metadata.chains)} != {len(input.chains)})"
    )

    # ==================================================
    # Estimate sizes
    # ==================================================
    num_chains = len(input.chains)
    num_residues = 0
    num_tokens = 0
    num_atoms = 0
    num_bonds = 0
    chain_token_num: dict[int, int] = {}
    chain_token_st: dict[int, int] = {}
    chain_atom_num: dict[int, int] = {}
    chain_atom_st: dict[int, int] = {}

    for chain in input.chains:
        # Count tokens in the chain
        num_tokens_in_chain = 0
        residues: structure.Residue = chain.residue
        for res_i in range(len(residues)):
            residue_index = res_i + 1  # 1-based index
            if residues.is_standard[res_i]:
                # Single token for standard residues
                num_tokens_in_chain += 1
            else:
                # One token per atom for non-standard residues
                num_tokens_in_chain += int(residues.num_atoms[res_i])

        chain_token_num[chain.asym_id] = num_tokens_in_chain
        chain_token_st[chain.asym_id] = num_tokens
        chain_atom_num[chain.asym_id] = int(chain.num_atoms)
        chain_atom_st[chain.asym_id] = num_atoms

        num_residues += int(chain.num_residues)
        num_atoms += int(chain.num_atoms)
        num_bonds += int(chain.num_bonds)
        num_tokens += num_tokens_in_chain

    # Add bonds from connections
    num_bonds += input.num_connections

    # ==================================================
    # Create empty tokenized structure
    # ==================================================
    struct = tokenized.TokenizedStructure.get_empty(
        num_chains,
        num_residues,
        num_tokens,
        num_bonds,
        input.metadata,
    )

    # ==================================================
    # Fill chain structures
    # ==================================================

    for chain_i in range(num_chains):
        chain: structure.Chain = input.chains[chain_i]
        # Insert chain info
        struct.chain.chain_type[chain_i] = chain.chain_type
        struct.chain.entity_id[chain_i] = chain.entity_id
        struct.chain.asym_id[chain_i] = chain.asym_id
        struct.chain.sym_id[chain_i] = chain.sym_id
        struct.chain.num_residues[chain_i] = chain.num_residues
        struct.chain.num_atoms[chain_i] = chain.num_atoms
        struct.chain.num_tokens[chain_i] = chain_token_num[chain.asym_id]

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
        # Iterate residues in the chain and fill token and some atom info
        # NOTE: res_index is reindexed per chain
        for res_i in range(chain.num_residues):
            residue_index = res_i + 1  # 1-based index

            # Get residue info
            ccd_name = str(chain.residue.name[res_i])
            res_name: C.ResidueName = C.residue.get_residue_name_with_unk(ccd_name, ctype)
            restype: int = res_name.value
            is_res_standard = chain.residue.is_standard[res_i]

            natoms = int(chain.residue.num_atoms[res_i])
            ntokens = 1 if is_res_standard else natoms

            # Insert additional residue info
            struct.residue.residue_index[g_res_i] = residue_index
            struct.residue.name[g_res_i] = ccd_name
            struct.residue.res_type[g_res_i] = restype
            struct.residue.num_atoms[g_res_i] = natoms
            struct.residue.num_tokens[g_res_i] = ntokens
            struct.residue.is_standard[g_res_i] = is_res_standard

            if is_res_standard:
                # Standard protein/dna/rna residues (including ambiguous residues)
                atom_list: tuple[C.AtomName, ...] = C.atom.RESIDUE_ATOMS[res_name]
                ref_atom: C.AtomName = C.atom.REF_ATOM[res_name]
                beta_atom: C.AtomName = C.atom.PSEUDO_BETA_ATOM[res_name]

                # Insert token info
                struct.token.num_atoms[g_tok_i] = natoms
                struct.token.center_index[g_tok_i] = atom_list.index(ref_atom)
                struct.token.disto_index[g_tok_i] = atom_list.index(beta_atom)
                struct.token.is_standard[g_tok_i] = True

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

    # ==================================================
    # Fill atom structures
    # ==================================================
    g_tok_i = 0
    for chain in input.chains:
        # Valid atom mask for the chain
        asym_id = chain.asym_id
        token_st = chain_token_st[asym_id]
        token_end = token_st + chain_token_num[asym_id]
        assert g_tok_i == token_st, "Global token index does not match."
        pad_mask = struct.atom.pad_mask[token_st:token_end]  # (chain_tokens, 24)
        assert pad_mask.sum() == chain.num_atoms, "Number of valid atoms does not match"

        # Insert ground-truth coordinates
        label_coords = chain.atom.label_coords  # (num_atoms, 3)
        struct.atom.coords[token_st:token_end][pad_mask] = label_coords

        # Insert apo coordinates (randomly sample one if multiple are available)
        if (num_apo := chain.num_apo) > 0:
            apo_idx = rng.integers(0, num_apo)
            apo_coords = chain.atom.apo_coords[apo_idx]  # (num_atoms, 3)
            struct.atom.apo_coords[pad_mask] = apo_coords

        # Insert reference molecules
        for res_i in range(chain.num_residues):
            residue_index = res_i + 1  # 1-based index
            ccd_name = str(chain.residue.name[res_i])

            if ccd_name.startswith("LIG"):
                # This residue is from a smiles string, load smiles from metadata
                smiles = _metadata.get_chain_by_asym_id(asym_id).smiles
                assert smiles is not None, (
                    "Smiles string not found in metadata for LIG residue."
                )
                assert chain.num_residues == 1, (
                    "Residue with LIG prefix found in chain with multiple residues."
                )
                ref_mol: Component = Component.from_smiles(ccd_name, smiles)
            else:
                assert ccd_name in ccd, f"Residue name {ccd_name} not found in CCD."
                ref_mol: Component = ccd[ccd_name]

            ref_atom_order: dict[str, int] = ref_mol.get_atom_index_map()
            ref_atom_name_chars: np.ndarray = np.array(
                [C.atom.encode_atom_name(name) for name in ref_mol.atom_names]
            )  # (num_atoms, 4)
            ref_element: np.ndarray = ref_mol.elements  # (num_atoms,)
            ref_charge: np.ndarray = ref_mol.charges  # (num_atoms,)
            ref_pos: np.ndarray = ref_mol.get_conformer("auto", rng)  # type: ignore
            assert ref_pos is not None, "Auto mode always provides a conformer."

            res_atom_st = int(chain.residue.atom_starts[res_i])
            natoms = int(chain.residue.num_atoms[res_i])

            # get indices of atoms in the reference molecule
            atom_indices = []
            for atom_i in range(res_atom_st, res_atom_st + natoms):
                atom_name = str(chain.atom.name[atom_i])
                assert atom_name in ref_atom_order, (
                    f"Atom name {atom_name} not found in reference molecule {ccd_name}."
                )
                ref_atom_i = ref_atom_order[atom_name]
                atom_indices.append(ref_atom_i)

            if chain.residue.is_standard[res_i]:
                # Standard residue
                assert np.all(struct.atom.pad_mask[g_tok_i, :natoms]), (
                    "Atom pad mask mismatch for standard residue."
                    f" (g_tok_i={g_tok_i}, natoms={natoms})"
                )
                struct.atom.ref_atom_name_chars[g_tok_i, :natoms, :] = (
                    ref_atom_name_chars[atom_indices]
                )
                struct.atom.ref_element[g_tok_i, :natoms] = ref_element[atom_indices]
                struct.atom.ref_charge[g_tok_i, :natoms] = ref_charge[atom_indices]
                struct.atom.ref_pos[g_tok_i, :natoms, :] = ref_pos[atom_indices, :]
                g_tok_i += 1
            else:
                # Non-standard residue
                assert np.all(struct.atom.pad_mask[g_tok_i : g_tok_i + natoms, 0]), (
                    "Atom pad mask mismatch for non-standard residue."
                )
                st = g_tok_i
                end = g_tok_i + natoms
                struct.atom.ref_atom_name_chars[st:end, 0, :] = ref_atom_name_chars[
                    atom_indices
                ]
                struct.atom.ref_element[st:end, 0] = ref_element[atom_indices]
                struct.atom.ref_charge[st:end, 0] = ref_charge[atom_indices]
                struct.atom.ref_pos[st:end, 0, :] = ref_pos[atom_indices, :]
                g_tok_i += natoms

    # Update atom masks at once
    struct.atom.ref_mask[:] = np.isfinite(struct.atom.ref_pos).all(axis=-1)
    struct.atom.resolved_mask[:] = np.isfinite(struct.atom.coords).all(axis=-1)
    struct.atom.apo_mask[:] = np.isfinite(struct.atom.apo_coords).all(axis=-1)

    # ==================================================
    # Fill bond structures
    # ==================================================
    # First iterate intra-chain bonds
    g_bond_i = 0
    for chain_i in range(num_chains):
        chain = input.chains[chain_i]
        asym_id = chain.asym_id
        for bond_i in range(chain.num_bonds):
            res1, res2 = chain.bond.residue_index[bond_i]
            atom1, atom2 = chain.bond.atom_index[bond_i]
            bondtype: C.ConnectionType = C.bond.bond_type_to_connection_type(
                Chem.BondType.values[int(chain.bond.bond_type[bond_i])]
            )

            # Map chain-local atom indices to global atom indices
            g_atom1 = chain_atom_st[asym_id] + int(atom1)
            g_atom2 = chain_atom_st[asym_id] + int(atom2)

            # Map atom indices to token and atom indices
            token1, local_atom1 = g_atom_to_token_map[g_atom1]
            token2, local_atom2 = g_atom_to_token_map[g_atom2]

            assert (
                not struct.token.is_standard[token1]
                and not struct.token.is_standard[token2]
            ), "Intra-chain bonds should only exist between non-standard residues."

            # Insert bond info
            struct.bond.asym_id[g_bond_i, :] = asym_id
            struct.bond.token_index[g_bond_i] = (token1, token2)
            struct.bond.atom_index[g_bond_i] = (local_atom1, local_atom2)
            struct.bond.bond_type[g_bond_i] = bondtype.value

            # Update global bond index
            g_bond_i += 1

    # Then iterate cross-chain bonds
    for conn_i in range(input.num_connections):
        res1, res2 = input.connections[conn_i].residue_index
        atom1, atom2 = input.connections[conn_i].atom_names
        asymid1, asymid2 = input.connections[conn_i].asym_id
        bondtype = C.ConnectionType.INTERMOLECULAR

        # Find atom index
        chain1 = input.get_chain_by_asym_id(asymid1)
        chain2 = input.get_chain_by_asym_id(asymid2)

        for a_i in chain1.residue.iter_residue_atoms(res1):
            if str(chain1.atom.name[a_i]) == atom1:
                atom1 = a_i
                break
        else:
            raise ValueError(
                f"Atom name {atom1} not found in residue {res1} of chain {asymid1}."
            )
        for a_i in chain2.residue.iter_residue_atoms(res2):
            if str(chain2.atom.name[a_i]) == atom2:
                atom2 = a_i
                break
        else:
            raise ValueError(
                f"Atom name {atom2} not found in residue {res2} of chain {asymid2}."
            )

        # Map chain-local atom indices to global atom indices
        g_atom1 = chain_atom_st[asymid1] + int(atom1)
        g_atom2 = chain_atom_st[asymid2] + int(atom2)

        # Map atom indices to token and atom indices
        token1, local_atom1 = g_atom_to_token_map[g_atom1]
        token2, local_atom2 = g_atom_to_token_map[g_atom2]

        # Insert bond info
        struct.bond.asym_id[g_bond_i] = (asymid1, asymid2)
        struct.bond.token_index[g_bond_i] = (token1, token2)
        struct.bond.atom_index[g_bond_i] = (local_atom1, local_atom2)
        struct.bond.bond_type[g_bond_i] = bondtype.value

        # Update global bond index
        g_bond_i += 1

    return struct
