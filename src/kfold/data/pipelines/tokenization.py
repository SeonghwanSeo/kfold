"""Tokenization pipeline for structures."""

import json
import os
from functools import lru_cache

import numpy as np
from rdkit import Chem

import kfold.constants as C
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.metadata import Metadata
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.data.utils.ligand_interactions import compute_ligand_interaction_types
from kfold.utils.geometry.random_augment import center_random_augmentation, do_centering


def _log_ligand_interaction_error(
    log_path: str,
    metadata: Metadata,
    ccd_name: str,
    asym_id: int,
    chain_name: str,
    smiles: str | None,
    error: Exception,
) -> None:
    record = {
        "id": metadata.id,
        "ccd": ccd_name,
        "asym_id": asym_id,
        "chain_name": chain_name,
        "smiles": smiles,
        "error": repr(error),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "pid": os.getpid(),
    }
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        return


class Tokenizer:
    def __init__(self, ccd: CCD, use_interaction: bool = True) -> None:
        """Tokenizer for structures.

        Parameters
        ----------
        ccd : CCD
            The chemical component dictionary.
        use_interaction : bool, optional
            Whether to compute interaction types, by default True.
        """
        self.ccd: CCD = ccd
        self.use_interaction: bool = use_interaction

    def __call__(
        self,
        input: RefStructure,
        rng: np.random.Generator | None = None,
        use_only_cached_conformers: bool = False,
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

        Returns
        -------
        struct: TokenizedStructure
            The parsed tokenized structure.
        """
        return self.tokenize(input, rng, use_only_cached_conformers)

    def tokenize(
        self,
        input: RefStructure,
        rng: np.random.Generator | None = None,
        use_only_cached_conformers: bool = False,
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

        Returns
        -------
        struct: TokenizedStructure
            The parsed tokenized structure.
        """
        return tokenize_structure(
            input,
            self.ccd,
            rng,
            use_only_cached_conformers,
            use_interaction=self.use_interaction,
        )


def tokenize_structure(
    input: RefStructure,
    ccd: CCD,
    rng: np.random.Generator | None = None,
    use_only_cached_conformers: bool = False,
    use_interaction: bool = True,
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
    use_interaction : bool, optional
        Whether to compute interaction types, by default True.

    Returns
    -------
    struct: TokenizedStructure
        The parsed tokenized structure.
    """

    @lru_cache(maxsize=128)
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

    # Get metadata
    _metadata: Metadata = input.metadata
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
        residues = chain.residue
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
    struct = TokenizedStructure.get_empty(
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
        chain = input.chains[chain_i]
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
                if use_interaction:
                    interaction_indices = C.interaction.get_residue_interaction_type(
                        res_name, chain.chain_type
                    )
                    if interaction_indices:
                        struct.token.interaction_type[
                            g_tok_i, list(interaction_indices)
                        ] = 1

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
                if use_interaction:
                    interaction_indices = C.interaction.get_residue_interaction_type(
                        C.residue.ResidueName.UNK, chain.chain_type
                    )
                    if interaction_indices:
                        struct.token.interaction_type[
                            st:end, list(interaction_indices)
                        ] = 1

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

    # Set default token index
    struct.token.token_index[:] = np.arange(num_tokens, dtype=np.int32)

    # ==================================================
    # Fill atom structures
    # ==================================================
    g_tok_i = 0
    for chain in input.chains:
        ctype = chain.ctype
        # Valid atom mask for the chain
        asym_id = chain.asym_id
        token_st = chain_token_st[asym_id]
        token_end = token_st + chain_token_num[asym_id]
        assert g_tok_i == token_st, "Global token index does not match."
        pad_mask = struct.atom.pad_mask[token_st:token_end]  # (chain_tokens, 24)
        assert pad_mask.sum() == chain.num_atoms, "Number of valid atoms does not match"

        # Insert ground-truth coordinates
        struct.atom.label_coords[token_st:token_end][pad_mask] = chain.atom.coords

        # Insert apo coordinates (NOTE: apo_coords are already center-random-augmented)
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
        chain_meta = _metadata.get_chain_by_asym_id(asym_id)
        for res_i in range(chain.num_residues):
            residue_index = res_i + 1  # 1-based index
            ccd_name = str(chain.residue.name[res_i])
            smiles: str | None = None

            if ccd_name.startswith("LIG"):
                # This residue is from a smiles string, load smiles from metadata
                smiles = chain_meta.smiles
                assert smiles is not None, (
                    "Smiles string not found in metadata for LIG residue."
                )
                assert chain.num_residues == 1, (
                    "Residue with LIG prefix found in chain with multiple residues."
                )
                ref_mol: Component = Component.from_smiles(ccd_name, smiles, num_confs=1)
            else:
                assert ccd_name in ccd, f"Residue name {ccd_name} not found in CCD."
                ref_mol: Component = get_ccd_component(ccd_name)

            # Get reference conformer positions with random augmentation
            ref_pos: np.ndarray = ref_mol.get_conformer(conformer_mode, rng)  # type: ignore
            assert ref_pos is not None, "Auto mode always provides a conformer."
            ref_mask = np.isfinite(ref_pos).all(axis=-1)
            if ref_mask.any():
                ref_pos = center_random_augmentation(ref_pos, ref_mask, rng=rng)

            # Insert coordinates based on atom names
            atom_indices = []
            ref_atom_order: dict[str, int] = ref_mol.get_atom_index_map()
            for atom_i in chain.residue.iter_residue_atoms(residue_index):
                atom_name = str(chain.atom.name[atom_i])
                assert atom_name in ref_atom_order, (
                    f"Atom name {atom_name} not found in reference molecule {ccd_name}."
                )
                ref_atom_i = ref_atom_order[atom_name]
                atom_indices.append(ref_atom_i)
            # Ensure the atom_indices are ascending order
            assert atom_indices == sorted(atom_indices), (
                "Atom indices are not in ascending order."
            )
            natoms = int(chain.residue.num_atoms[res_i])
            if (
                use_interaction
                and not chain.residue.is_standard[res_i]
                and ctype in (C.ChainType.LIGAND, C.ChainType.ION)
            ):
                try:
                    ligand_interactions = compute_ligand_interaction_types(ref_mol.mol)
                except Exception as e:
                    smiles_info = f", smiles={smiles}" if smiles is not None else ""
                    print(
                        "Error computing ligand interactions for "
                        f"{_metadata.id} (ccd={ccd_name}, asym_id={asym_id}, "
                        f"chain_name={chain_meta.chain_name}{smiles_info}): {e}",
                        flush=True,
                    )
                    log_path = os.environ.get("KFO_BAD_LIGAND_LOG")
                    if log_path:
                        _log_ligand_interaction_error(
                            log_path=log_path,
                            metadata=_metadata,
                            ccd_name=ccd_name,
                            asym_id=asym_id,
                            chain_name=chain_meta.chain_name,
                            smiles=smiles,
                            error=e,
                        )
                    if os.environ.get("KFO_SKIP_BAD_LIGANDS", "0") == "1":
                        num_atoms = ref_mol.mol.GetNumAtoms()
                        ligand_interactions = np.zeros(
                            (num_atoms, C.NUM_INTERACTION_TYPES),
                            dtype=np.int8,
                        )
                    else:
                        raise
                st = g_tok_i
                end = g_tok_i + natoms
                struct.token.interaction_type[st:end, :] = ligand_interactions[
                    atom_indices
                ]
            if chain.residue.is_standard[res_i]:
                # Standard residue
                assert np.all(struct.atom.pad_mask[g_tok_i, :natoms]), (
                    "Atom pad mask mismatch for standard residue."
                    f" (g_tok_i={g_tok_i}, natoms={natoms})"
                )
                struct.atom.ref_pos[g_tok_i, :natoms, :] = ref_pos[atom_indices, :]
                g_tok_i += 1
            else:
                # Non-standard residue
                assert np.all(struct.atom.pad_mask[g_tok_i : g_tok_i + natoms, 0]), (
                    "Atom pad mask mismatch for non-standard residue."
                )
                st = g_tok_i
                end = g_tok_i + natoms
                struct.atom.ref_pos[st:end, 0, :] = ref_pos[atom_indices, :]
                g_tok_i += natoms

    # Update atom masks at once
    struct.atom.ref_mask[:] = np.isfinite(struct.atom.ref_pos).all(axis=-1)
    struct.atom.resolved_mask[:] = np.isfinite(struct.atom.label_coords).all(axis=-1)
    struct.atom.apo_mask[:] = np.isfinite(struct.atom.apo_coords).all(axis=-1)

    # Update NaN to zero
    struct.atom.ref_charge[np.isnan(struct.atom.ref_charge)] = 0.0

    # Update apo coordinates (NaN to zero)
    struct.atom.apo_coords[:] = do_centering(
        struct.atom.apo_coords.reshape(-1, 3),
        struct.atom.apo_mask.reshape(-1),
        mask_to_zero=True,
    ).reshape(struct.atom.apo_coords.shape)
    # Update holo coordinates (centering & NaN to zero)
    struct.atom.label_coords[:] = do_centering(
        struct.atom.label_coords.reshape(-1, 3),
        struct.atom.resolved_mask.reshape(-1),
        mask_to_zero=True,
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
                not struct.token.is_standard[g_tok_i1]
                and not struct.token.is_standard[g_tok_i2]
            ), "Intra-chain bonds should only exist between non-standard residues."

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

    return struct
