import numpy as np

import kfold.constants as C
from kfold.data import metadata, tokenized
from kfold.utils.errors import BoltzDataProcessingError

from .structure import BoltzStructure

# TODO: Move to connections to constraint.


def centering(
    coords: np.ndarray, mask: np.ndarray, mask_to_zero: bool = True
) -> np.ndarray:
    """Center coordinates based on the masked mean position.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates tensor of shape (N, ..., 3).
    mask : np.ndarray
        The boolean mask tensor of shape (N,).
    mask_to_zero : bool, optional
        If True, positions where mask is False will be set to zero after centering.

    Returns
    -------
    np.ndarray
        The centered coordinates tensor of shape (N, ..., 3).
    """
    if not mask.any():
        return coords
    masked_coords = coords[mask]
    center_pos = masked_coords.mean(axis=0, keepdims=True)
    centered_coords = coords - center_pos
    if mask_to_zero:
        centered_coords[~mask] = 0.0
    return centered_coords  # type: ignore


def parse_record(record_json: dict) -> metadata.Metadata:
    """Parse metadata record from JSON dictionary.

    Parameters
    ----------
    record_json : dict
        The metadata record in JSON format.

    Returns
    -------
    Metadata
        The parsed metadata object.
    """
    pdb_id = record_json["id"]
    exp_record = metadata.ExperimentRecord(
        pdb_id=record_json["id"], **record_json["structure"]
    )

    asym_id = 1  # starts from 1
    chain_infos = []
    for cinfo in record_json["chains"]:
        cinfo_meta = metadata.ChainInfo(
            chain_type=C.chain.ChainType(cinfo["mol_type"]),
            chain_name=cinfo["chain_name"],
            num_residues=cinfo["num_residues"],
            cluster_id=str(cinfo["cluster_id"]),
            valid=cinfo["valid"],
            entity_id=-1,  # dummy value
            asym_id=asym_id,
            sym_id=-1,  # dummy value
        )
        asym_id += 1
        chain_infos.append(cinfo_meta)

    interface_infos = []
    for iinfo in record_json["interfaces"]:
        if iinfo["valid"] is False:
            continue
        asym_id1 = chain_infos[iinfo["chain_1"]].asym_id
        asym_id2 = chain_infos[iinfo["chain_2"]].asym_id
        interface_meta = metadata.InterfaceInfo(
            asym_ids=(asym_id1, asym_id2),
            valid=iinfo["valid"],
            is_bonded=False,
        )
        interface_infos.append(interface_meta)

    return metadata.Metadata(
        id=pdb_id,
        source="rcsb",
        exp=exp_record,
        prediction=None,
        chains=chain_infos,
        interfaces=interface_infos,
    )


def tokenize_structure(structure: BoltzStructure) -> tokenized.TokenizedStructure:
    """Tokenize structure.

    Parameters
    ----------
    structure : BoltzStructure
        The BoltzStructure object.

    Returns
    -------
    FoldingInput
        The parsed structure layout.

    """

    # Define field types
    token_field_dtype: dict[str, type] = {
        "res_type": np.int8,  # 0-31
        "chain_type": np.int8,  # 0-3
        "entity_id": np.int16,
        "asym_id": np.int16,
        "sym_id": np.int16,
        "token_index": np.int32,
        "residue_index": np.int32,
        "disto_index": np.int8,  # 0-23
        "center_index": np.int8,  # 0-23
        "num_atoms": np.int8,  # 0-23
        "resolved_mask": np.bool_,
        "is_standard": np.bool_,
    }
    atom_field_dtype: dict[str, type] = {
        "ref_charge": np.float16,
        "ref_element": np.int8,
        "ref_atom_name_chars": np.int8,  # 0-63
        "ref_pos": np.float32,
        "coords": np.float32,
        "apo_coords": np.float32,
        "resolved_mask": np.bool_,
    }
    bond_field_dtype: dict[str, type] = {
        "asym_id": np.int16,
        "token_index": np.int32,
        "atom_index": np.int8,  # 0-23
        "bond_type": np.int8,
    }

    if not structure.mask.any():
        raise BoltzDataProcessingError("No valid chains in the structure.")

    chains = structure.chains[structure.mask]

    # Shape: [Nchain, 24]
    chain_info: dict[str, np.ndarray] = {
        "chain_type": chains["mol_type"].astype(np.uint8),
        "entity_id": chains["entity_id"].astype(np.uint16) + 1,
        "asym_id": chains["asym_id"].astype(np.uint16) + 1,
        "sym_id": chains["sym_id"].astype(np.uint16) + 1,
        "num_residues": chains["res_num"].astype(np.uint32),
        "num_atoms": chains["atom_num"].astype(np.uint32),
        # 'num_tokens' will be computed below
        "num_tokens": np.zeros(len(chains), dtype=np.uint32),
    }

    # Shape: [Ntoken,]
    token_info = {
        "res_type": [],
        "chain_type": [],
        "entity_id": [],
        "asym_id": [],
        "sym_id": [],
        "residue_index": [],
        "disto_index": [],
        "center_index": [],
        "num_atoms": [],
        "resolved_mask": [],
        "is_standard": [],
    }

    # Shape: [Ntoken, 24]
    atom_info = {
        "ref_atom_name_chars": [],
        "ref_element": [],
        "ref_charge": [],
        "ref_pos": [],
        "resolved_mask": [],
        "coords": [],
    }

    # Shape: [Nbond, 24]
    bond_info = {
        "asym_id": [],
        "token_index": [],
        "atom_index": [],
        "bond_type": [],
    }

    atom_index_map: dict[int, tuple[int, int]] = {}  # (token_index, atom_index)

    # === Get token features === #
    global_token_index = 0
    for chain_index, chain in enumerate(chains):
        res_start = chain["res_idx"]
        res_end = res_start + chain["res_num"]

        # Chain indices
        chain_type = chain_info["chain_type"][chain_index]
        asym_id = chain_info["asym_id"][chain_index]
        entity_id = chain_info["entity_id"][chain_index]
        sym_id = chain_info["sym_id"][chain_index]

        # === Iterate residues === #
        # Iterate residues in the chain and fill token and some atom info
        # NOTE: res_index is reindexed per chain
        num_tokens_in_chain = 0
        for i, residue in enumerate(structure.residues[res_start:res_end]):
            # AF3 indexing rule: starts from 1
            chain_res_index = i + 1

            # residue
            is_res_present = residue["is_present"]

            atom_start = residue["atom_idx"]
            num_atoms_in_res = residue["atom_num"]
            atom_end = atom_start + num_atoms_in_res
            residue_atoms = structure.atoms[atom_start:atom_end]

            if residue["is_standard"]:
                # Proteins' amino acid and nucleic acids' base
                res_name = C.residue.ResidueName(str(residue["name"]).strip())
                token_info["res_type"].append(res_name.index)

                # === Insert token info === #
                token_info["chain_type"].append(chain_type)
                token_info["asym_id"].append(asym_id)
                token_info["entity_id"].append(entity_id)
                token_info["sym_id"].append(sym_id)
                token_info["residue_index"].append(chain_res_index)
                token_info["num_atoms"].append(num_atoms_in_res)
                token_info["center_index"].append(residue["atom_center"] - atom_start)
                token_info["disto_index"].append(residue["atom_disto"] - atom_start)
                token_info["resolved_mask"].append(is_res_present)
                token_info["is_standard"].append(True)

                # === Insert atom info === #
                atom_info["ref_atom_name_chars"].append(residue_atoms["name"])
                atom_info["ref_element"].append(residue_atoms["element"])
                atom_info["ref_charge"].append(residue_atoms["charge"])
                atom_info["ref_pos"].append(residue_atoms["conformer"])
                atom_info["resolved_mask"].append(residue_atoms["is_present"])
                atom_info["coords"].append(residue_atoms["coords"])

                # === Add mapping === #
                for j, atom_index in enumerate(range(atom_start, atom_end)):
                    atom_index_map[atom_index] = (global_token_index, j)

                # === Update offset === #
                num_tokens_in_chain += 1
                global_token_index += 1
            else:
                # Ligands, Modifications, Covalent inhibitors
                res_name = C.residue.ResidueName("UNK")  # use unknown residue name
                res_type = res_name.index

                for atom_idx, atom in zip(
                    range(atom_start, atom_end), residue_atoms, strict=True
                ):
                    # === Insert token info === #
                    token_info["chain_type"].append(chain_type)
                    token_info["asym_id"].append(asym_id)
                    token_info["entity_id"].append(entity_id)
                    token_info["sym_id"].append(sym_id)
                    token_info["res_type"].append(res_type)
                    token_info["residue_index"].append(chain_res_index)
                    token_info["disto_index"].append(0)
                    token_info["center_index"].append(0)
                    token_info["num_atoms"].append(1)
                    token_info["resolved_mask"].append(
                        is_res_present & atom["is_present"]
                    )
                    token_info["is_standard"].append(False)

                    # === Insert atom info === #
                    atom_info["ref_atom_name_chars"].append(atom["name"][None])
                    atom_info["ref_element"].append(atom["element"][None])
                    atom_info["ref_charge"].append(atom["charge"][None])
                    atom_info["ref_pos"].append(atom["conformer"][None])
                    atom_info["resolved_mask"].append(atom["is_present"][None])
                    atom_info["coords"].append(atom["coords"][None])

                    # === Add mapping === #
                    atom_index_map[atom_idx] = (global_token_index, 0)

                    # === Update index === #
                    num_tokens_in_chain += 1
                    global_token_index += 1

        # Update number of tokens in the chain
        chain_info["num_tokens"][chain_index] = num_tokens_in_chain

    # === Get bond features === #
    # First iterate bonds (intra-chain)
    for bond in structure.bonds:
        if bond["atom_1"] not in atom_index_map or bond["atom_2"] not in atom_index_map:
            continue

        # Map original atom indices to reindexed atom indices
        token1, atom1 = atom_index_map[bond["atom_1"]]
        token2, atom2 = atom_index_map[bond["atom_2"]]
        bond_type = bond["type"]
        bond_info["token_index"].append((token1, token2))
        bond_info["atom_index"].append((atom1, atom2))
        bond_info["bond_type"].append(bond_type)

        asym_id1, asym_id2 = token_info["asym_id"][token1], token_info["asym_id"][token2]
        bond_info["asym_id"].append((asym_id1, asym_id2))

    # Then iterate cross-chain bonds
    # TODO: Move connections from "bond" to "constraints".
    for bond in structure.connections:
        if bond["atom_1"] not in atom_index_map or bond["atom_2"] not in atom_index_map:
            continue

        # Map original atom indices to reindexed atom indices
        token1, atom1 = atom_index_map[bond["atom_1"]]
        token2, atom2 = atom_index_map[bond["atom_2"]]
        bond_info["token_index"].append((token1, token2))
        bond_info["atom_index"].append((atom1, atom2))
        bond_info["bond_type"].append(C.bond.ConnectionType.SINGLE.value)

        asym_id1, asym_id2 = token_info["asym_id"][token1], token_info["asym_id"][token2]
        bond_info["asym_id"].append((asym_id1, asym_id2))

    # === Convert lists to tensors === #

    def create_atom_array(key: str, lst: list, dtype: type = np.uint16) -> np.ndarray:
        """Create a fixed-size atom array by padding with zeros."""
        if len(lst) == 0:
            raise ValueError(f"No data for atom field: {key}")
        dim = lst[0].shape[1:]  # exclude first dimension
        arr = np.zeros((len(lst), 24, *dim), dtype=dtype)
        for i, v in enumerate(lst):
            arr[i, : v.shape[0]] = v
        return arr

    chain_arr: dict[str, np.ndarray] = chain_info  # already in array format

    token_arr: dict[str, np.ndarray] = {
        key: np.array(value, dtype=token_field_dtype[key])
        for key, value in token_info.items()
    }

    atom_arr: dict[str, np.ndarray] = {
        key: create_atom_array(key, value, dtype=atom_field_dtype[key])
        for key, value in atom_info.items()
    }

    bond_arr = {
        key: np.array(value, dtype=bond_field_dtype[key])
        for key, value in bond_info.items()
    }

    # replace actual apo coords when available
    Ntoken = token_arr["res_type"].shape[0]
    atom_arr["apo_coords"] = np.zeros((Ntoken, 24, 1, 3), dtype=np.float32)
    atom_arr["apo_mask"] = np.zeros((Ntoken, 24, 1), dtype=bool)

    Nc = chain_arr["chain_type"].shape[0]
    Nt = token_arr["res_type"].shape[0]
    Nb = bond_arr["bond_type"].shape[0]
    chain_data = tokenized.Chain(
        chain_type=chain_arr["chain_type"].reshape(Nc),
        entity_id=chain_arr["entity_id"].reshape(Nc),
        asym_id=chain_arr["asym_id"].reshape(Nc),
        sym_id=chain_arr["sym_id"].reshape(Nc),
        num_residues=chain_arr["num_residues"].reshape(Nc),
        num_atoms=chain_arr["num_atoms"].reshape(Nc),
        num_tokens=chain_arr["num_tokens"].reshape(Nc),
    )
    token_data = tokenized.Token(
        res_type=token_arr["res_type"].reshape(Nt),
        chain_type=token_arr["chain_type"].reshape(Nt),
        entity_id=token_arr["entity_id"].reshape(Nt),
        asym_id=token_arr["asym_id"].reshape(Nt),
        sym_id=token_arr["sym_id"].reshape(Nt),
        token_index=np.arange(Nt, dtype=np.uint32),
        residue_index=token_arr["residue_index"].reshape(Nt),
        disto_index=token_arr["disto_index"].reshape(Nt),
        center_index=token_arr["center_index"].reshape(Nt),
        num_atoms=token_arr["num_atoms"].reshape(Nt),
        resolved_mask=token_arr["resolved_mask"].reshape(Nt),
        is_standard=token_arr["is_standard"].reshape(Nt),
    )
    atom_data = tokenized.Atom(
        ref_atom_name_chars=atom_arr["ref_atom_name_chars"].reshape(Nt, 24, 4),
        ref_element=atom_arr["ref_element"].reshape(Nt, 24),
        ref_charge=atom_arr["ref_charge"].reshape(Nt, 24),
        ref_pos=atom_arr["ref_pos"].reshape(Nt, 24, 3),
        coords=atom_arr["coords"].reshape(Nt, 24, -1, 3),
        apo_coords=atom_arr["apo_coords"].reshape(Nt, 24, 1, 3),
        resolved_mask=atom_arr["resolved_mask"].reshape(Nt, 24),
        apo_mask=atom_arr["apo_mask"].reshape(Nt, 24, 1),
    )
    bond_data = tokenized.Bond(
        asym_id=bond_arr["asym_id"].reshape(Nb, 2),
        token_index=bond_arr["token_index"].reshape(Nb, 2),
        atom_index=bond_arr["atom_index"].reshape(Nb, 2),
        bond_type=bond_arr["bond_type"].reshape(Nb),
    )

    return tokenized.TokenizedStructure(
        chain=chain_data,
        token=token_data,
        atom=atom_data,
        bond=bond_data,
    )
