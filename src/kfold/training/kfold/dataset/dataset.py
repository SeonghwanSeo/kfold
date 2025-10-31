from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

import kfold.constants as C
from kfold.data import model_input
from kfold.utils.boltz.structure import BoltzStructure
from kfold.utils.boltz.utils import get_atom_name


class BaseDataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> model_input.FoldingInput: ...


class SafeIterDataset(BaseDataset):
    def __getitem__(self, index: int) -> model_input.FoldingInput:
        return self.get_item_safe(index)

    def get_item_safe(self, index: int) -> model_input.FoldingInput:
        try_indexes = []
        for _ in range(10):
            try:
                return self.get_item(index)
            except Exception as e:
                print(f"Error loading index {index}: {e}. Retrying...")
                index = np.random.randint(0, len(self))
                try_indexes.append(index)
        else:
            raise RuntimeError(
                f"Failed to load data after 10 attempts. Tried indexes: {try_indexes}"
            )

    @abstractmethod
    def get_item(self, index: int) -> model_input.FoldingInput: ...


class BoltzDataset(torch.utils.data.Dataset):
    def __init__(self, boltz_processed_dir: str | Path, keys: list[str]):
        self.boltz_processed_dir: Path = Path(boltz_processed_dir)
        self.keys: list[str] = keys

    def __len__(self) -> int:
        return len(self.keys)

    def get_item(self, index: int) -> model_input.FoldingInput:
        name = self.keys[index]
        path = self.boltz_processed_dir / f"{name}.npz"
        boltz_structure = BoltzStructure.load(path)

        chains = boltz_structure.chains[boltz_structure.mask]

        return self.parse_structure(chains, boltz_structure)

    @staticmethod
    def parse_structure(
        chains: np.ndarray,
        structure: BoltzStructure,
    ) -> FoldingInput:
        """Extract structure layout from BoltzStructure.

        Parameters
        ----------
        chains : np.ndarray (boltz.types.Chain)
            The chain array.
        structure : BoltzStructure
            The BoltzStructure object.

        Returns
        -------
        FoldingInput
            The parsed structure layout.

        """
        chain_info = {
            "chain_type": torch.as_tensor(chains["mol_type"], dtype=torch.int32),
            "entity_id": torch.as_tensor(chains["entity_id"], dtype=torch.int32) + 1,
            "asym_id": torch.as_tensor(chains["asym_id"], dtype=torch.int32),
            "sym_id": torch.as_tensor(chains["sym_id"], dtype=torch.int32),
            "num_residues": torch.as_tensor(chains["res_num"], dtype=torch.int32),
            "num_atoms": torch.as_tensor(chains["atom_num"], dtype=torch.int32),
            # 'num_tokens' will be computed below
            "num_tokens": torch.zeros(len(chains), dtype=torch.int32),
        }

        token_info = {
            "token_type": [],
            "chain_type": [],
            "asym_id": [],
            "sym_id": [],
            "residue_idx": [],
            "disto_idx": [],
            "center_idx": [],
            "resolved_mask": [],
            "cyclic_period": [],
        }

        atom_info = {
            "atom_name": [],
            "element": [],
            "charge": [],
            "token_idx": [],
            "resolved_mask": [],
            "label_coords": [],
        }

        # Since some chains can be masked,
        # we reindex the chain, residue, and atom indices
        chain_idx_map = {}  # starts from 0
        res_idx_map = {}  # starts from 0 for each chain
        atom_idx_map = {}  # shifted for masked chains

        # === Get token features and some atom features === #
        atom_ofs = 0
        token_ofs = 0
        for chain_idx, chain in enumerate(chains):
            res_start = chain["res_idx"]
            res_end = res_start + chain["res_num"]
            chain_idx_map[chain["asym_id"]] = chain_idx

            # Keep track of original atom offset before processing this chain
            original_atom_ofs = atom_ofs

            # === Iterate residues === #
            # Iterate residues in the chain and fill token and some atom info
            # NOTE: res_idx is reindexed per chain
            num_tokens_in_chain = 0
            for res_idx, residue in enumerate(structure.residues[res_start:res_end]):
                # Map original residue index to reindexed residue index
                res_idx_map[residue["res_idx"]] = res_idx

                atom_start = residue["atom_idx"]
                num_atoms_in_res = residue["atom_num"]
                atom_end = atom_start + num_atoms_in_res

                if residue["is_standard"]:
                    # Proteins' amino acid and nucleic acids' base

                    # === Insert token info === #
                    res_name = str(residue["name"]).strip()
                    token_info["token_type"].append(C.residue.ResidueName(res_name).index)
                    token_info["residue_idx"].append(res_idx)
                    # shift atom indices to be relative to the entire structure
                    token_info["disto_idx"].append(
                        atom_ofs + (residue["atom_disto"] - atom_start)
                    )
                    token_info["center_idx"].append(
                        atom_ofs + (residue["atom_center"] - atom_start)
                    )
                    token_info["resolved_mask"].append(residue["is_present"])
                    token_info["cyclic_period"].append(chain["cyclic_period"])

                    # === Insert atom info === #
                    atom_info["token_idx"].extend([token_ofs] * num_atoms_in_res)

                    # === Update offset === #
                    num_tokens_in_chain += 1
                    token_ofs += 1
                    atom_ofs += num_atoms_in_res
                else:
                    # Ligands, Motifications, Covalent inhibitors
                    res_name = "UNK"  # use unknown residue name
                    token_type = C.residue.ResidueName(res_name).index
                    for atom in structure.atoms[atom_start:atom_end]:
                        # === Insert token info === #
                        token_info["residue_idx"].append(res_idx)
                        token_info["token_type"].append(token_type)
                        token_info["disto_idx"].append(atom_ofs)
                        token_info["center_idx"].append(atom_ofs)
                        token_info["resolved_mask"].append(atom["is_present"])
                        token_info["cyclic_period"].append(chain["cyclic_period"])

                        # === Insert atom info === #
                        atom_info["token_idx"].append(token_ofs)

                        # === Update offset === #
                        num_tokens_in_chain += 1
                        token_ofs += 1
                        atom_ofs += 1

            # Update number of tokens in the chain
            chain_info["num_tokens"][chain_idx] = num_tokens_in_chain

        # === Get atom features === #
        atom_ofs = 0
        for chain in chains:
            atom_start = chain["atom_idx"]
            atom_end = atom_start + chain["atom_num"]
            for i, atom_idx in enumerate(range(atom_start, atom_end)):
                # Map original atom index to reindexed atom index
                atom_idx_map[atom_idx] = atom_ofs + i

                # === Insert atom info === #
                atom = structure.atoms[atom_idx]
                atom_info["atom_name"].append(atom["name"])
                atom_info["element"].append(atom["element"])
                atom_info["charge"].append(atom["charge"])
                atom_info["resolved_mask"].append(atom["is_present"])
                atom_info["label_coords"].append(atom["coords"])

            # Update atom offset
            atom_ofs += chain["atom_num"]

        # === Get bond features === #
        # First iterate bonds (intra-chain)
        for bond in structure.bonds:
            # Get features
            atom_1, atom_2, bond_type = bond

            # Get atom features

            # Map original atom indices to reindexed atom indices
            atom_1 = atom_idx_map[atom_1]
            atom_2 = atom_idx_map[atom_2]

        # === Convert lists to tensors === #
        boolean_fields = ["resolved_mask", "pad_mask", "is_pocket"]
        float_fields = ["label_coords", "apo_coords", "charge"]

        def get_dtype(key: str) -> torch.dtype:
            if key in boolean_fields:
                return torch.bool
            elif key in float_fields:
                return torch.float32
            else:
                return torch.int32

        token_info = {
            key: torch.as_tensor(value, dtype=get_dtype(key))
            for key, value in token_info.items()
        }
        atom_info = {
            key: torch.as_tensor(value, dtype=get_dtype(key))
            for key, value in atom_info.items()
        }

        # Add additional fields
        token_info["is_pocket"] = torch.zeros_like(token_info["resolved_mask"])
        atom_info["apo_coords"] = (
            atom_info["label_coords"].clone().unsqueeze(1)
        )  # (N, 1, 3), 1: n_apo

        # Add mask fields
        chain_info["pad_mask"] = torch.ones_like(
            chain_info["chain_type"], dtype=torch.bool
        )
        token_info["pad_mask"] = torch.ones_like(token_info["resolved_mask"])
        atom_info["pad_mask"] = torch.ones_like(atom_info["resolved_mask"])

    @staticmethod
    def parse_atom(
        chains: np.ndarray,
        structure: BoltzStructure,
    ) -> model_input.AtomLayout:
        """Extract atom layout from BoltzStructure.

        Parameters
        ----------
        chain : np.ndarray (boltz.types.Chain)
            The chain array.

        Returns
        -------
        AtomLayout
            The parsed atom layout.
        """
        atom_type_list = []
        element_list = []
        charge_list = []
        is_resolved_list = []
        label_coords_list = []
        token_idx_list = []

        token_ofs = 0
        for chain in chains:
            atom_start = chain["atom_idx"]
            atom_end = atom_start + chain["atom_num"]

            for atom in structure.atoms[atom_start:atom_end]:
                name = get_atom_name(atom["name"])
                atom_type_list.append(C.atom.AtomName(name).index)
                element_list.append(atom["element"])
                charge_list.append(atom["charge"])
                is_resolved_list.append(atom["is_present"])
                label_coords_list.append(atom["coords"])

            # add atom to token map
            res_start = chain["res_idx"]
            res_end = res_start + chain["res_num"]
            for residue in structure.residues[res_start:res_end]:
                num_atoms_in_res = residue["atom_num"]
                if residue["is_standard"]:
                    # Proteins' amino acid and nucleic acids' base
                    token_idx_list.extend([token_ofs] * num_atoms_in_res)
                    token_ofs += 1
                else:
                    # Ligands, Motifications, Covalent inhibitors
                    token_idx_list.extend(
                        list(range(token_ofs, token_ofs + num_atoms_in_res))
                    )
                    token_ofs += num_atoms_in_res

        atom_type = torch.as_tensor(atom_type_list, dtype=torch.int32)
        element = torch.as_tensor(element_list, dtype=torch.int32)
        charge = torch.as_tensor(charge_list, dtype=torch.float)
        token_idx = torch.as_tensor(token_idx_list, dtype=torch.int32)
        resolved_mask = torch.as_tensor(resolve_mask_list, dtype=torch.bool)
        label_coords = torch.as_tensor(label_coords_list, dtype=torch.float)  # (N , 3)

        # HACK: right now just use label coords as apo coords
        apo_coords = label_coords.clone().unsqueeze(1)  # (N, 1, 3), 1: n_apo

        return model_input.AtomLayout(
            atom_type=atom_type,
            element=element,
            charge=charge,
            token_idx=token_idx,
            apo_coords=apo_coords,
            resolved_mask=resolved_mask,
            pad_mask=torch.ones_like(resolved_mask),
            label_coords=label_coords,
        )

    @staticmethod
    def parse_bond(
        chains: np.ndarray,
        structure: BoltzStructure,
    ) -> model_input.BondLayout:
        """Extract bond layout from BoltzStructure.

        Parameters
        ----------
        structure : BoltzStructure
            The BoltzStructure object.

        Returns
        -------
        BondLayout
            The parsed bond layout.
        """
        atom_1 = torch.as_tensor(structure.bonds["atom_1"], dtype=torch.int32)
        atom_2 = torch.as_tensor(structure.bonds["atom_2"], dtype=torch.int32)
        bond_type = torch.as_tensor(structure.bonds["type"], dtype=torch.int32)

        return model_input.BondLayout(
            atom_1=atom_1,
            atom_2=atom_2,
            bond_type=bond_type,
            pad_mask=torch.ones(len(atom_1), dtype=torch.bool),
        )
