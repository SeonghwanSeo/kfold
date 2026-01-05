# Common component dictionary (CCD)
import dataclasses
import datetime
import pathlib
import pickle
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Self, TypeVar

import gemmi
import numpy as np
from rdkit import Chem

from kfold.data.utils import rdkit_utils

# Helper function

_T = TypeVar("_T")
Point3D = tuple[float, float, float]


def filter_items(items: Iterable[_T], mask: Iterable[Any]) -> list[_T]:
    """Filter items based on a boolean mask."""
    return [item for item, m in zip(items, mask, strict=True) if m]


def get_ideal_coordinates(cif_block: gemmi.cif.Block) -> dict[str, Point3D] | None:
    """Get the ideal coordinates as a numpy array.

    Parameters
    ----------
    cif_block : gemmi.cif.Block
        The CIF block containing the component data.

    Returns
    -------
    dict[str, Point3D] | None
        A dictionary mapping atom names to their ideal coordinates.
    """
    # Return None if no coordinate information is available
    if "_chem_comp_atom.pdbx_model_Cartn_x_ideal" not in cif_block:
        return None

    missing_flag = {"?", "."}

    names = cif_block.find_values("_chem_comp_atom.atom_id")
    x_coords = cif_block.find_values("_chem_comp_atom.pdbx_model_Cartn_x_ideal")
    y_coords = cif_block.find_values("_chem_comp_atom.pdbx_model_Cartn_y_ideal")
    z_coords = cif_block.find_values("_chem_comp_atom.pdbx_model_Cartn_z_ideal")

    num_atoms: int = len(x_coords)
    coord_dict: dict[str, Point3D] = {}
    for i in range(num_atoms):
        x_i, y_i, z_i = x_coords[i], y_coords[i], z_coords[i]
        if all(v not in missing_flag for v in (x_i, y_i, z_i)):
            coord_dict[names[i]] = (float(x_i), float(y_i), float(z_i))

    if len(coord_dict) == 0:
        # All coordinates are missing
        return None

    return coord_dict


def get_model_coordinates(
    cif_block: gemmi.cif.Block, date_cutoff: datetime.date | None = None
) -> dict[str, Point3D] | None:
    """Get the model coordinates as a numpy array.

    Parameters
    ----------
    cif_block : gemmi.cif.Block
        The CIF block containing the component data.
    date_cutoff : datetime.date | None, optional
        The date cutoff for using model coordinates (default is None).

    Returns
    -------
    dict[str, Point3D] | None
        A dictionary mapping atom names to their model coordinates.
    """
    # Return None if no coordinate information is available
    if "_chem_comp_atom.model_Cartn_x" not in cif_block:
        return None

    # If there are missing atoms in ideal coordinates, try to use model coordinates
    if date_cutoff is not None and "_chem_comp.pdbx_modified_date" in cif_block:
        # check release date and skip if after cutoff date
        release_date = datetime.date.fromisoformat(
            cif_block.find_value("_chem_comp.pdbx_modified_date")
        )
        if release_date > date_cutoff:
            return None

    missing_flag = {"?", "."}

    names = cif_block.find_values("_chem_comp_atom.atom_id")
    x_coords = cif_block.find_values("_chem_comp_atom.model_Cartn_x")
    y_coords = cif_block.find_values("_chem_comp_atom.model_Cartn_y")
    z_coords = cif_block.find_values("_chem_comp_atom.model_Cartn_z")

    num_atoms: int = len(x_coords)
    coord_dict: dict[str, Point3D] = {}
    for i in range(num_atoms):
        x_i, y_i, z_i = x_coords[i], y_coords[i], z_coords[i]
        if all(v not in missing_flag for v in (x_i, y_i, z_i)):
            coord_dict[names[i]] = (float(x_i), float(y_i), float(z_i))

    if len(coord_dict) == 0:
        # All coordinates are missing
        return None

    return coord_dict


@dataclasses.dataclass(frozen=True, slots=True)
class Component:
    """Base class for data processing components.
    NOTE: Skip field validation to reduce overhead.

    Attributes
    ----------
    code : str
        The unique code (CCD or SMILES) of the component.
    mol : Chem.Mol
        The RDKit molecule object.
    atom_names : tuple[str, ...]
        An array of atom names with shape (n_atoms,).
    elements : np.ndarray (np.uint8)
        An array of atomic numbers with shape (n_atoms,).
    charges : np.ndarray (np.int8)
        An array of formal charges with shape (n_atoms,).
    is_leaving_atom : np.ndarray (bool)
        A boolean array indicating leaving atoms with shape (n_atoms,).
    bonds : dict[tuple[str, str], int]
        A dictionary of bond orders between atom pairs.
    etkdg_coords : np.ndarray (np.float16) | None
        An array of pre-computed ektdg coordinates with shape (n_etkdg, n_atoms, 3).
    ideal_coords : np.ndarray (np.float16) | None
        An array of ideal coordinates with shape (n_atoms, 3),
    model_coords : np.ndarray (np.float16) | None
        An array of model coordinates with shape (n_atoms, 3),
    symmetries : tuple[list[int], ...]
        A tuple of arrays representing permutational symmetries.
        Only used in training.
    """

    code: str
    mol: Chem.Mol
    atom_names: tuple[str, ...]  # (n_atoms,)
    elements: np.ndarray  # (n_atoms,) with dtype=np.uint8
    charges: np.ndarray  # (n_atoms,) with dtype=np.int8
    is_leaving_atom: np.ndarray  # (n_atoms,) with dtype=bool
    bonds: dict[tuple[str, str], int]  # Bond orders between atom pairs
    etkdg_coords: np.ndarray | None  # (n_conf, n_atoms, 3) with dtype=np.float16
    ideal_coords: np.ndarray | None  # (n_atoms, 3) with dtype=np.float16
    model_coords: np.ndarray | None  # (n_atoms, 3) with dtype=np.float16
    symmetries: Sequence[list[int]] | None = None  # Permutational symmetries

    @property
    def num_atoms(self) -> int:
        """Get the number of atoms in the component."""
        return len(self.atom_names)

    @property
    def non_leaving_atom_names(self) -> tuple[str, ...]:
        """Get the names of non-leaving atoms in the component."""
        return tuple(filter_items(self.atom_names, ~self.is_leaving_atom))

    @property
    def num_leaving_atoms(self) -> int:
        """Get the number of leaving atoms in the component."""
        return int(np.sum(self.is_leaving_atom))

    @property
    def num_non_leaving_atoms(self) -> int:
        """Get the number of non-leaving atoms in the component."""
        return int(np.sum(~self.is_leaving_atom))

    def to_dict(self) -> dict:
        """Convert the Component instance to a dictionary without deepcopy"""
        fields = dataclasses.fields(self)
        return {field.name: getattr(self, field.name) for field in fields}

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create a Component instance from a dictionary."""
        return cls(**data)

    def get_atom_index_map(self) -> dict[str, int]:
        """Get a mapping from atom names to their indices.

        Returns
        -------
        dict[str, int]
            A dictionary mapping atom names to their indices.
        """
        return {name: idx for idx, name in enumerate(self.atom_names)}

    def get_conformer(
        self,
        conformer_type: str,
        rng: np.random.Generator | None = None,
        timeout: int = 30,
    ) -> np.ndarray | None:
        """Get the coordinates of the specified conformer type.

        Parameters
        ----------
        conformer_type : str
            The type of conformer to retrieve.
            Options:
              - auto: automatically select the most preferred conformer.
              - train: auto without 'etkdg' for faster retrieval during training.
              - etkdg: generate a new ETKDG conformer.
              - etkdg-cached: pre-computed ETKDG conformer (if available).
              - ideal: CCD ideal conformer (if available).
              - nan: return NaN coordinates.
        rng : np.random.Generator | None, optional
            A random number generator for conformer generation (default is None).
        timeout : int, optional
            Timeout for conformer generation in seconds.

        Returns
        -------
        np.ndarray
            An array of shape (n_atoms, 3) representing the coordinates.
            Returns None if the specified conformer type is not available.

        Notes
        -----
        - In general case, `etkdg-cached` -> `etkdg` -> `ideal` -> `model` is preferred.
        """
        cast = lambda x: x.astype(np.float32) if x is not None else None  # noqa: E731

        available_types = {
            "auto",
            "train",
            "etkdg",
            "etkdg-cached",
            "ideal",
            "model",
            "nan",
        }
        if conformer_type not in available_types:
            raise ValueError(
                f"Invalid conformer_type: {conformer_type}. "
                f"Available options are: {available_types}"
            )

        # If there is only one heavy atom, return zero coordinates
        if self.num_atoms == 1:
            return np.zeros((1, 3), dtype=np.float32)

        rng = rng or np.random.default_rng()

        if conformer_type in {"auto", "train"}:
            # Automatically select the most preferred conformer
            # Try to get cached ETKDG conformer first
            coords = self.get_conformer("etkdg-cached", rng)
            if coords is not None:
                return coords

            # Then try to generate a new ETKDG conformer
            # Skip during training for faster retrieval
            if conformer_type != "train":
                coords = self.get_conformer("etkdg", rng, timeout)
                if coords is not None:
                    return coords

            # Then try to get ideal conformer
            coords = self.get_conformer("ideal", rng)
            if coords is not None and np.isfinite(coords).all():
                # NOTE: Use ideal conformer only if all coordinates are finite
                return coords

            model_coords = self.get_conformer("model", rng)
            if model_coords is not None and np.isfinite(model_coords).any():
                # NOTE: Use model conformer if any coordinate is finite
                coords = model_coords

            if coords is not None:
                return coords

            # If no conformer is available, return NaN coordinates
            return self.get_conformer("nan", rng)

        elif conformer_type == "etkdg":
            rng = rng or np.random.default_rng()
            seed = int(rng.integers(1, 1 << 16))
            # Return a new ETKDG conformer
            mol = Chem.AddHs(self.mol)
            mol = rdkit_utils.compute_rdkit_conformer(mol, seed=seed, timeout=timeout)
            mol = Chem.RemoveHs(mol, sanitize=False)
            if mol.GetNumConformers() == 0:
                # Failed to generate conformer
                return None
            conf = mol.GetConformer(0)
            coords = np.array(conf.GetPositions(), dtype=np.float32)
            coords -= np.mean(coords, axis=0, keepdims=True)  # Center the coordinates
            return coords
        elif conformer_type == "etkdg-cached":
            # Return one of pre-computed ektdg conformers
            if self.etkdg_coords is None:
                return None
            num_confs = self.etkdg_coords.shape[0]
            if num_confs == 0:
                return None  # No conformers available
            elif num_confs == 1:
                return cast(self.etkdg_coords[0])  # Only one conformer available
            else:
                # Randomly select one conformer
                conf_idx = rng.integers(0, num_confs)
                return cast(self.etkdg_coords[conf_idx])
        if conformer_type == "ideal":
            if self.ideal_coords is None:
                return None
            return cast(self.ideal_coords)
        elif conformer_type == "model":
            if self.model_coords is None:
                return None
            return cast(self.model_coords)
        elif conformer_type == "nan":
            n_atoms = len(self.atom_names)
            return np.full((n_atoms, 3), np.nan, dtype=np.float32)
        else:
            raise RuntimeError(f"Unhandled conformer_type: {conformer_type}")

    @classmethod
    def from_mol(
        cls,
        code: str,
        mol: Chem.Mol,
        num_confs: int = 0,
        ideal_coords: dict[str, Point3D] | None = None,
        model_coords: dict[str, Point3D] | None = None,
        compute_symmetry: bool = False,
        is_ccd_component: bool = False,
        remove_hydrogens: bool = True,
        sanitize: bool = True,
        timeout: int = 30,
        rng: np.random.Generator | None = None,
    ) -> Self:
        """Create a Component instance from an RDKit molecule and coordinates.

        Parameters
        ----------
        code : str
            The unique code (CCD or SMILES) of the component.
        mol : Chem.Mol
            The RDKit molecule object.
        num_confs : int, optional
            The number of etkdg conformers to generate (default is 0).
        ideal_coords : dict[str, Point3D] | None, optional
            A dictionary of ideal coordinates (default is None).
        model_coords : dict[str, Point3D] | None, optional
            A dictionary of model coordinates (default is None).
        compute_symmetry : bool, optional
            Whether to compute permutational symmetries (default is False).
        is_ccd_component : bool, optional
            Whether the component is from CCD (default is False).
        sanitize : bool, optional
            Whether to sanitize the molecule (default is True).
        timeout : int, optional
            Timeout for conformer generation in seconds.
        rng : np.random.Generator | None, optional
            A random number generator for conformer generation (default is None).

        Returns
        -------
        Component
            A Component instance with the specified properties.
        """
        # 1. Prepare molecule
        if remove_hydrogens:
            mol = Chem.RemoveAllHs(mol, sanitize=False)  # Remove hydrogens for processing
        else:
            mol = Chem.Mol(mol)  # Create a copy to avoid modifying the original
        if sanitize:
            # Sanitize molecule
            success = rdkit_utils.sanitize_molecule(mol, allow_fail=True)
            if not success:
                print(f"Warning: Molecule {code} failed sanitization.")

        # 2. Check and assign atom names
        has_atom_names = all(atom.HasProp("name") for atom in mol.GetAtoms())
        if not has_atom_names:
            # Ensure all atom names are present for CCD components
            assert not is_ccd_component, f"CCD component {code} is missing atom names."
            # Assign default atom names if missing
            rdkit_utils.assign_atom_names(mol, max_name_length=4)

        # 3. Get the reference molecule properties
        # Get atom names
        ref_atom_names = [atom.GetProp("name") for atom in mol.GetAtoms()]
        assert all(name not in ("H", "D", "T") for name in ref_atom_names), (
            "Hydrogen atom names found in the molecule. "
        )

        # Get elements
        ref_elements: np.ndarray = np.array(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.uint8
        )
        # Get formal charges
        ref_charges: np.ndarray = np.array(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=np.int8
        )
        # Get leaving atom mask
        check_leaving_atom = lambda atom: (  # noqa:E731
            atom.HasProp("leaving_atom") and atom.GetBoolProp("leaving_atom")
        )
        is_leaving_atom: np.ndarray = np.array(
            [check_leaving_atom(atom) for atom in mol.GetAtoms()],
            dtype=bool,
        )

        # Get bonds
        bonds: dict[tuple[str, str], int] = {}
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            name1 = atom1.GetProp("name")
            name2 = atom2.GetProp("name")
            bond_key = (name1, name2) if name1 < name2 else (name2, name1)
            bond_type = int(
                bond.GetBondType()
            )  # 1: single, 2: double, 3: triple, 12: aromatic
            bonds[bond_key] = bond_type

        # Generate etkdg conformers if requested
        etkdg_coords_list: list[dict[str, Point3D]] = []
        if num_confs > 0 and mol.GetNumHeavyAtoms() > 1:
            # Only compute ETKDG for molecules with more than 1 atom
            rng = rng or np.random.default_rng()
            seed = int(rng.integers(1, 1 << 16))
            etkdg_mol = rdkit_utils.compute_rdkit_conformer(
                mol, num_confs, add_hydrogens=True, seed=seed, timeout=timeout
            )
            for conf in etkdg_mol.GetConformers():
                coords_dict: dict[str, Point3D] = {}
                for atom in etkdg_mol.GetAtoms():
                    if atom.HasProp("name"):
                        atom_name = atom.GetProp("name")
                        pos = conf.GetAtomPosition(atom.GetIdx())
                        coords_dict[atom_name] = (pos.x, pos.y, pos.z)
                etkdg_coords_list.append(coords_dict)

        # Get coordinates
        ideal_coords_arr: np.ndarray | None = None
        model_coords_arr: np.ndarray | None = None
        etkdg_coords_arr: np.ndarray | None = None

        def to_array(
            coords_dict: dict[str, Point3D], atom_names: list[str]
        ) -> np.ndarray | None:
            missing = (np.nan, np.nan, np.nan)
            coords_arr = np.array(
                [coords_dict.get(n, missing) for n in atom_names],
                dtype=np.float32,
            )
            if np.isnan(coords_arr).all():
                return None
            return coords_arr.astype(np.float16)  # Use float16 to save storage

        if mol.GetNumHeavyAtoms() > 1:
            if ideal_coords is not None:
                ideal_coords_arr = to_array(ideal_coords, ref_atom_names)
            if model_coords is not None:
                model_coords_arr = to_array(model_coords, ref_atom_names)
            if len(etkdg_coords_list) > 0:
                etkdg_coords_arr_list = [
                    to_array(cdict, ref_atom_names) for cdict in etkdg_coords_list
                ]
                etkdg_coords_arr_list = [
                    arr for arr in etkdg_coords_arr_list if arr is not None
                ]
                if len(etkdg_coords_arr_list) > 0:
                    etkdg_coords_arr = np.stack(etkdg_coords_arr_list, axis=0)
                else:
                    etkdg_coords_arr = None

        # 6. Compute symmetries if requested
        if compute_symmetry:
            symmetries = rdkit_utils.compute_molecule_symmetry(mol)
        else:
            symmetries = None

        # 7. Clean up molecule properties and conformers
        mol.RemoveAllConformers()
        for prop_name in mol.GetPropNames():
            mol.ClearProp(prop_name)
        for atom in mol.GetAtoms():
            for prop_name in atom.GetPropNames():
                atom.ClearProp(prop_name)
        for bond in mol.GetBonds():
            for prop_name in bond.GetPropNames():
                bond.ClearProp(prop_name)

        return cls(
            code=code,
            mol=mol,
            atom_names=tuple(ref_atom_names),
            elements=ref_elements,
            charges=ref_charges,
            is_leaving_atom=is_leaving_atom,
            bonds=bonds,
            ideal_coords=ideal_coords_arr,
            model_coords=model_coords_arr,
            etkdg_coords=etkdg_coords_arr,
            symmetries=symmetries,
        )

    @classmethod
    def from_ccd_cif(
        cls,
        code: str,
        mol: Chem.Mol,
        cif_block: gemmi.cif.Block,
        num_confs: int = 0,
        compute_symmetry: bool = False,
        date_cutoff: datetime.date | None = None,
        timeout: int = 30,
        rng: np.random.Generator | None = None,
    ) -> Self:
        # Get RDKit molecule with hydrogens
        # NOTE: chirality is already set in the original RDKit molecule
        mol = Chem.Mol(mol)

        # Get ideal and model coordinates
        ideal_coords = get_ideal_coordinates(cif_block)
        model_coords = get_model_coordinates(cif_block, date_cutoff=date_cutoff)

        # Set chirality from 3D coordinates
        if mol.GetNumConformers() > 0 and mol.GetNumHeavyAtoms() > 1:
            try:
                # Use the first conformer(ideal) to assign stereochemistry
                Chem.AssignStereochemistryFrom3D(mol, confId=0, replaceExistingTags=False)
            except RuntimeError:
                print(
                    f"Warning: Failed to assign stereochemistry for CCD component {code}."
                )

        # Remove molecule coordinates
        mol.RemoveAllConformers()

        # Sanitize
        mol = Chem.RemoveAllHs(mol, sanitize=False)
        success = rdkit_utils.sanitize_molecule(mol, allow_fail=True)
        if not success:
            print(f"Warning: Molecule {code} failed sanitization.")

        # Atom names in original CCD entry (including Hs)
        for atom in mol.GetAtoms():
            assert atom.HasProp("name"), (
                f"Atom in CCD component {code} is missing 'name' property."
            )

        return cls.from_mol(
            code=code,
            mol=mol,
            num_confs=num_confs,
            ideal_coords=ideal_coords,
            model_coords=model_coords,
            compute_symmetry=compute_symmetry,
            is_ccd_component=True,
            sanitize=False,
            timeout=timeout,
            rng=rng,
        )

    @classmethod
    def from_smiles(
        cls,
        code: str,
        smiles: str,
        num_confs: int = 0,
        compute_symmetry: bool = False,
        timeout: int = 30,
        rng: np.random.Generator | None = None,
    ) -> Self:
        """Create a Component instance from an RDKit molecule and coordinates.

        Parameters
        ----------
        code : str
            The unique code (CCD or SMILES) of the component.
        smiles : str
            The SMILES string of the component.
        num_confs : int, optional
            The number of etkdg conformers to generate (default is 0).
        compute_symmetry : bool, optional
            Whether to compute permutational symmetries (default is False).
        rng : np.random.Generator | None, optional
            A random number generator for conformer generation (default is None).

        Returns
        -------
        Component
            A Component instance with the specified properties.
        """
        # 1. Remove hydrogens for processing
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return cls.from_mol(
            code=code,
            mol=mol,
            num_confs=num_confs,
            compute_symmetry=compute_symmetry,
            is_ccd_component=False,
            sanitize=True,
            timeout=timeout,
            rng=rng,
        )


class CCD(Mapping[str, Component]):
    """Common Component Dictionary (CCD) for data processing components."""

    def __init__(self, components: dict[str, Component]) -> None:
        self.components: dict[str, Component] = components

    def __keys__(self):
        return self.components.keys()

    def __getitem__(self, key: str) -> Component:
        return self.components[key]

    def __iter__(self):
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def copy(self) -> Self:
        """Create a shallow copy of the CCD instance."""
        return self.__class__(self.components.copy())

    def add_component(self, component: Component) -> None:
        """Add a new component to the CCD."""
        self.components[component.code] = component

    def save(self, save_path: str | pathlib.Path) -> None:
        """Save the CCD instance to a file.

        Parameters
        ----------
        save_path : str | pathlib.Path
            The path to save the CCD file.
        """
        dicts = {code: comp.to_dict() for code, comp in self.components.items()}
        with open(save_path, "wb") as f:
            pickle.dump(dicts, f)

    @classmethod
    def load(cls, load_path: str | pathlib.Path) -> Self:
        """Load a CCD instance from a file.

        Parameters
        ----------
        load_path : str | pathlib.Path
            The path to load the CCD file from.

        Returns
        -------
        CCD
            A CCD instance loaded from the file.
        """
        with open(load_path, "rb") as f:
            dicts = pickle.load(f)
        components = {
            code: Component.from_dict(comp_dict) for code, comp_dict in dicts.items()
        }
        return cls(components)
