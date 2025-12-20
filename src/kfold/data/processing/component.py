# Common component dictionary (CCD)
import dataclasses
import datetime
import pathlib
import pickle
from collections.abc import Mapping
from typing import Any, Self, TypeVar

import gemmi
import numpy as np
from rdkit import Chem

from . import rdkit_utils

# Helpfer function

_T = TypeVar("_T")


def filter_items(items: list[_T], mask: list[Any]) -> list[_T]:
    """Filter items based on a boolean mask."""
    return [item for item, m in zip(items, mask, strict=True) if m]


def get_ideal_coordinates(cif_block: gemmi.cif.Block) -> np.ndarray | None:
    """Get the ideal coordinates as a numpy array.

    Parameters
    ----------
    cif_block : gemmi.cif.Block
        The CIF block containing the component data.

    Returns
    -------
    np.ndarray | None
        An array of ideal coordinates with shape (n_atoms, 3),
        or None if no ideal coordinates are available.
    """
    # Return None if no coordinate information is available
    if "_chem_comp_atom.pdbx_model_Cartn_x_ideal" not in cif_block:
        return None

    missing_flag = {"?", "."}

    x_coords = cif_block.find_values("_chem_comp_atom.pdbx_model_Cartn_x_ideal")
    y_coords = cif_block.find_values("_chem_comp_atom.pdbx_model_Cartn_y_ideal")
    z_coords = cif_block.find_values("_chem_comp_atom.pdbx_model_Cartn_z_ideal")

    num_atoms: int = len(x_coords)
    ideal_coords: np.ndarray = np.full((num_atoms, 3), np.nan, dtype=np.float32)
    for i in range(num_atoms):
        x_i, y_i, z_i = x_coords[i], y_coords[i], z_coords[i]
        if all(v not in missing_flag for v in (x_i, y_i, z_i)):
            ideal_coords[i, 0] = float(x_i)
            ideal_coords[i, 1] = float(y_i)
            ideal_coords[i, 2] = float(z_i)

    if np.isnan(ideal_coords).all():
        # All coordinates are missing
        return None

    return ideal_coords


def get_model_coordinates(
    cif_block: gemmi.cif.Block, date_cutoff: datetime.date | None = None
) -> np.ndarray | None:
    """Get the model coordinates as a numpy array.

    Parameters
    ----------
    cif_block : gemmi.cif.Block
        The CIF block containing the component data.
    date_cutoff : datetime.date | None, optional
        The date cutoff for using model coordinates (default is None).

    Returns
    -------
    np.ndarray | None
        An array of model coordinates with shape (n_atoms, 3),
        or None if no model coordinates are available.
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

    x_coords = cif_block.find_values("_chem_comp_atom.model_Cartn_x")
    y_coords = cif_block.find_values("_chem_comp_atom.model_Cartn_y")
    z_coords = cif_block.find_values("_chem_comp_atom.model_Cartn_z")

    num_atoms: int = len(x_coords)
    model_coords = np.full((num_atoms, 3), np.nan, dtype=np.float32)
    for i in range(num_atoms):
        x_i, y_i, z_i = x_coords[i], y_coords[i], z_coords[i]
        if all(v not in missing_flag for v in (x_i, y_i, z_i)):
            model_coords[i, 0] = float(x_i)
            model_coords[i, 1] = float(y_i)
            model_coords[i, 2] = float(z_i)

    if np.isnan(model_coords).all():
        # All coordinates are missing
        return None

    return model_coords


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
    etkdg_coords : np.ndarray (np.float32) | None
        An array of pre-computed ektdg coordinates with shape (n_etkdg, n_atoms, 3).
    ideal_coords : np.ndarray (np.float32) | None
        An array of ideal coordinates with shape (n_atoms, 3),
    model_coords : np.ndarray (np.float32) | None
        An array of model coordinates with shape (n_atoms, 3),
    symmetries : tuple[list[int], ...]
        A tuple of arrays representing permutational symmetries.
        Only used in training.
    properties : dict
        A dictionary of additional properties.
        Since rdkit Molecule objects' properties are often not serializable
        during multiprocessing (num_workers>0), we store them here.
    """

    code: str
    mol: Chem.Mol
    atom_names: tuple[str, ...]  # (n_atoms,)
    elements: np.ndarray  # (n_atoms,) with dtype=np.uint8
    charges: np.ndarray  # (n_atoms,) with dtype=np.int8
    is_leaving_atom: np.ndarray  # (n_atoms,) with dtype=bool
    etkdg_coords: np.ndarray | None  # (n_conf, n_atoms, 3) with dtype=np.float32
    ideal_coords: np.ndarray | None  # (n_atoms, 3) with dtype=np.float32
    model_coords: np.ndarray | None  # (n_atoms, 3) with dtype=np.float32
    symmetries: tuple[list[int], ...] = ()  # Permutational symmetries
    properties: dict = dataclasses.field(default_factory=dict)  # Additional properties

    def to_dict(self) -> dict:
        """Convert the Component instance to a dictionary without deepcopy"""
        fields = dataclasses.fields(self)
        return {field.name: getattr(self, field.name) for field in fields}

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create a Component instance from a dictionary."""
        return cls(**data)

    def get_conformer(
        self,
        conformer_type: str,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray | None:
        """Get the coordinates of the specified conformer type.

        Parameters
        ----------
        conformer_type : str
            The type of conformer to retrieve.
            Options:
              - auto: automatically select the most preferred conformer.
              - etkdg: generate a new ETKDG conformer.
              - etkdg-cached: pre-computed ETKDG conformer (if available).
              - ideal: CCD ideal conformer (if available).
              - nan: return NaN coordinates.

        Returns
        -------
        np.ndarray
            An array of shape (n_atoms, 3) representing the coordinates.
            Returns None if the specified conformer type is not available.

        Notes
        -----
        - In general case, `etkdg-cached` -> `etkdg` -> `ideal` -> `model` is preferred.
        """
        available_types = {"auto", "etkdg", "etkdg-cached", "ideal", "model", "nan"}
        if conformer_type not in available_types:
            raise ValueError(
                f"Invalid conformer_type: {conformer_type}. "
                f"Available options are: {available_types}"
            )

        rng = rng or np.random.default_rng()

        if conformer_type == "auto":
            # Automatically select the most preferred conformer
            # Try to get cached ETKDG conformer first
            coords = self.get_conformer("etkdg-cached", rng)
            if coords is not None:
                return coords

            # Then try to generate a new ETKDG conformer
            coords = self.get_conformer("etkdg", rng)
            if coords is not None:
                return coords

            # Then try to get ideal conformer
            coords = self.get_conformer("ideal", rng)
            if coords is not None and np.isfinite(coords).all():
                # NOTE: Use ideal conformer only if all coordinates are finite
                return coords

            model_coords = self.get_conformer("model", rng)
            if model_coords is not None and not np.isnan(model_coords).all():
                coords = model_coords

            if coords is not None:
                return coords

            # If no conformer is available, return NaN coordinates
            return self.get_conformer("nan", rng)

        elif conformer_type == "etkdg":
            # Return a new ETKDG conformer
            mol = Chem.AddHs(self.mol)
            mol = rdkit_utils.compute_rdkit_conformer(mol, rng=rng)
            mol = Chem.RemoveHs(mol, sanitize=False)
            if mol.GetNumConformers() == 0:
                # Failed to generate conformer
                return None
            conf = mol.GetConformer(0)
            coords = np.array(conf.GetPositions(), dtype=np.float32)
            return coords
        elif conformer_type == "etkdg-cached":
            # Return one of pre-computed ektdg conformers
            if self.etkdg_coords is None:
                return None
            num_confs = self.etkdg_coords.shape[0]
            if num_confs == 0:
                return None  # No conformers available
            elif num_confs == 1:
                return self.etkdg_coords[0]  # Only one conformer available
            else:
                # Randomly select one conformer
                conf_idx = rng.integers(0, num_confs)
                coords = self.etkdg_coords[conf_idx]
                return coords
        if conformer_type == "ideal":
            return self.ideal_coords
        elif conformer_type == "model":
            return self.model_coords
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
        ideal_conf_id: int | None = None,
        model_conf_id: int | None = None,
        etkdg_conf_ids: list[int] | None = None,
        compute_symmetry: bool = False,
        is_ccd_component: bool = False,
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
        ideal_conf_id : int | None, optional
            The conformer ID for ideal coordinates (default is None).
        model_conf_id : int | None, optional
            The conformer ID for model coordinates (default is None).
        etkdg_conf_ids : list[int] | None, optional
            The conformer IDs for etkdg coordinates (default is None).
        compute_symmetry : bool, optional
            Whether to compute permutational symmetries (default is False).
        is_ccd_component : bool, optional
            Whether the component is from CCD (default is False).
        rng : np.random.Generator | None, optional
            A random number generator for conformer generation (default is None).

        Returns
        -------
        Component
            A Component instance with the specified properties.
        """
        # 1. Remove hydrogens for processing
        mol = Chem.Mol(mol)  # Create a copy to avoid modifying the original
        success = rdkit_utils.sanitize_molecule(mol, allow_fail=True)
        if not success:
            print(f"Warning: Molecule {code} failed sanitization.")

        # 2. Label ideal and model conformers if provided
        ideal_conf: Chem.Conformer | None = None
        if ideal_conf_id is not None:
            ideal_conf = rdkit_utils.get_conformer(mol, ideal_conf_id)
            if ideal_conf is not None:
                ideal_conf = Chem.Conformer(ideal_conf)  # Create a copy
                ideal_conf.SetProp("source", "ideal")

        model_conf: Chem.Conformer | None = None
        if model_conf_id is not None:
            model_conf = rdkit_utils.get_conformer(mol, model_conf_id)
            if model_conf is not None:
                model_conf = Chem.Conformer(model_conf)  # Create a copy
                model_conf.SetProp("source", "model")

        # 3. Label etkdg conformers if provided and generate new ones if requested
        etkdg_confs: list[Chem.Conformer] = []
        if etkdg_conf_ids is not None:
            for conf_id in etkdg_conf_ids:
                etkdg_conf = rdkit_utils.get_conformer(mol, conf_id)
                if etkdg_conf is not None:
                    etkdg_conf = Chem.Conformer(etkdg_conf)  # Create a copy
                    etkdg_conf.SetProp("source", "etkdg")
                    etkdg_confs.append(etkdg_conf)

        if num_confs > 0 and mol.GetNumHeavyAtoms() > 1:
            # Only compute ETKDG for molecules with more than 1 atom
            rng = rng or np.random.default_rng()
            etkdg_mol = rdkit_utils.compute_rdkit_conformer(mol, num_confs, rng)

            for conf in etkdg_mol.GetConformers():
                conf.SetProp("source", "etkdg")
                etkdg_confs.append(conf)

        # 4. Combine conformers to extract heavy atom coordinates
        mol.RemoveAllConformers()
        if ideal_conf is not None:
            mol.AddConformer(ideal_conf)
        if model_conf is not None:
            mol.AddConformer(model_conf)
        for conf in etkdg_confs:
            mol.AddConformer(conf)

        # Remove hydrogens
        mol = Chem.RemoveAllHs(mol, sanitize=False)

        # Get coordinates
        ideal_coords: np.ndarray | None = None
        model_coords: np.ndarray | None = None
        etkdg_coords_list: list[np.ndarray] = []
        for conf in mol.GetConformers():
            source = conf.GetProp("source")
            coords = np.array(conf.GetPositions(), dtype=np.float32)
            if source == "ideal":
                ideal_coords = coords
            elif source == "model":
                model_coords = coords
            elif source == "etkdg":
                etkdg_coords_list.append(coords)

        etkdg_coords: np.ndarray | None = None
        if len(etkdg_coords_list) > 0:
            etkdg_coords = np.stack(etkdg_coords_list, axis=0)

        # 5. Get atom informations
        # Get atom_name
        has_atom_names = all(atom.HasProp("name") for atom in mol.GetAtoms())
        if is_ccd_component:
            # Ensure all atom names are present for CCD components
            assert has_atom_names, (
                "CCD component molecule is missing atom names. "
                "Please ensure atom names are assigned."
            )
        if not has_atom_names:
            # Assign default atom names if missing
            rdkit_utils.assign_atom_names(mol)
        ref_atom_names = [atom.GetProp("name") for atom in mol.GetAtoms()]
        assert all(name not in ("H", "D", "T") for name in ref_atom_names), (
            "Hydrogen atom names found in the molecule. "
            "Please remove hydrogens before processing."
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

        # 6. Compute symmetries if requested
        if compute_symmetry:
            symmetries = rdkit_utils.compute_molecule_symmetry(mol)
        else:
            symmetries = ()

        # 7. Clean up molecule properties and conformers
        rdkit_utils.sanitize_molecule(mol, allow_fail=True)
        mol.RemoveAllConformers()
        for prop_name in mol.GetPropNames():
            mol.ClearProp(prop_name)
        for atom in mol.GetAtoms():
            for prop_name in atom.GetPropNames():
                atom.ClearProp(prop_name)

        return cls(
            code=code,
            mol=mol,
            atom_names=tuple(ref_atom_names),
            is_leaving_atom=is_leaving_atom,
            elements=ref_elements,
            charges=ref_charges,
            ideal_coords=ideal_coords,
            model_coords=model_coords,
            etkdg_coords=etkdg_coords,
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
        rng: np.random.Generator | None = None,
    ) -> Self:
        # Get RDKit molecule with hydrogens
        # NOTE: chirality is already set in the original RDKit molecule
        mol = Chem.Mol(mol)

        # Sanitize
        success = rdkit_utils.sanitize_molecule(mol, allow_fail=True)
        if not success:
            print(f"Warning: Molecule {code} failed sanitization.")

        # Atom names in original CCD entry (including Hs)
        for atom in mol.GetAtoms():
            assert atom.HasProp("name"), (
                f"Atom in CCD component {code} is missing 'name' property."
            )

        # Get ideal and model coordinates
        ideal_coords = get_ideal_coordinates(cif_block)
        model_coords = get_model_coordinates(cif_block, date_cutoff=date_cutoff)

        # Add ideal and model conformers to the molecule
        if ideal_coords is not None:
            ideal_conf_id = rdkit_utils.add_conformer_with_coordinates(
                mol, ideal_coords, set_chirality=False
            )
        else:
            ideal_conf_id = None

        if model_coords is not None:
            model_conf_id = rdkit_utils.add_conformer_with_coordinates(
                mol, model_coords, set_chirality=False
            )
        else:
            model_conf_id = None

        return cls.from_mol(
            code=code,
            mol=mol,
            num_confs=num_confs,
            ideal_conf_id=ideal_conf_id,
            model_conf_id=model_conf_id,
            compute_symmetry=compute_symmetry,
            is_ccd_component=True,
            rng=rng,
        )

    @classmethod
    def from_smiles(
        cls,
        code: str,
        smiles: str,
        num_confs: int = 0,
        compute_symmetry: bool = False,
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
        mol = Chem.AddHs(mol)  # Add hydrogens for conformer generation
        return cls.from_mol(
            code=code,
            mol=mol,
            num_confs=num_confs,
            compute_symmetry=compute_symmetry,
            is_ccd_component=False,
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
