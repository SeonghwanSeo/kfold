from pathlib import Path

import gemmi
import numpy as np

from kfold.data.types.structure import RefStructure

from .gemmi_utils import create_gemmi_structure, make_mmcif_block


class KFoldWriter:
    # =========================================================
    # Single structure write methods
    # =========================================================
    @classmethod
    def write_new_coords(
        cls,
        struct: RefStructure,
        coordinates: np.ndarray,
        filename: str | Path,
    ):
        if coordinates.shape != (struct.num_atoms, 3):
            raise ValueError(
                f"Coordinates shape {coordinates.shape} does not match shape "
                f"({struct.num_atoms}, 3)"
            )
        struct = struct.copy_with_new_coords(coordinates)
        cls.write(struct, filename, save_apo=False)

    @classmethod
    def write(
        cls,
        struct: RefStructure,
        filename: str | Path,
        save_apo: bool = False,
    ):
        format = Path(filename).suffix.lower()
        if format not in {".pdb", ".cif"}:
            raise ValueError(f"Unsupported file format: {format}")
        try:
            if format == ".pdb":
                cls.write_pdb(struct, filename, save_apo)
            else:
                cls.write_mmcif(struct, filename, save_apo)
        except Exception as e:
            print(f"Failed to write structure to {filename}: {e}")

    @staticmethod
    def write_mmcif(
        struct: RefStructure,
        filename: str | Path,
        save_apo: bool = False,
        ost_compatible: bool = True,
    ) -> None:
        gemmi_struct: gemmi.Structure = create_gemmi_structure(struct, save_apo)
        block: gemmi.cif.Block = make_mmcif_block(gemmi_struct, ost_compatible)
        block.write_file(str(filename))

    @staticmethod
    def write_pdb(
        struct: RefStructure,
        filename: str | Path,
        save_apo: bool = False,
    ) -> None:
        gemmi_struct: gemmi.Structure = create_gemmi_structure(
            struct, save_apo, pdb_compatible=True
        )
        gemmi_struct.write_pdb(str(filename))

    @staticmethod
    def write_mmcifstring(
        struct: RefStructure,
        save_apo: bool = False,
        ost_compatible: bool = True,
    ) -> str:
        gemmi_struct: gemmi.Structure = create_gemmi_structure(struct, save_apo)
        block: gemmi.cif.Block = make_mmcif_block(gemmi_struct, ost_compatible)
        return block.as_string()

    @staticmethod
    def write_pdbstring(
        struct: RefStructure,
        save_apo: bool = False,
    ) -> str:
        gemmi_struct: gemmi.Structure = create_gemmi_structure(
            struct, save_apo, pdb_compatible=True
        )
        return gemmi_struct.make_pdb_string()

    # =========================================================
    # Trajectory write methods
    # =========================================================
    @classmethod
    def write_trajectory(
        cls,
        struct: RefStructure,
        trajectory: np.ndarray,
        filename: str | Path,
    ):
        if trajectory.shape[1:] != (struct.num_atoms, 3):
            raise ValueError(
                f"Trajectory shape {trajectory.shape} does not match shape "
                f"(*, {struct.num_atoms}, 3)"
            )

        format = Path(filename).suffix.lower()
        if format not in {".pdb", ".cif"}:
            raise ValueError(f"Unsupported file format: {format}")

        pdb_compatible = format == ".pdb"

        try:
            n_frames = trajectory.shape[0]
            traj_structures: gemmi.Structure = gemmi.Structure()
            # Add models
            for i in range(n_frames):
                frame_coords = trajectory[i]
                frame_struct = struct.copy_with_new_coords(frame_coords)
                _struct: gemmi.Structure = create_gemmi_structure(
                    frame_struct, pdb_compatible=pdb_compatible
                )
                # Convert to block and back to ensure proper model addition
                block = gemmi.cif.read_string(
                    make_mmcif_block(_struct).as_string()
                ).sole_block()
                model = gemmi.make_structure_from_block(block)[0]
                if hasattr(model, "name"):
                    model.name = str(i + 1)
                else:
                    model.num = i + 1
                traj_structures.add_model(model, pos=-1)

            # Write to file
            if format == ".pdb":
                traj_structures.write_pdb(str(filename))
            else:
                block: gemmi.cif.Block = make_mmcif_block(traj_structures)
                block.write_file(str(filename))
        except Exception as e:
            print(f"Failed to write trajectory to {filename}: {e}")
