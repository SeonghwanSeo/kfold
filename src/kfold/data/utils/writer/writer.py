from pathlib import Path
from textwrap import wrap

import gemmi
import numpy as np

from kfold import __version__
from kfold.data.types.structure import RefStructure
from kfold.utils.geometry.rigid_align import rigid_align

from .gemmi_utils import create_gemmi_structure, make_mmcif_block

MODEL_NAME = "K-Fold"
ARTICLE_TITLE = (
    "Generative modeling of binding-induced structural change in biomolecular complexes"
)
AUTHOR_NAME = "Team KAIST"


def _add_pdb_header(struct: gemmi.Structure) -> None:
    """Add prediction and reference metadata to PDB output."""
    remarks = [
        "HEADER",
        f"TITLE     {MODEL_NAME} prediction",
        "REMARK   1 REFERENCE 1",
        f"REMARK   1  AUTH   {AUTHOR_NAME}",
    ]
    for index, line in enumerate(wrap(ARTICLE_TITLE, width=61), start=1):
        continuation = "  " if index == 1 else f"{index:2d}"
        remarks.append(f"REMARK   1  TITL{continuation} {line}")
    struct.raw_remarks = remarks


def _make_prediction_mmcif_block(
    struct: gemmi.Structure, ost_compatible: bool = True
) -> gemmi.cif.Block:
    """Create prediction metadata before adding structure categories."""
    block = gemmi.cif.Block(struct.name)
    block.set_pair("_entry.id", gemmi.cif.quote(struct.name))
    audit_loop = block.init_loop("_audit_author.", ["name", "pdbx_ordinal"])
    audit_loop.add_row([gemmi.cif.quote(AUTHOR_NAME), "1"])
    # block.set_pair("_citation.id", "primary")
    # block.set_pair("_citation.title", gemmi.cif.quote(ARTICLE_TITLE))
    # author_loop = block.init_loop(
    #     "_citation_author.", ["citation_id", "ordinal", "name"]
    # )
    software_loop = block.init_loop(
        "_software.",
        ["pdbx_ordinal", "name", "type", "description", "classification", "version"],
    )
    software_loop.add_row(
        [
            "1",
            MODEL_NAME,
            "package",
            gemmi.cif.quote(f"{MODEL_NAME} prediction pipeline"),
            gemmi.cif.quote("model building"),
            gemmi.cif.quote(__version__),
        ]
    )
    software_loop.add_row(
        [
            "2",
            "AtlasFold",
            "package",
            gemmi.cif.quote("Apo structure and prior candidate generation"),
            gemmi.cif.quote("model building"),
            "1.0.2",
        ]
    )
    return make_mmcif_block(struct, ost_compatible, block=block)


class KFoldWriter:
    # =========================================================
    # Single structure write methods
    # =========================================================
    @classmethod
    def write_new_coords(
        cls,
        struct: RefStructure,
        filename: str | Path,
        coordinates: np.ndarray,
        b_factors: np.ndarray | None = None,
    ):
        if coordinates.shape != (struct.num_atoms, 3):
            raise ValueError(
                f"Coordinates shape {coordinates.shape} does not match shape "
                f"({struct.num_atoms}, 3)"
            )
        struct = struct.copy_with_new_coords(coordinates, b_factors=b_factors)
        cls.write(struct, filename)

    @classmethod
    def write(
        cls,
        struct: RefStructure,
        filename: str | Path,
    ):
        format = Path(filename).suffix.lower()
        if format not in {".pdb", ".cif"}:
            raise ValueError(f"Unsupported file format: {format}")
        try:
            if format == ".pdb":
                cls.write_pdb(struct, filename)
            else:
                cls.write_mmcif(struct, filename)
        except Exception as e:
            print(f"Failed to write structure to {filename}: {e}")

    @staticmethod
    def write_mmcif(
        struct: RefStructure,
        filename: str | Path,
        ost_compatible: bool = True,
    ) -> None:
        gemmi_struct: gemmi.Structure = create_gemmi_structure(struct)
        block = _make_prediction_mmcif_block(gemmi_struct, ost_compatible)
        block.write_file(str(filename))

    @staticmethod
    def write_pdb(
        struct: RefStructure,
        filename: str | Path,
    ) -> None:
        gemmi_struct: gemmi.Structure = create_gemmi_structure(
            struct, pdb_compatible=True
        )
        _add_pdb_header(gemmi_struct)
        gemmi_struct.write_pdb(str(filename))

    @staticmethod
    def write_mmcifstring(
        struct: RefStructure,
        ost_compatible: bool = True,
    ) -> str:
        gemmi_struct: gemmi.Structure = create_gemmi_structure(struct)
        block = _make_prediction_mmcif_block(gemmi_struct, ost_compatible)
        return block.as_string()

    @staticmethod
    def write_pdbstring(
        struct: RefStructure,
    ) -> str:
        gemmi_struct: gemmi.Structure = create_gemmi_structure(
            struct, pdb_compatible=True
        )
        _add_pdb_header(gemmi_struct)
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
        align: bool = True,
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
            traj_structures.name = struct.metadata.id
            # Add models
            prev_coords = None
            for i in range(n_frames):
                frame_coords = trajectory[i]
                if align and prev_coords is not None:
                    frame_coords = rigid_align(frame_coords, prev_coords, mask=None)
                prev_coords = frame_coords
                frame_struct = struct.copy_with_new_coords(frame_coords)
                _struct: gemmi.Structure = create_gemmi_structure(
                    frame_struct, pdb_compatible=pdb_compatible
                )
                if i == 0:
                    traj_structures.connections = _struct.connections
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
                _add_pdb_header(traj_structures)
                traj_structures.write_pdb(str(filename))
            else:
                block = _make_prediction_mmcif_block(traj_structures)
                block.write_file(str(filename))
        except Exception as e:
            raise OSError(f"Failed to write trajectory to {filename}") from e
