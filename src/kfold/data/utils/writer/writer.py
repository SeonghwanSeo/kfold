from pathlib import Path

import gemmi

from kfold.data.types.structure import RefStructure

from .gemmi_utils import create_gemmi_structure, make_mmcif_block


class KFoldWriter:
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
            print(f"Error writing file {filename}: {e}")

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
