from pathlib import Path

from kfold.data.tokenized import TokenizedStructure

from .mmcif import to_mmcifstring
from .pdb import to_pdbstring


class KFoldWriter:
    @classmethod
    def write(
        cls,
        struct: TokenizedStructure,
        save_path: str | Path,
        save_apo: bool = False,
    ):
        format = Path(save_path).suffix.lower()
        if format == ".pdb":
            cls.write_pdb(struct, save_path, save_apo)
        elif format in [".cif", ".mmcif"]:
            cls.write_mmcif(struct, save_path, save_apo)
        else:
            raise ValueError(f"Unsupported file format: {save_path}")

    @staticmethod
    def write_mmcifstring(
        struct: TokenizedStructure,
        save_apo: bool = False,
    ) -> str:
        return to_mmcifstring(struct, save_apo)

    @classmethod
    def write_mmcif(
        cls,
        struct: TokenizedStructure,
        save_path: str | Path,
        save_apo: bool = False,
    ):
        with open(save_path, "w") as f:
            f.write(to_mmcifstring(struct, save_apo))

    @staticmethod
    def write_pdbstring(
        struct: TokenizedStructure,
        save_apo: bool = False,
    ) -> str:
        return to_pdbstring(struct, save_apo)

    @classmethod
    def write_pdb(
        cls,
        struct: TokenizedStructure,
        save_path: str | Path,
        save_apo: bool = False,
    ):
        with open(save_path, "w") as f:
            f.write(to_pdbstring(struct, save_apo))
