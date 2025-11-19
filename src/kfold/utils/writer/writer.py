from pathlib import Path

from kfold.data.structure import TokenizedStructure

from .pdb import to_pdbstring

# TODO: add mmCIF writer


class KFoldWriter:
    @classmethod
    def write(
        cls,
        structure: TokenizedStructure,
        save_path: str | Path,
        conformer_id: int = 0,
        is_predicted: bool = True,
        save_apo: bool = False,
    ):
        format = Path(save_path).suffix.lower()
        if format == ".pdb":
            cls.write_pdb(structure, save_path, conformer_id, is_predicted, save_apo)
        elif format in [".cif", ".mmcif"]:
            cls.write_mmcif(structure, save_path, conformer_id, is_predicted, save_apo)
        else:
            raise ValueError(f"Unsupported file format: {save_path}")

    @staticmethod
    def write_mmcifstring(
        structure: TokenizedStructure,
        conformer_id: int = 0,
        is_predicted: bool = True,
        save_apo: bool = False,
    ) -> str:
        raise NotImplementedError("mmCIF format is not yet supported.")

    @classmethod
    def write_mmcif(
        cls,
        structure: TokenizedStructure,
        save_path: str | Path,
        conformer_id: int = 0,
        is_predicted: bool = True,
        save_apo: bool = False,
    ):
        with open(save_path, "w") as f:
            f.write(
                cls.write_mmcifstring(structure, conformer_id, is_predicted, save_apo)
            )

    @staticmethod
    def write_pdbstring(
        structure: TokenizedStructure,
        conformer_id: int = 0,
        is_predicted: bool = True,
        save_apo: bool = False,
    ) -> str:
        return to_pdbstring(structure, conformer_id, is_predicted, save_apo)

    @classmethod
    def write_pdb(
        cls,
        structure: TokenizedStructure,
        save_path: str | Path,
        conformer_id: int = 0,
        is_predicted: bool = True,
        save_apo: bool = False,
    ):
        with open(save_path, "w") as f:
            f.write(cls.write_pdbstring(structure, conformer_id, is_predicted, save_apo))

    # === Simple wrappers === #
    @classmethod
    def write_pdbstring_apo(
        cls,
        structure: TokenizedStructure,
        conformer_id: int = 0,
    ) -> str:
        return cls.write_pdbstring(structure, conformer_id, save_apo=True)

    @classmethod
    def write_pdb_apo(
        cls,
        structure: TokenizedStructure,
        save_path: str | Path,
        conformer_id: int = 0,
    ):
        cls.write_pdb(structure, save_path, conformer_id, save_apo=True)

    @classmethod
    def write_pdbstring_label(
        cls,
        structure: TokenizedStructure,
        conformer_id: int = 0,
    ) -> str:
        return cls.write_pdbstring(structure, conformer_id, is_predicted=False)

    @classmethod
    def write_pdb_label(
        cls,
        structure: TokenizedStructure,
        save_path: str | Path,
        conformer_id: int = 0,
    ):
        cls.write_pdb(structure, save_path, conformer_id, is_predicted=False)

    @classmethod
    def write_pdbstring_predicted(
        cls,
        structure: TokenizedStructure,
        conformer_id: int = 0,
    ) -> str:
        return cls.write_pdbstring(structure, conformer_id, is_predicted=True)

    @classmethod
    def write_pdb_predicted(
        cls,
        structure: TokenizedStructure,
        save_path: str | Path,
        conformer_id: int = 0,
    ):
        cls.write_pdb(structure, save_path, conformer_id, is_predicted=True)
