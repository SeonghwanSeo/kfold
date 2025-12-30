from pathlib import Path

from kfold.data.tokenized import TokenizedStructure
from kfold.utils import errors

PROCESSED_NPZ_PATH = Path("/cache/wykim_lab/kfold_rcsb_processed_v251120/")
SAVE_PDB_PATH = Path("./tmp/pdb/")


if __name__ == "__main__":
    files = list(PROCESSED_NPZ_PATH.glob("*.npz"))
    keys = [file.stem for file in files]
    keys = ["6cyg"]  # which the apo/holo structures are quite different

    SAVE_PDB_PATH.mkdir(parents=True, exist_ok=True)

    for key in keys:
        file = PROCESSED_NPZ_PATH / f"{key}.npz"
        tokenized = TokenizedStructure.load_npz(file)

        # Save full and cropped structures
        try:
            tokenized.to_pdb(SAVE_PDB_PATH / f"{file.stem}-apo.pdb", save_apo=True)
            tokenized.to_pdb(
                SAVE_PDB_PATH / f"{file.stem}-holo.pdb",
                is_predicted=False,
                save_apo=False,
            )
        except errors.PDBWriterMaxChainError as e:
            print(f"Skipping {file.stem} due to max chain error: {e}")
