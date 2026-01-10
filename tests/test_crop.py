import io
import random
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm

from kfold.data.types.metadata import Metadata
from kfold.data.types.tokenized import TokenizedStructure
from kfold.training.dataset.cropper.multi_anchor import MultiAnchorCropper
from kfold.training.dataset.datamodule import load_manifest

LMDB_PATH = Path("/cache/wykim_lab/kfold_data/kfold_rcsb_processed_v251120.lmdb/")
MANIFEST_PATH = Path("/cache/wykim_lab/kfold_data/manifests/af3_manifest.pkl")
SAVE_PATH = Path("./tmp/pdb-crop/")


if __name__ == "__main__":
    all_metadatas: list[Metadata] = load_manifest(MANIFEST_PATH)

    env = lmdb.open(str(LMDB_PATH), readonly=True, lock=False, readahead=False)

    # data cropping
    cropper = MultiAnchorCropper(
        MultiAnchorCropper.Config(
            w_contiguous=0.3,
            w_spatial=0.2,
            w_spatial_interface=0.5,
        )
    )

    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    with env.begin(write=False) as txn:
        for i, metadata in enumerate(tqdm(all_metadatas[:100])):
            # Set random seed for reproducibility
            random.seed(i)
            np.random.seed(i)

            key = metadata.id

            if metadata.num_chains > 52:
                continue

            byte_data = txn.get(key.encode("utf-8"))

            with io.BytesIO(byte_data) as byte_stream:
                struct = TokenizedStructure.load_npz(byte_stream)
            struct = struct.copy_with(metadata=metadata)

            if struct.num_tokens < 768:
                # Skip small structures for testing
                continue

            cropped_struct = cropper.crop(struct, 384, None)

            # Save full and cropped structures
            try:
                struct.to_pdb(SAVE_PATH / f"{key}-full.pdb")
                cropped_struct.to_pdb(SAVE_PATH / f"{key}-cropped.pdb")
            except Exception as e:
                print(f"Failed to save {key}: {e}")
