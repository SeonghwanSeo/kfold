import json
import random
from pathlib import Path

from tqdm import tqdm

from kfold.training.folding.dataset.cropper.alphafold import AlphaFold3Cropper
from kfold.utils.boltz.process import parse_record, tokenize_structure
from kfold.utils.boltz.structure import BoltzStructure

BOLTZ_PATH = Path("/cache/wykim_lab/rcsb_processed_targets/")
BOLTZ_MANIFEST_PATH = BOLTZ_PATH / "manifest.json"
BOLTZ_STRUCTURE_DIR = BOLTZ_PATH / "structures"


if __name__ == "__main__":
    with open(BOLTZ_MANIFEST_PATH) as f:
        manifest = json.load(f)

    manifest = {v["id"]: v for v in manifest}
    keys = sorted(list(manifest.keys()))
    random.seed(42)
    random.shuffle(keys)

    # data cropping
    cropper = AlphaFold3Cropper(AlphaFold3Cropper.Config())

    keys = keys[:100]
    for key in tqdm(keys):
        record = parse_record(manifest[key])
        if record.num_chains > 50:
            # Skip large structures for testing
            continue

        # Set random seed for reproducibility
        random.seed(key)
        path = BOLTZ_STRUCTURE_DIR / f"{key}.npz"
        try:
            boltz_structure = BoltzStructure.load(path)
            struct = tokenize_structure(boltz_structure)
            struct = struct.copy_with(metadata=record)
        except Exception as e:
            print(f"Error loading structure {key}: {e}")
            continue

        if struct.num_tokens < 768:
            continue

        # Crop structure
        cropped_struct = cropper.crop(struct, 384, None)

        # Save full and cropped structures
        struct.to_pdb(f"./tmp/pdb-crop/{key}-full.pdb", is_predicted=False)
        cropped_struct.to_pdb(f"./tmp/pdb-crop/{key}-crop.pdb", is_predicted=False)
