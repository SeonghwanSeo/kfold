import json
import random
from pathlib import Path

from tqdm import tqdm

from kfold.training.folding.dataset.cropper.boltz import BoltzCropper
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
    cropper = BoltzCropper(BoltzCropper.Config())

    keys = keys[:20]
    for key in tqdm(keys):
        record = parse_record(manifest[key])
        if record.num_chains > 20:
            # Skip large structures for testing
            continue

        # Set random seed for reproducibility
        random.seed(key)
        path = BOLTZ_STRUCTURE_DIR / f"{key}.npz"
        boltz_structure = BoltzStructure.load(path)
        tokenized = tokenize_structure(boltz_structure)
        # Crop structure
        cropped_tokenized = cropper.crop(tokenized, 384, None)

        # Save full and cropped structures
        tokenized.to_pdb(f"./tmp/{key}-full.pdb")
        cropped_tokenized.to_pdb(f"./tmp/{key}-crop.pdb")
