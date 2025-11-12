import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from kfold.data.metadata import Metadata
from kfold.utils.boltz.process import parse_record

BOLTZ_PATH = Path("/cache/wykim_lab/rcsb_processed_targets/")
BOLTZ_MANIFEST_PATH = BOLTZ_PATH / "manifest.json"
BOLTZ_STRUCTURE_DIR = BOLTZ_PATH / "structures"
BOLTZ_VAL_KEY_PATH = Path("./assets/boltz_split/validation_ids.txt")


if __name__ == "__main__":
    with open(BOLTZ_MANIFEST_PATH) as f:
        manifest = json.load(f)

    manifest = {v["id"]: v for v in manifest}

    with open(BOLTZ_VAL_KEY_PATH) as f:
        keys = list(line.strip().lower() for line in f)

    # keys = sorted(list(manifest.keys()))
    random.seed(42)
    random.shuffle(keys)

    # keys = keys[:10000]

    total_length = []
    interface_length = []

    for key in tqdm(keys):
        record: Metadata = parse_record(manifest[key])

        num_chains = sum(chain.valid for chain in record.chains)
        if num_chains > 20:
            print(num_chains)

        total_length.append(
            sum(chain.num_residues for chain in record.chains if chain.valid)
        )

        chain_dict = {chain.asym_id: chain for chain in record.chains}
        for interface in record.interfaces:
            if interface.valid is False:
                continue
            chains = [chain_dict[asym_id] for asym_id in interface.asym_ids]
            interface_length.append(sum(chain.num_residues for chain in chains))

    length_arr = np.array(total_length)
    print("Complex Lengths")
    print("Total", len(length_arr))
    print("384", np.sum(length_arr < 384) / len(length_arr))
    print("512", np.sum(length_arr < 512) / len(length_arr))
    print("768", np.sum(length_arr < 768) / len(length_arr))
    print("1024", np.sum(length_arr < 1024) / len(length_arr))
    print("-----")

    length_arr = np.array(interface_length)
    print("Interface Lengths")
    print("Total", len(length_arr))
    print("384", np.sum(length_arr < 384) / len(length_arr))
    print("512", np.sum(length_arr < 512) / len(length_arr))
    print("768", np.sum(length_arr < 768) / len(length_arr))
    print("1024", np.sum(length_arr < 1024) / len(length_arr))
