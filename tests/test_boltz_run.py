import io
import random
from pathlib import Path

import lmdb
import torch

from kfold.config import load_config
from kfold.data.featurize import featurize_structure
from kfold.data.tokenized import TokenizedStructure
from kfold.model.models.boltz1 import Boltz1

TEST_CONFIG_PATH = Path("./configs/train-boltz1.yaml")
VALIDATION_ID_PATH = Path("./assets/splits/boltz1/validation_ids.txt")
LMDB_PATH = Path(
    "/mnt/parallel_storage/wykim_lab/icl_shwan/data/structures/kfold_rcsb_processed_v251116.lmdb/"
)
SAVE_FEATURE_PATH = Path("./tmp/features.pt")

if __name__ == "__main__":
    # === Get data samples === #
    # load validation ids
    with open(VALIDATION_ID_PATH) as f:
        validation_ids = [line.strip().lower() for line in f.readlines()]
    validation_ids.sort()
    random.seed(42)
    random.shuffle(validation_ids)

    # Load LMDB
    lmdb_env = lmdb.open(
        str(LMDB_PATH),
        map_size=100 * 1024 * 1024 * 1024,  # 100 GB
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    # Get first 10 validation samples
    data: list[tuple[str, TokenizedStructure]] = []
    with lmdb_env.begin(write=False) as txn:
        for pdb_id in validation_ids[:100]:
            data_bytes = txn.get(pdb_id.lower().encode("utf-8"))
            if data_bytes is None:
                print(f"Data for {pdb_id} not found in LMDB.")
                continue

            with io.BytesIO(data_bytes) as byte_stream:
                tokenized_structure = TokenizedStructure.load_npz(byte_stream)
                data.append((pdb_id, tokenized_structure))

    # === Load Model === #
    global_config = load_config(TEST_CONFIG_PATH)
    model = Boltz1(global_config)
    model = model.eval()
    model = model.cuda()

    # === Get Embeddings and Save === #
    SAVE_FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for pdb_id, sample in data:
            print(
                f"Loaded sample {pdb_id}: {sample.num_chains} chains and "
                f"{sample.num_residues} residues."
            )

            # store token count before padding
            n_tokens = sample.num_tokens

            # Featurization
            f_input = featurize_structure(sample)
            f_input = f_input.pad_to_multiple_of(8)  # Make input size compatible
            f_input = f_input.from_list([f_input])  # Add batch dimension
            f_input = f_input.to(device="cuda")

            print(f_input)

            s_inputs, s_init, z_init = model.input_embedder(f_input)

            s_trunk, z_trunk = model.trunk(
                s_inputs,
                s_init,
                z_init,
                f_input,
                num_cycles=2,
            )

            p_distogram = model.distogram_head(z_trunk)

            feature_save_path = SAVE_FEATURE_PATH.parent / f"{pdb_id}_features.pt"
            features = {
                "s_inputs": s_inputs.cpu()[0, :n_tokens],
                "s_init": s_init.cpu()[0, :n_tokens],
                "z_init": z_init.cpu()[0, :n_tokens, :n_tokens],
                "s_trunk": s_trunk.cpu()[0, :n_tokens],
                "z_trunk": z_trunk.cpu()[0, :n_tokens, :n_tokens],
                "p_distogram": p_distogram.cpu()[0, :n_tokens, :n_tokens],
            }
            torch.save(features, feature_save_path)
            print(f"Saved features for {pdb_id} to {feature_save_path}")
