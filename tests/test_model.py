from pathlib import Path

import torch
from omegaconf import OmegaConf

from kfold.config import load_config
from kfold.data.model_input import FoldingInput
from kfold.model.models.kfold import KFold
from kfold.utils.boltz.process import parse_structure
from kfold.utils.boltz.structure import BoltzStructure

TEST_CONFIG_PATH = Path("./configs/af3-mini.yaml")

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
BOLTZ_PATH = Path(
    "/mnt/parallel_storage/wykim_lab/icl_mseok/BOLTZ1/rcsb_processed_targets/"
)
BOLTZ_MANIFEST_PATH = BOLTZ_PATH / "manifest.json"
BOLTZ_STRUCTURE_DIR = BOLTZ_PATH / "structures"

if __name__ == "__main__":
    global_config = load_config(TEST_CONFIG_PATH)

    # debug purpose: skip compile
    print("Disable trunk compilation for testing")
    global_config.model.compile_trunk = False
    global_config.model.compile_score_model = False

    # print config
    print(OmegaConf.to_yaml(global_config))

    # instantiate model
    model = KFold(global_config)
    model = model.to(device=DEVICE, dtype=DTYPE)
    print(model)

    # Print the number of parameters for each sub-module
    total_params = 0
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {num_params / 1e6:.2f}M parameters")
        total_params += num_params
    print(f"Total parameters: {total_params / 1e6:.2f}M parameters")

    # Test a forward pass with dummy data

    dummy_data_path = BOLTZ_STRUCTURE_DIR / "10gs.npz"
    boltz_structure = BoltzStructure.load(dummy_data_path)
    chains = boltz_structure.chains[boltz_structure.mask]
    folding_input = parse_structure(chains, boltz_structure)
    print(folding_input)

    # pad
    max_chains = 100
    max_tokens = 512
    folding_input = FoldingInput(
        chain=folding_input.chain.pad(max_chains),
        atom=folding_input.atom.pad(max_tokens * 24),
        token=folding_input.token.pad(max_tokens),
        bond=folding_input.bond.pad(max_tokens * 10),
    )
    folding_input = FoldingInput.from_list([folding_input])
    print(folding_input)
    folding_input = folding_input.to(device=DEVICE)

    with torch.autocast(device_type=DEVICE.type, dtype=DTYPE):
        model.forward(folding_input, 4, 20, 1)
