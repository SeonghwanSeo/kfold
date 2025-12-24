import pathlib
import random

import numpy as np
import torch
from tqdm import tqdm

from kfold.config import load_config
from kfold.data import model_input, structure
from kfold.data.processing.component import CCD
from kfold.inference.dataset import prepare_inference_dataloader
from kfold.inference.query import InputFile, parse_input_files
from kfold.model.models import KFold


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="KFold Inference Script")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to the model configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path to the input(s) for inference (file or directory).",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=pathlib.Path,
        default=pathlib.Path("./inference_results/"),
        help="Root directory to save inference results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for inference reproducibility.",
    )
    parser.add_argument(
        "--num_recycles",
        type=int,
        default=10,
        help="Number of trunk cycles to run during inference.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Number of diffusion steps to run during inference.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate per input.",
    )
    parser.add_argument(
        "--ccd",
        type=pathlib.Path,
        default="/mnt/parallel_storage/wykim_lab/icl_shwan/data/ccd-boltz1.pkl",
        help="Path to the CCD data file.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for inference instead of GPU.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading.",
    )

    return parser.parse_args()


def main():
    # Setup environment
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)
    torch.set_autocast_dtype("cuda", torch.bfloat16)
    torch.set_autocast_enabled(True)

    args = parse_args()

    if args.cpu:
        raise NotImplementedError("CPU inference is not implemented yet.")

    # Load model
    config = load_config(args.config)
    if "model" in config:
        # Get model config if wrapped in a higher-level config
        config = config.model
    model: KFold = KFold.from_checkpoint(config, args.checkpoint)
    model = model.eval().cuda()

    # Load CCD data
    ccd: CCD = CCD.load(args.ccd)

    # Parse input query(s)
    # If directory is provided, invalid files are skipped.
    input_queries: list[InputFile] = parse_input_files(
        args.input,
        ccd=ccd,
        skip_invalid=True,
    )

    # Create data loader
    dataloader = prepare_inference_dataloader(
        input_files=input_queries,
        ccd=ccd,
        seq_embedding_dim=model.channel_seq_encoder,
        struct_embedding_dim=model.channel_struct_encoder,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # Run inference
    for batch in tqdm(dataloader, desc="Inference"):
        if batch is None:
            # Skip invalid input
            continue

        # Unpack batch
        query: InputFile = batch[0]
        struct: structure.TokenizedStructure = batch[1]
        f_input: model_input.FoldingInput = batch[2]

        if not f_input.is_batched:
            f_input = model_input.FoldingInput.from_list([f_input])

        assert f_input.batch_size == 1, "Inference batch size should be 1"
        f_input = f_input.to(device="cuda")

        # HACK: Set random seed for reproducibility
        # FIXME: pass random generator to model sampling function instead
        set_seed(args.seed)

        model_out, time_logs = model.sample(
            f_input,
            num_recycles=args.num_recycles,
            num_steps=args.num_steps,
            num_diffusion_samples=args.num_samples,
        )

        # remove batch dimension
        model_out = {k: v.squeeze(0) for k, v in model_out.items()}

        # NOTE: model_out contains:
        #   - s_trunk: final trunk outputs [Ntoken, C_s]
        #   - z_trunk: final trunk latent [Ntoken, Ntoken, C_z]
        #   - distogram_logits: predicted distogram logits [Ntoken, Ntoken, bin]
        #   - sample_coordinates: generated coordinates [num_samples, Natom, 3]
        # *) Ntoken and Natom may be different to original ones due to padding.

        # Save predictions
        name = query.name
        save_dir = args.out_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save query
        with open(save_dir / "query.yaml", "w") as f:
            f.write(query.yaml)

        # Save sampled structures
        new_struct: structure.TokenizedStructure = struct.replace_atom_coords(
            model_out["sample_coordinates"].cpu().numpy()
        )
        for i in range(args.num_samples):
            save_path = save_dir / f"sample-{i}.cif"
            new_struct.write(save_path, conformer_id=i, is_predicted=True)


if __name__ == "__main__":
    main()
