import argparse
import logging
import pathlib

import torch
from tqdm import tqdm

from kfold.data.types.ccd import CCD
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.inference.dataset import prepare_inference_dataloader
from kfold.inference.query import Query, parse_input_files
from kfold.model.models import KFold

logger = logging.getLogger("kfold.inference")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
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
        "--use_sequence_masking",
        action="store_true",
        help=(
            "Whether to mask sequence to increase sampling diversity."
            "This is only meaningful when using multiple seeds"
        ),
    )
    parser.add_argument(
        "--ccd",
        type=pathlib.Path,
        default="/mnt/parallel_storage/wykim_lab/icl_shwan/data/ccd-train.pkl",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from previous inference results if available.",
    )

    return parser.parse_args()


@torch.inference_mode()
def main():
    # Setup environment
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")

    args = parse_args()

    if args.cpu:
        raise NotImplementedError("CPU inference is not implemented yet.")

    # Load CCD data
    logger.info(f"Loading CCD data from: {args.ccd}")
    ccd: CCD = CCD.load(args.ccd)
    logger.info("CCD data loaded successfully.")

    # Parse input query(s)
    # If directory is provided, invalid files are skipped.
    logger.info(f"Parsing input queries from: {args.input}")
    input_queries: list[Query] = parse_input_files(
        args.input,
        ccd=ccd,
        skip_invalid=True,
    )
    logger.info(f"Parsed {len(input_queries)} valid input queries")

    if args.resume:
        # Filter out queries that already have results saved
        logger.info("Filtering out queries with existing results for resuming inference")
        input_queries = [q for q in input_queries if not (args.out_dir / q.name).exists()]
        logger.info(f"{len(input_queries)} queries remaining after filtering for resume")

    # Create data loader
    dataloader = prepare_inference_dataloader(
        queries=input_queries,
        ccd=ccd,
        num_samples=args.num_samples,
        use_sequence_masking=args.use_sequence_masking,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model: KFold = KFold.from_checkpoint(args.config, args.checkpoint)
    model = model.cast_to_bf16().eval().cuda()
    logger.info("Model loaded successfully.")

    # mmCIF writer
    writer = KFoldWriter()

    # Run inference
    logger.info("Starting inference...")
    for batch in tqdm(dataloader, desc="Inference"):
        if batch is None:
            # Skip invalid input
            continue

        # Unpack batch
        query: Query = batch[0]
        ref_struct: RefStructure = batch[1]
        f_input: FoldingInput = batch[2]
        apo_dict: dict[int, dict] = batch[3]

        if not f_input.is_batched:
            f_input = FoldingInput.from_list([f_input])

        assert f_input.batch_size == 1, "Inference batch size should be 1"
        f_input = f_input.to(device="cuda")

        # Tokenize apo structure and fill in input features
        tokenize_apo = model.structure_encoder.tokenize
        for entity_id, apo_info in apo_dict.items():  # noqa
            aatypes, coords = apo_info["aatypes"], apo_info["coords"]
            seq_st, seq_ed, apo_st, apo_ed = apo_info["mapping"]
            seq_slc, apo_slc = slice(seq_st, seq_ed), slice(apo_st, apo_ed)
            bb_tok_ids, fa_tok_ids = tokenize_apo(aatypes.cuda(), coords.cuda())
            f_input.sequence.bb_struct_token_id[0, seq_slc] = bb_tok_ids[apo_slc]
            f_input.sequence.fa_struct_token_id[0, seq_slc] = fa_tok_ids[apo_slc]

        # HACK: Set random seed for reproducibility
        # FIXME: pass random generator to model sampling function instead
        set_seed(args.seed)

        # Sample structures
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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

        # Save apo structure
        apo_save_path = save_dir / "apo.cif"
        try:
            writer.write(ref_struct, apo_save_path, save_apo=True)
        except Exception as e:
            tqdm.write(f"Warning: Failed to save apo structure for {name}: {e}")

        # Save sampled coordinates
        sample_coords = model_out["sample_coordinates"]  # [num_samples, Natom, 3]
        # Remove padding atoms to match reference structure
        assert ref_struct.num_atoms == f_input.atom.pad_mask.sum().item()
        num_atoms = ref_struct.num_atoms
        sample_coords_arr = sample_coords[:, :num_atoms, :].cpu().numpy()

        for i in range(args.num_samples):
            save_path = save_dir / f"sample-{i}.cif"
            coords_i = sample_coords_arr[i]
            try:
                writer.write_new_coords(ref_struct, coords_i, save_path)
            except Exception as e:
                tqdm.write(f"Warning: Failed to save sample {i} for {name}: {e}")
    logger.info("Inference completed.")


if __name__ == "__main__":
    main()
