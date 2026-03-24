import argparse
import logging
import pathlib
import time

import torch
from tqdm import tqdm

from kfold.data.types.ccd import CCD
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.inference.dataset import InferenceDataset
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
        "--ccd",
        type=pathlib.Path,
        default=pathlib.Path(
            "/mnt/parallel_storage/wykim_lab/icl_shwan/data/ccd-test.pkl"
        ),
        help="Path to the CCD data file.",
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
        nargs="+",
        type=int,
        default=[42],
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
        "--save_trajectory",
        action="store_true",
        help="Whether to save diffusion trajectory.",
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
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing inference results.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "OmegaConf dotlist override applied before model construction, "
            "e.g. model.structure_module.sampling_schedule_type=phase_power"
        ),
    )
    return parser.parse_args()


@torch.inference_mode()
def main():
    # Setup environment
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")

    args = parse_args()

    # Check output directory
    logger.info(f"Output directory: {args.out_dir}")
    if args.out_dir.exists() and not args.overwrite:
        logger.error(
            f"Output directory {args.out_dir} already exists. "
            f"Use --overwrite to overwrite existing results."
        )
        return

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
        args.input, ccd, args.seed, skip_invalid=True
    )
    npredict = len(input_queries)
    nseed = len(args.seed)
    nquery = npredict // nseed
    logger.info(
        f"Parsed {npredict} valid input queries: {nquery} samples x {nseed} seeds."
    )
    if len(input_queries) == 0:
        logger.warning("No valid input queries to process. Exiting.")
        return

    # Create output directories for each query
    save_dir = args.out_dir
    for query in input_queries:
        query_path = save_dir / query.name / "query.yaml"
        if query_path.exists():
            continue  # skip if already exists
        query_path.parent.mkdir(parents=True, exist_ok=True)
        query.save(query_path)

    # Create data loader
    dataset = InferenceDataset(
        input_queries, ccd, args.num_samples, args.use_sequence_masking
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.num_workers
    )

    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model: KFold = KFold.from_checkpoint(
        args.config, args.checkpoint, override_args=args.override
    )
    model = model.cast_to_bf16().eval().cuda()
    logger.info("Model loaded successfully.")

    # mmCIF writer
    writer = KFoldWriter()

    # Run inference
    logger.info("Starting inference...")
    st = time.time()
    for batch in tqdm(dataloader, desc="Inference"):
        if batch is None:
            continue  # skip invalid batch

        # Unpack batch
        query: Query = batch[0]
        ref_struct: RefStructure = batch[1]
        f_input: FoldingInput = batch[2]
        apo_dict: dict[int, dict] = batch[3]  # (entity_id -> apo_info)
        assert f_input.batch_size == 1, "Inference batch size should be 1"

        name: str = query.name
        seed: int = query.seed
        assert seed >= 0  # Seed should be overridden by user input

        # HACK: Set seed for each sample to ensure reproducibility.
        # TODO: Use torch.Generator.
        set_seed(seed)

        # Move to device
        to_cuda = lambda x: x.cuda() if isinstance(x, torch.Tensor) else x  # noqa
        f_input = f_input.to("cuda")
        apo_dict = {
            eid: {k: to_cuda(v) for k, v in dic.items()} for eid, dic in apo_dict.items()
        }

        # Run model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model_out, time_log = model.inference(  # noqa
                f_input,
                apo_dict,
                num_recycles=args.num_recycles,
                num_steps=args.num_steps,
                num_samples=args.num_samples,
                return_traj=args.save_trajectory,
            )

        # NOTE: model_out contains:
        #   - s_inputs: input features [Ntoken, C_s]
        #   - s_trunk: final trunk outputs [Ntoken, C_s]
        #   - z_trunk: final trunk latent [Ntoken, Ntoken, C_z]
        #   - distogram_logits: predicted distogram logits [Ntoken, Ntoken, bin]
        #   - sample_coordinates: generated coordinates [num_samples, Natom, 3]
        #   - traj: (optional) diffusion trajectory [num_samples, num_frames, Natom, 3]
        # *) Ntoken and Natom may be different to original ones due to padding.

        # Save predictions
        save_dir = args.out_dir / name
        assert save_dir.exists()

        # Save sampled coordinates
        sample_coords = model_out["sample_coordinates"]  # [num_samples, Natom, 3]
        # Remove padding atoms to match reference structure
        assert ref_struct.num_atoms == f_input.atom.pad_mask.sum().item()
        num_atoms = ref_struct.num_atoms
        sample_coords_arr = sample_coords[:, :num_atoms, :].cpu().numpy()

        for i in range(args.num_samples):
            save_path = save_dir / f"{name}_seed-{seed}_sample-{i}.cif"
            coords_i = sample_coords_arr[i]
            try:
                writer.write_new_coords(ref_struct, coords_i, save_path)
            except Exception as e:
                logger.error(f"Warning: Failed to save sample {i} for {name}: {e}")

        # Save trajectory
        # [num_samples, num_frames, Natom, 3]
        traj_coords = model_out["traj"][:, :, :num_atoms, :]
        traj_coords = traj_coords.cpu().numpy()
        for i in range(args.num_samples):
            save_path = save_dir / f"{name}_seed-{seed}_sample-{i}_traj.pdb"
            coords_i = traj_coords[i]
            try:
                writer.write_trajectory(ref_struct, coords_i, save_path)
            except Exception as e:
                logger.error(
                    f"Warning: Failed to save trajectory for sample {i} of {name}: {e}"
                )

    et = time.time()
    logger.info(f"Inference completed. ({et - st:.2f} seconds)")


if __name__ == "__main__":
    main()
