import argparse
import json
import logging
import pathlib
import time

import numpy as np
import torch
from tqdm import tqdm

from kfold.data.types.ccd import CCD
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.inference.affinity import (
    AUTO_AFFINITY_HEAD_CHECKPOINT,
    PerQueryAffinityConfig,
    PerQueryAffinityPredictor,
    affinity_prediction_record,
    attach_affinity_prediction,
    resolve_affinity_head_checkpoint,
    validate_affinity_system,
)
from kfold.inference.dataset import InferenceDataset
from kfold.inference.query import Query, parse_input_files
from kfold.inference.structure_tokenization import apply_apo_structure_tokens
from kfold.model import KFold
from kfold.utils import confidence_metrics

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
        "--weight",
        type=pathlib.Path,
        help="Path to the model weight file. Required unless --dry-run is used.",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("configs/model/kfold-ecsi.yaml"),
        help="Path to the model configuration file.",
    )
    parser.add_argument(
        "--ccd",
        type=pathlib.Path,
        required=True,
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
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("./inference_results/"),
        help="Root directory to save inference results.",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=[1],
        help="Random seed for inference reproducibility.",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=10,
        help="Number of trunk cycles to run during inference.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of diffusion steps to run during inference.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to generate per input.",
    )
    parser.add_argument(
        "--num-apo",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of apo structures to use per input (default: all).",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Whether to save diffusion trajectory.",
    )
    parser.add_argument(
        "--save-confidence",
        action="store_true",
        help="Whether to save raw confidence scores",
    )
    parser.add_argument(
        "--save-distogram",
        action="store_true",
        help="Save distogram logits, bin edges, and token indices in NPZ format.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads for data loading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing inference results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without model inference",
    )
    parser.add_argument(
        "--affinity",
        "--affinity-head-checkpoint",
        dest="affinity_head_checkpoint",
        type=pathlib.Path,
        nargs="?",
        const=pathlib.Path(AUTO_AFFINITY_HEAD_CHECKPOINT),
        metavar="HEAD_CHECKPOINT",
        help=(
            "Predict p_activity. With no HEAD_CHECKPOINT, load "
            "affinity-cliff-raw1k.ckpt from the backbone directory or weights/. "
            "--affinity-head-checkpoint is retained as a compatibility alias. "
            "The prediction reuses the same trunk/distogram pass with the "
            "submitted CASP16 per-query crop contract."
        ),
    )
    parser.add_argument("--affinity-crop-max-tokens", type=int, default=256)
    parser.add_argument("--affinity-crop-max-protein-tokens", type=int, default=200)
    parser.add_argument("--affinity-pocket-neighborhood-size", type=int, default=10)
    parser.add_argument(
        "--affinity-full-precision-inputs",
        action="store_true",
        help=(
            "Do not emulate the BF16 cache boundary used by the submitted "
            "CASP16 scorer. This changes the numerical inference contract."
        ),
    )
    parser.add_argument(
        "--allow-missing-backbone-keys",
        action="store_true",
        help=(
            "Load a legacy backbone with strict=False. Use only for a known "
            "checkpoint/config compatibility gap."
        ),
    )
    return parser.parse_args()


def dry_run(args):
    # Load CCD data
    logger.info(f"Loading CCD data from: {args.ccd}")
    ccd: CCD = CCD.load(args.ccd)
    logger.info("CCD data loaded successfully.")

    # Parse input query(s)
    # If directory is provided, invalid files are skipped.
    logger.info(f"Parsing input queries from: {args.input}")
    input_queries: list[Query] = parse_input_files(args.input, ccd, args.seed)
    npredict = len(input_queries)
    nseed = len(args.seed)
    nquery = npredict // nseed
    logger.info(
        f"Parsed {npredict} valid input queries: {nquery} samples x {nseed} seeds."
    )
    if len(input_queries) == 0:
        logger.warning("No valid input queries to process. Exiting.")
        return

    # Create data loader
    dataset = InferenceDataset(input_queries, ccd, args.num_samples, args.num_apo)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.num_workers
    )

    # Run inference
    logger.info("Starting inference...")
    st = time.time()
    for input in (pbar := tqdm(dataloader, desc="Inference")):
        if input is None:
            continue  # skip invalid batch
        # Unpack input
        query, ref_struct, f_input, struct_token_records = input  # noqa
        if args.affinity_head_checkpoint is not None:
            validate_affinity_system(
                token_mask=f_input.token.pad_mask,
                chain_type=f_input.token.chain_type,
            )
        pbar.set_postfix({"query": query.name, "num_tokens": ref_struct.num_tokens})

    et = time.time()
    logger.info(f"Dry run completed. ({et - st:.2f} seconds)")


@torch.inference_mode()
def main():
    args = parse_args()

    if args.dry_run:
        logger.info("Performing dry run...")
        dry_run(args)
        return
    if args.weight is None:
        raise ValueError("--weight is required unless --dry-run is used.")
    args.affinity_head_checkpoint = resolve_affinity_head_checkpoint(
        args.affinity_head_checkpoint,
        backbone_checkpoint=args.weight,
    )

    # Setup environment
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")

    # Check output directory
    logger.info(f"Output directory: {args.out_dir}")
    if args.out_dir.exists() and not args.overwrite:
        logger.error(
            f"Output directory {args.out_dir} already exists. "
            f"Use --overwrite to overwrite existing results."
        )
        return

    # Load CCD data
    logger.info(f"Loading CCD data from: {args.ccd}")
    ccd: CCD = CCD.load(args.ccd)
    logger.info("CCD data loaded successfully.")

    # Parse input query(s)
    # If directory is provided, invalid files are skipped.
    logger.info(f"Parsing input queries from: {args.input}")
    input_queries: list[Query] = parse_input_files(args.input, ccd, args.seed)
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
    dataset = InferenceDataset(input_queries, ccd, args.num_samples, args.num_apo)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.num_workers
    )

    # Load model
    logger.info(f"Loading model from weight: {args.weight}")
    model: KFold = KFold.from_checkpoint(
        args.config,
        args.weight,
        strict=not args.allow_missing_backbone_keys,
    )
    model = model.eval().cuda()
    logger.info("Model loaded successfully.")
    affinity_predictor = None
    if args.affinity_head_checkpoint is not None:
        affinity_predictor = PerQueryAffinityPredictor.from_checkpoint(
            args.affinity_head_checkpoint,
            device="cuda",
            config=PerQueryAffinityConfig(
                max_tokens=args.affinity_crop_max_tokens,
                max_protein_tokens=args.affinity_crop_max_protein_tokens,
                neighborhood_size=args.affinity_pocket_neighborhood_size,
                cache_compatible_bfloat16=not args.affinity_full_precision_inputs,
            ),
        ).eval()
        logger.info(
            "Affinity head loaded successfully (sha256=%s).",
            affinity_predictor.checkpoint_sha256,
        )

    # mmCIF writer
    writer = KFoldWriter()

    # Run inference
    logger.info("Starting inference...")
    st = time.time()
    for input in (pbar := tqdm(dataloader, desc="Inference")):
        if input is None:
            continue  # skip invalid batch

        # Unpack input
        query: Query
        ref_struct: RefStructure
        f_input: FoldingInput
        struct_token_records: list[list[dict]]
        query, ref_struct, f_input, struct_token_records = input
        pbar.set_postfix({"query": query.name, "num_tokens": ref_struct.num_tokens})

        name: str = query.name
        seed: int = query.seed
        assert seed >= 0  # Seed should be overridden by user input

        # HACK: Set seed for each sample to ensure reproducibility.
        # TODO: Use torch.Generator.
        set_seed(seed)

        # Move to device
        f_input = f_input.to("cuda")
        if affinity_predictor is not None:
            validate_affinity_system(
                token_mask=f_input.token.pad_mask,
                chain_type=f_input.token.chain_type,
            )

        # Run model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if hasattr(model, "prot_struct_encoder"):
                apply_apo_structure_tokens(
                    f_input, struct_token_records, model.prot_struct_encoder
                )
            model_out, time_log = model.inference(  # noqa
                f_input,
                num_recycles=args.num_recycles,
                num_steps=args.num_steps,
                num_samples=args.num_samples,
                return_embeddings=affinity_predictor is not None,
                return_traj=args.save_trajectory,
            )
            if affinity_predictor is not None:
                attach_affinity_prediction(
                    model_out,
                    predictor=affinity_predictor,
                    token_mask=f_input.token.pad_mask,
                    chain_type=f_input.token.chain_type,
                )

        # NOTE: model_out contains:
        #   - s_inputs: input features [Ntoken, C_s]
        #   - s_trunk: final trunk outputs [Ntoken, C_s]
        #   - z_trunk: final trunk latent [Ntoken, Ntoken, C_z]
        #   - distogram_logits: predicted distogram logits [Ntoken, Ntoken, bin]
        #   - coordinates: generated coordinates [num_samples, Natom, 3]
        #   - traj: (optional) diffusion trajectory [num_samples, num_frames, Natom, 3]
        # *) Ntoken and Natom may be different to original ones due to padding.

        # Save predictions
        save_dir = args.out_dir / name
        assert save_dir.exists()

        n_atoms = ref_struct.num_atoms
        sample_coords = model_out["diffusion"]["coordinates"][:, :n_atoms].cpu().numpy()
        confidence_summary, confidence_scores = (
            confidence_metrics.summarize_confidence_metrics(
                f_input, ref_struct, model_out
            )
        )
        if affinity_predictor is not None:
            affinity_path = save_dir / f"{name}_seed-{seed}_affinity.json"
            affinity_path.write_text(
                json.dumps(
                    affinity_prediction_record(model_out["affinity"], affinity_predictor),
                    indent=2,
                )
                + "\n"
            )

        # The distogram is shared by all diffusion samples for this query.
        if args.save_distogram:
            distogram_path = save_dir / f"{name}_seed-{seed}_distogram.npz"
            mask = f_input.token.pad_mask
            distogram_out = model_out["distogram"]
            logits = distogram_out["logits"][mask][:, mask]
            bin_edges = distogram_out["bin_boundaries"]
            asym_ids = f_input.token.asym_id[mask]
            res_ids = f_input.token.residue_index[mask]
            np.savez_compressed(
                distogram_path,
                logits=logits.half().cpu().numpy(),
                bin_edges=bin_edges.float().cpu().numpy(),
                asym_ids=asym_ids.int().cpu().numpy(),
                res_ids=res_ids.int().cpu().numpy(),
            )

        # Save Diffusion Samples
        for i in range(sample_coords.shape[0]):
            sample_name = f"{name}_seed-{seed}_sample-{i}"
            save_path = save_dir / f"{sample_name}.cif"

            # Outputs to save
            coords_i = sample_coords[i]
            summary_i = confidence_summary[i]
            score_i = confidence_scores[i]

            try:
                writer.write_new_coords(ref_struct, save_path, coords_i, score_i["plddt"])
            except Exception as e:
                logger.error(f"Error saving sample {i} for {name}: {e}")
                continue

            # Save confidence scores in JSON format
            confidence_path = save_dir / f"{sample_name}_confidences.json"
            with open(confidence_path, "w") as f:
                json.dump(summary_i, f, indent=2)

            # Save confidence scores in npz format
            if args.save_confidence:
                confidence_npz_path = save_dir / f"{sample_name}_confidences.npz"
                np.savez_compressed(
                    confidence_npz_path,
                    plddt=score_i["plddt"],
                    pae=score_i["pae"],
                    pde=score_i["pde"],
                )

        if args.save_trajectory:
            # Save trajectory
            # [num_samples, num_frames, Natom, 3]
            traj_coords = model_out["diffusion"]["traj"][:, :, :n_atoms, :]
            traj_coords = traj_coords.cpu().numpy()
            for i in range(args.num_samples):
                save_path = save_dir / f"{name}_seed-{seed}_sample-{i}_traj.pdb"
                coords_i = traj_coords[i]
                try:
                    writer.write_trajectory(ref_struct, coords_i, save_path)
                except Exception as e:
                    logger.error(
                        f"Failed to save trajectory for sample {i} of {name}: {e}"
                    )

    et = time.time()
    logger.info(f"Inference completed. ({et - st:.2f} seconds)")


if __name__ == "__main__":
    main()
