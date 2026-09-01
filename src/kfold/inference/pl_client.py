import gc
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import BasePredictionWriter

from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.model import KFold
from kfold.utils import confidence_metrics

from .affinity import (
    PerQueryAffinityPredictor,
    affinity_prediction_record,
    attach_affinity_prediction,
    validate_affinity_system,
)
from .dataset import InferenceInput
from .query import Query
from .structure_tokenization import apply_apo_structure_tokens


@dataclass
class InferenceConfig:
    """Configuration for inference.

    Attributes
    ----------
    num_recycles : int
        Number of trunk cycles to run during inference.
    num_steps : int
        Number of diffusion steps to run during inference.
    num_samples : int
        Number of samples to generate per input.
    """

    num_recycles: int = 10
    num_steps: int = 100
    num_samples: int = 5
    # TODO: add more hyperparameters as needed


class KFoldInferenceClient(pl.LightningModule):
    """PyTorch Lightning module for inference on multi-gpu environments."""

    def __init__(
        self,
        model: KFold,
        inference_config: InferenceConfig,
        affinity_predictor: PerQueryAffinityPredictor | None = None,
    ):
        super().__init__()
        self.model: KFold = model
        self.inference_config: InferenceConfig = inference_config
        self.num_trunk_recycles: int = inference_config.num_recycles
        self.num_diffusion_steps: int = inference_config.num_steps
        self.num_samples: int = inference_config.num_samples
        self.affinity_predictor = affinity_predictor

        # Logger
        self._logger = logging.getLogger("KFoldInferenceClient")
        self._logger.setLevel(logging.INFO)

    # === Main forward method === #
    def forward(
        self, f_input: FoldingInput, struct_token_records: list[list[dict]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        if self.affinity_predictor is not None:
            validate_affinity_system(
                token_mask=f_input.token.pad_mask,
                chain_type=f_input.token.chain_type,
            )
        if hasattr(self.model, "prot_struct_encoder"):
            apply_apo_structure_tokens(
                f_input, struct_token_records, self.model.prot_struct_encoder
            )
        dict_out, _ = self.model.inference(
            f_input,
            num_recycles=self.num_trunk_recycles,
            num_steps=self.num_diffusion_steps,
            num_samples=self.num_samples,
            return_embeddings=self.affinity_predictor is not None,
        )
        if self.affinity_predictor is not None:
            attach_affinity_prediction(
                dict_out,
                predictor=self.affinity_predictor,
                token_mask=f_input.token.pad_mask,
                chain_type=f_input.token.chain_type,
            )
        return dict_out

    def predict_step(
        self, batch: InferenceInput | None
    ) -> tuple[Query, RefStructure, FoldingInput, dict]:
        """Predict step for inference.

        Parameters
        ----------
        batch :
            - Query: the input query.
            - RefStructure: the reference structure for the query.
            - FoldingInput: the input features for the model.
            - struct_token_records: raw apo structures and target ranges.

        Returns
        -------
        tuple[Query, RefStructure, dict]
            - Query: the input query (same as input).
            - RefStructure: the reference structure (same as input).
            - dict: the model outputs
        """
        if batch is None:
            return None  # Skip empty batch (occured by processing error)

        # Unpack batch and validate
        query, ref_struct, f_input, struct_token_records = batch
        assert query.seed >= 0  # seed should be overridden by user input

        # HACK: Set random seed for reproducibility
        # FIXME: pass random generator to model sampling function instead
        pl.seed_everything(query.seed, verbose=False)

        # === Run model inference === #
        try:
            model_out = self(f_input, struct_token_records)
        except Exception as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                name = query.name
                size = f_input.num_tokens
                self._logger.error(
                    f"Out of memory error for {name} ({size} tokens). "
                    f"Skipping this input."
                )
                gc.collect()
                torch.cuda.empty_cache()
                return None  # type: ignore[return-value]
            else:
                raise e
        return query, ref_struct, f_input, model_out


class KFoldPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str | pathlib.Path,
        queries: list[Query],
        save_trajectory: bool = False,  # TODO: implement trajectory saving
        save_confidence_scores: bool = True,  # TODO: implement confidence score saving
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        save_distogram: bool = False,
    ):
        super().__init__(write_interval)
        self.output_dir = pathlib.Path(output_dir)
        self.queries: list[Query] = queries
        # mmCIF writer
        self.writer = KFoldWriter()
        # Logger
        self.logger = logging.getLogger("KFoldPredictionWriter")

        # Save options
        self.save_trajectory = save_trajectory
        self.save_confidence_scores = save_confidence_scores
        self.save_distogram = save_distogram

    def on_predict_start(self, trainer, pl_module) -> None:
        """Called at the start of prediction."""
        # Ensure all processes wait until directory is created
        trainer.strategy.barrier()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Check output directory
        if trainer.global_rank == 0:
            for query in self.queries:
                query_path = self.output_dir / query.name / "query.yaml"
                if query_path.exists():
                    continue
                query_path.parent.mkdir(parents=True, exist_ok=True)
                query.save(query_path)

        # Ensure all processes wait until directory is created
        trainer.strategy.barrier()

    def write_on_batch_end(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: KFoldInferenceClient,
        prediction: tuple[Query, RefStructure, FoldingInput, dict],
        batch_indices: list[int],
        batch: list[tuple[Query, RefStructure, FoldingInput, list[list[dict]]]],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Lightning calls this automatically after each predict_step.
        'prediction' is whatever your predict_step returns.
        """
        if prediction is None:
            return

        # Unpack the prediction and batch data
        # Note: We return these from predict_step now
        query, ref_struct, f_input, model_out = prediction
        n_atoms = ref_struct.num_atoms

        sample_coords = model_out["diffusion"]["coordinates"][:, :n_atoms].cpu().numpy()
        confidence_summary, confidence_scores = (
            confidence_metrics.summarize_confidence_metrics(
                f_input, ref_struct, model_out
            )
        )

        # Create save directory for this query
        name = query.name
        seed = query.seed
        save_dir = self.output_dir / name / f"{name}_seed-{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # The distogram is shared by all diffusion samples for this query.
        if self.save_distogram:
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

        if "affinity" in model_out:
            affinity = model_out["affinity"]
            predictor = pl_module.affinity_predictor
            assert predictor is not None
            affinity_path = save_dir / f"{name}_seed-{seed}_affinity.json"
            with affinity_path.open("w") as handle:
                json.dump(
                    affinity_prediction_record(affinity, predictor),
                    handle,
                    indent=2,
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
                self.writer.write_new_coords(
                    ref_struct, save_path, coords_i, score_i["plddt"]
                )
            except Exception as e:
                self.logger.error(f"Error saving sample {i} for {name}: {e}")
                continue

            # Save confidence scores in JSON format
            confidence_path = save_dir / f"{sample_name}_confidences.json"
            with open(confidence_path, "w") as f:
                json.dump(summary_i, f, indent=2)

            # Save confidence scores in npz format
            if self.save_confidence_scores:
                confidence_npz_path = save_dir / f"{sample_name}_confidences.npz"
                np.savez_compressed(
                    confidence_npz_path,
                    plddt=score_i["plddt"],
                    pae=score_i["pae"],
                    pde=score_i["pde"],
                )

            # Save trajectory if requested
            if self.save_trajectory:
                raise NotImplementedError("Trajectory saving is not implemented yet.")

        # Free up memory
        model_out.clear()
