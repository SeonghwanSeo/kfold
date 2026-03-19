import gc
import logging
import pathlib
from dataclasses import dataclass
from typing import Literal

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter

from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.model.models.kfold import KFold

from .dataset import InferenceBatch
from .query import Query


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
    num_steps: int = 200
    num_samples: int = 5
    # TODO: add more hyperparameters as needed


class KFoldInferenceClient(pl.LightningModule):
    """PyTorch Lightning module for inference on multi-gpu environments."""

    def __init__(
        self,
        model: KFold,
        inference_config: InferenceConfig,
    ):
        super().__init__()
        self.model: KFold = model
        self.model.cast_to_bf16()  # Use bfloat16 to save memory and speed up
        self.inference_config: InferenceConfig = inference_config
        self.num_trunk_recycles: int = inference_config.num_recycles
        self.num_diffusion_steps: int = inference_config.num_steps
        self.num_diffusion_samples: int = inference_config.num_samples

        # Logger
        self._logger = logging.getLogger("KFoldInferenceClient")
        self._logger.setLevel(logging.INFO)

    # === Main forward method === #
    def forward(
        self, f_input: FoldingInput, apo_dict: dict[int, dict]
    ) -> dict[str, torch.Tensor]:
        dict_out, _ = self.model.inference(
            f_input,
            apo_dict,
            num_recycles=self.num_trunk_recycles,
            num_steps=self.num_diffusion_steps,
            num_samples=self.num_diffusion_samples,
        )
        return dict_out

    def predict_step(
        self, batch: InferenceBatch | None
    ) -> tuple[Query, RefStructure, dict]:
        """Predict step for inference.

        Parameters
        ----------
        batch :
            - Query: the input query.
            - RefStructure: the reference structure for the query.
            - FoldingInput: the input features for the model.
            - apo_dict: a dictionary for apo structure tokenization.

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
        query, ref_struct, f_input, apo_dict = batch
        assert f_input.batch_size == 1, "Inference batch size should be 1"
        assert query.seed >= 0  # seed should be overridden by user input

        # HACK: Set random seed for reproducibility
        # FIXME: pass random generator to model sampling function instead
        pl.seed_everything(query.seed, verbose=False)

        # === Run model inference === #
        try:
            model_out = self(f_input, apo_dict)
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

        # Remove batch dimension from outputs
        model_out = {k: v.squeeze(0) for k, v in model_out.items()}
        return query, ref_struct, model_out


class KFoldPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str | pathlib.Path,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ):
        super().__init__(write_interval)
        self.output_dir = pathlib.Path(output_dir)
        # mmCIF writer
        self.writer = KFoldWriter()
        # Logger
        self.logger = logging.getLogger("KFoldPredictionWriter")

    def write_on_batch_end(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: KFoldInferenceClient,
        prediction: tuple[Query, RefStructure, dict],
        batch_indices: list[int],
        batch: list[tuple[Query, RefStructure, FoldingInput, dict[int, dict]]],
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
        query, ref_struct, model_out = prediction

        # Create save directory for this query
        name = query.name
        seed = query.seed
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save Diffusion Samples
        num_atoms = ref_struct.num_atoms
        sample_coords = model_out["sample_coordinates"][:, :num_atoms]
        coords_np = sample_coords.cpu().numpy()
        for i, coord in enumerate(coords_np):
            try:
                save_path = save_dir / f"{name}_seed-{seed}_sample-{i}.cif"
                self.writer.write_new_coords(ref_struct, coord, save_path)
            except Exception as e:
                self.logger.error(f"Error saving sample {i} for {name}: {e}")

        # Free up memory
        model_out.clear()
        del model_out, sample_coords, coords_np
