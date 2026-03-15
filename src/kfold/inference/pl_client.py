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

from .query import Query


@dataclass(kw_only=True)
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
    seed: int = 1
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

        # Logger
        self._logger = logging.getLogger("KFoldInferenceClient")
        self._logger.setLevel(logging.INFO)

    # === Main forward method === #
    def forward(
        self,
        f_input: FoldingInput,
        num_recycles: int = 10,
        num_steps: int = 200,
        num_diffusion_samples: int = 5,
    ) -> dict[str, torch.Tensor]:
        dict_out, _ = self.model.sample(
            f_input,
            num_recycles=num_recycles,
            num_steps=num_steps,
            num_diffusion_samples=num_diffusion_samples,
        )
        return dict_out

    def predict_step(
        self,
        batch: tuple[Query, RefStructure, FoldingInput, dict[int, dict]],
    ) -> tuple[Query, RefStructure, dict[str, torch.Tensor]]:
        """Predict step for inference.

        Parameters
        ----------
        batch :
            - Query: the input query.
            - RefStructure: the reference structure for the query.
            - FoldingInput: the input features for the model.
            - dict: a dictionary for apo structure tokenization.
        """
        if batch is None:
            return None  # Skip empty batch (occured by processing error)

        cfg = self.inference_config
        num_trunk_recycles = cfg.num_recycles
        num_diffusion_steps = cfg.num_steps
        num_diffusion_samples = cfg.num_samples
        seed = cfg.seed

        # HACK: Set random seed for reproducibility
        # FIXME: pass random generator to model sampling function instead
        pl.seed_everything(seed, verbose=False)

        # Unpack batch and validate
        query, ref_struct, f_input, apo_dict = batch
        assert f_input.batch_size == 1, "Inference batch size should be 1"

        # === Tokenize apo structure === #
        tokenize_apo = self.model.structure_encoder.tokenize
        for entity_id, apo_info in apo_dict.items():  # noqa
            aatypes, coords = apo_info["aatypes"], apo_info["coords"]
            seq_st, seq_ed, apo_st, apo_ed = apo_info["mapping"]
            seq_slc, apo_slc = slice(seq_st, seq_ed), slice(apo_st, apo_ed)
            bb_tok_ids, fa_tok_ids = tokenize_apo(aatypes, coords)
            f_input.sequence.bb_struct_token_id[0, seq_slc] = bb_tok_ids[apo_slc]
            f_input.sequence.fa_struct_token_id[0, seq_slc] = fa_tok_ids[apo_slc]

        # === Run model inference === #
        try:
            model_out = self(
                f_input=f_input,
                num_recycles=num_trunk_recycles,
                num_steps=num_diffusion_steps,
                num_diffusion_samples=num_diffusion_samples,
            )
        except RuntimeError as e:  # catch out of memory exceptions
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # mmCIF writer
        self.writer = KFoldWriter()
        # Logger
        self.logger = logging.getLogger("KFoldPredictionWriter")

    def write_on_batch_end(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: KFoldInferenceClient,
        prediction: tuple[Query, RefStructure, dict[str, torch.Tensor]],
        batch_indices: list[int],
        batch: list[tuple[Query, RefStructure, FoldingInput, dict[int, dict]]],
        batch_idx: int,
        dataloader_idx: int,
    ):
        """
        Lightning calls this automatically after each predict_step.
        'prediction' is whatever your predict_step returns.
        """
        if prediction is None:
            return

        # Unpack the prediction and batch data
        # Note: We return these from predict_step now
        query, ref_struct, model_out = prediction

        name = query.name
        save_dir = self.output_dir / name
        save_dir.mkdir(exist_ok=True)

        # 1. Save Query YAML
        with open(save_dir / "query.yaml", "w") as f:
            f.write(query.yaml)

        # 2. Save Apo Structure
        try:
            self.writer.write_mmcif(ref_struct, save_dir / "apo.cif", save_apo=True)
        except Exception as e:
            self.logger.error(f"Error saving apo for {name}: {e}")

        # 3. Save Diffusion Samples
        num_atoms = ref_struct.num_atoms
        sample_coords = model_out["sample_coordinates"][:, :num_atoms]
        coords_np = sample_coords.cpu().numpy()
        for i, coord in enumerate(coords_np):
            try:
                save_path = save_dir / f"sample-{i}.cif"
                self.writer.write_new_coords(ref_struct, coord, save_path)
            except Exception as e:
                self.logger.error(f"Error saving sample {i} for {name}: {e}")

        # Free up memory
        model_out.clear()
        del model_out, sample_coords, coords_np
