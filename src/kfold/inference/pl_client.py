import gc
import logging
import pathlib
from dataclasses import dataclass

import lightning.pytorch as pl
import torch

from kfold.data.model_input import FoldingInput
from kfold.data.structure import TokenizedStructure
from kfold.model.models.kfold import KFold

from .query import InputFile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        save_dir: str | pathlib.Path = pathlib.Path("./inference_results/"),
    ):
        super().__init__()
        self.model: KFold = model
        self.inference_config: InferenceConfig = inference_config
        self.save_dir: pathlib.Path = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

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
        batch: tuple[InputFile, TokenizedStructure, FoldingInput],
    ) -> None:
        if batch is None:
            # Skip empty batch (occured by processing error)
            return

        # Unpack batch and validate
        query, struct, f_input = batch
        assert f_input.batch_size == 1, "Inference batch size should be 1"

        cfg = self.inference_config
        num_trunk_recycles = cfg.num_recycles
        num_diffusion_steps = cfg.num_steps
        num_diffusion_samples = cfg.num_samples
        seed = cfg.seed

        # HACK: Set random seed for reproducibility
        # FIXME: pass random generator to model sampling function instead
        pl.seed_everything(seed, verbose=False)

        # === Run model inference === #
        try:
            model_out = self.forward(
                f_input=f_input,
                num_recycles=num_trunk_recycles,
                num_steps=num_diffusion_steps,
                num_diffusion_samples=num_diffusion_samples,
            )
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("**WARNING**: ran out of memory, skipping batch")
                gc.collect()
                torch.cuda.empty_cache()
                return
            else:
                raise e
        # remove batch dimension
        model_out = {k: v.squeeze(0) for k, v in model_out.items()}

        # === Save results === #
        name = query.name
        save_dir = self.save_dir / name
        save_dir.mkdir(exist_ok=True)

        # Save query
        with open(save_dir / "query.yaml", "w") as f:
            f.write(query.yaml)

        # Save predictions
        sample_coords: torch.Tensor = model_out["sample_coordinates"]
        sample_coords_arr = sample_coords.cpu().numpy()
        new_struct = struct.replace_atom_coords(sample_coords_arr)
        try:
            for i in range(num_diffusion_samples):
                save_path = save_dir / f"sample-{i}.cif"
                new_struct.write(save_path, i, is_predicted=True)
        except Exception as e:
            logger.error(f"Error saving structure for {name}: {e}")
