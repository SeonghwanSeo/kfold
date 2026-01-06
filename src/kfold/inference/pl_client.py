import gc
import logging
import pathlib
from dataclasses import dataclass

import lightning.pytorch as pl
import torch

from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.model.models.kfold import KFold

from .query import Query

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

        # mmCIF writer
        self.writer = KFoldWriter()

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
        batch: tuple[Query, RefStructure, TokenizedStructure, FoldingInput],
    ) -> None:
        if batch is None:
            # Skip empty batch (occured by processing error)
            return

        # Unpack batch and validate
        query, ref_struct, struct, f_input = batch
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

        try:
            apo_save_path = save_dir / "apo.cif"
            self.writer.write_mmcif(ref_struct, apo_save_path, save_apo=True)
        except Exception as e:
            logger.error(f"Error saving apo structure for {name}: {e}")

        # Save predictions
        sample_coords: torch.Tensor = model_out["sample_coordinates"]
        # Remove padding atoms to match reference structure
        num_atoms = ref_struct.num_atoms
        sample_coords_arr = sample_coords[:, :num_atoms, :].cpu().numpy()
        try:
            for i in range(num_diffusion_samples):
                save_path = save_dir / f"sample-{i}.cif"
                new_struct = ref_struct.copy_with_new_coords(sample_coords_arr[i])
                self.writer.write_mmcif(new_struct, save_path, save_apo=False)
        except Exception as e:
            logger.error(f"Error saving structure for {name}: {e}")
