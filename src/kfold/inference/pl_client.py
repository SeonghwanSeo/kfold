import gc
import logging
import pathlib
from dataclasses import dataclass

import lightning.pytorch as pl
import torch

from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
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
        batch: tuple[Query, RefStructure, FoldingInput, dict[int, dict]],
    ) -> None:
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
            return  # Skip empty batch (occured by processing error)

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
            model_out = self(
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
                self.writer.write_new_coords(ref_struct, sample_coords_arr[i], save_path)
        except Exception as e:
            logger.error(f"Error saving structure for {name}: {e}")
