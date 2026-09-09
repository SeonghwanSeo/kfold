"""Python inference API with one shared AtlasLM and CCD per runner."""

import gc
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import torch

from kfold.inference import assets
from kfold.inference.dataset import InferenceDataset, InferenceInput
from kfold.inference.query import Query, parse_input_files
from kfold.inference.result import FoldingResult
from kfold.inference.structure_tokenization import apply_apo_structure_tokens
from kfold.utils import confidence_metrics


class KFoldRunner:
    """Lazy KFold/AtlasFold models sharing one AtlasLM on a single device.

    Use one runner per GPU/process. ``prepare`` writes apo/prior inputs;
    ``fold`` returns one CPU result without writing files.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda:0",
        cache_dir: str | Path | None = None,
        weight: str | Path | None = None,
        config: str | Path | None = None,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.cache_dir = cache_dir
        self.weight = weight
        self.config = config

    @cached_property
    def ccd(self):
        return assets.load_ccd(self.cache_dir)

    @cached_property
    def atlaslm(self):
        from atlaslm.pretrained import load_model

        source = None
        cache_dir = self.cache_dir
        if self.config is not None:
            from kfold.utils.config import load_config

            config = load_config(self.config).protein_sequence_encoder
            source = config.get("model_path")
            if cache_dir is None:
                cache_dir = config.get("cache_dir")
        if source is None:
            model = load_model(cache_dir=cache_dir, dtype=torch.bfloat16)
        else:
            model = load_model(Path(source), dtype=torch.bfloat16)
        return model.eval().to(self.device)

    @cached_property
    def model(self):
        return (
            assets.load_model(
                self.cache_dir,
                weight=self.weight,
                config=self.config,
                atlaslm=self.atlaslm,
            )
            .eval()
            .to(self.device)
        )

    @cached_property
    def atlasfold(self):
        from atlasfold.pretrained import load_model
        from atlasfold.runner import FoldingRunner

        model = load_model(
            "atlasfold", device=self.device, cache_dir=self.cache_dir, lm=self.atlaslm
        )
        return FoldingRunner(model)

    @cached_property
    def atlasfold_multimer(self):
        from atlasfold.pretrained import load_model
        from atlasfold.runner_multimer import MultimerFoldingRunner

        model = load_model(
            "atlasfold-m", device=self.device, cache_dir=self.cache_dir, lm=self.atlaslm
        )
        return MultimerFoldingRunner(model)

    def read_queries(
        self, path: str | Path, *, seeds: Sequence[int] = (1,)
    ) -> list[Query]:
        return parse_input_files(Path(path), self.ccd, list(seeds), skip_invalid=False)

    def release_apo_models(self) -> None:
        """Release AtlasFold heads while retaining the shared AtlasLM and CCD."""
        loaded = "atlasfold" in self.__dict__ or "atlasfold_multimer" in self.__dict__
        self.__dict__.pop("atlasfold", None)
        self.__dict__.pop("atlasfold_multimer", None)
        if loaded:
            gc.collect()
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

    def prepare(
        self,
        input_path: str | Path,
        out_dir: str | Path,
        *,
        seeds: Sequence[int] = (1,),
        num_samples: int = 5,
        overwrite: bool = False,
    ) -> list[Path]:
        from kfold.inference.preparation import prepare_files

        return prepare_files(
            self,
            input_path,
            out_dir,
            seeds=seeds,
            num_samples=num_samples,
            overwrite=overwrite,
        )

    def prepare_document(self, document: dict, out_dir: str | Path, **kwargs) -> dict:
        """Prepare an in-memory query mapping; returns a new mapping."""
        from kfold.inference.preparation import prepare_document

        return prepare_document(self, document, out_dir, **kwargs)

    def fold(
        self,
        query: Query,
        *,
        num_samples: int = 5,
        num_recycles: int = 10,
        num_steps: int = 100,
        num_apo: int | None = 3,
        return_trajectory: bool = False,
        return_distogram: bool = False,
    ) -> FoldingResult:
        """Predict one parsed query, using its seed, and return CPU results."""
        if num_samples < 1 or num_steps < 1 or num_recycles < 1:
            raise ValueError("num_samples, num_steps and num_recycles must be positive.")
        if num_apo is not None and num_apo < 1:
            raise ValueError("num_apo must be positive.")
        if self.device.type != "cuda":
            raise NotImplementedError("KFold prediction requires a CUDA device.")
        self.release_apo_models()
        item = InferenceDataset([query], self.ccd, num_samples, num_apo)[0]
        return self._fold_input(
            item,
            num_samples=num_samples,
            num_recycles=num_recycles,
            num_steps=num_steps,
            return_trajectory=return_trajectory,
            return_distogram=return_distogram,
        )

    @torch.inference_mode()
    def _fold_input(
        self,
        item: InferenceInput,
        *,
        num_samples: int,
        num_recycles: int,
        num_steps: int,
        return_trajectory: bool,
        return_distogram: bool,
    ) -> FoldingResult:
        query, structure, f_input, records = item
        model = self.model
        f_input = f_input.to(self.device)
        with torch.random.fork_rng(devices=[self.device]), torch.cuda.device(self.device):
            torch.manual_seed(query.seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if model.prot_struct_encoder is not None:
                    apply_apo_structure_tokens(
                        f_input, records, model.prot_struct_encoder
                    )
                output = model.inference(
                    f_input,
                    num_recycles=num_recycles,
                    num_steps=num_steps,
                    num_samples=num_samples,
                    return_traj=return_trajectory,
                )
        summaries, scores = confidence_metrics.summarize_confidence_metrics(
            f_input, structure, output
        )
        coords = (
            output["diffusion"]["coordinates"][:, : structure.num_atoms]
            .float()
            .cpu()
            .numpy()
        )
        distogram = None
        if return_distogram:
            mask = f_input.token.pad_mask
            distogram = {
                "logits": output["distogram"]["logits"][mask][:, mask]
                .half()
                .cpu()
                .numpy(),
                "bin_edges": output["distogram"]["bin_boundaries"].float().cpu().numpy(),
                "asym_ids": f_input.token.asym_id[mask].int().cpu().numpy(),
                "res_ids": f_input.token.residue_index[mask].int().cpu().numpy(),
            }
        trajectory = None
        if return_trajectory:
            trajectory = (
                output["diffusion"]["traj"][:, :, : structure.num_atoms]
                .float()
                .cpu()
                .numpy()
            )
        return FoldingResult(
            query, structure, coords, summaries, scores, distogram, trajectory
        )
