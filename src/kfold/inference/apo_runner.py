# Copyright 2026 Korea Advanced Institute of Science and Technology (KAIST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load, predict, and save apo structures independently of K-Fold."""

import gc
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from atlasfold.pretrained import load_model as load_atlasfold
from atlasfold.runner import FoldingRunner as MonomerRunner
from atlasfold.runner import SamplingConfig
from atlasfold.runner_multimer import MultimerFoldingRunner as MultimerRunner
from atlaslm import AtlasLM
from atlaslm.pretrained import load_model as load_atlaslm
from numpy.typing import NDArray

from kfold.data.utils.io.structure import _read_protein_chain, read_gemmi_structure
from kfold.inference.utils import (
    align_sequences,
    atom14_to_atom37,
    write_protein_pdbs,
)

logger = logging.getLogger(__name__)


@dataclass
class AtlasFoldConfig:
    """AtlasFold settings."""

    num_recycles: int = 4
    mlm_prob: float = 0.15
    num_samples: int = 5
    num_steps: int | None = None

    def __post_init__(self) -> None:
        for name in ("num_recycles", "num_samples", "num_steps"):
            value = getattr(self, name)
            if name == "num_steps" and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.mlm_prob) not in (int, float) or not 0 < self.mlm_prob <= 1:
            raise ValueError("mlm_prob must be a number greater than 0 and at most 1.")


@dataclass
class ApoConfig:
    """Apo batching settings and per-model sampling configurations."""

    max_tokens_per_batch: int = 1024
    monomer: AtlasFoldConfig = field(default_factory=AtlasFoldConfig)
    multimer: AtlasFoldConfig = field(
        default_factory=lambda: AtlasFoldConfig(num_steps=100)
    )

    def __post_init__(self) -> None:
        if type(self.max_tokens_per_batch) is not int or self.max_tokens_per_batch < 1:
            raise ValueError("max_tokens_per_batch must be a positive integer.")
        for name in ("monomer", "multimer"):
            if not isinstance(getattr(self, name), AtlasFoldConfig):
                raise ValueError(f"{name} must be an AtlasFoldConfig object.")

    @classmethod
    def load(cls, path: str | Path) -> "ApoConfig":
        """Load and validate YAML overrides, retaining omitted settings' defaults.

        Parameters
        ----------
        path : str | Path
            YAML file containing apo settings.

        Returns
        -------
        ApoConfig
            Validated settings.

        Raises
        ------
        ValueError
            The configuration has invalid fields, types, or values.
        """
        # Read YAML overrides and validate the top-level configuration fields.
        path = Path(path)
        if path.suffix != ".yaml":
            raise ValueError("Apo configuration must be a YAML file.")
        with path.open() as f:
            overrides = yaml.safe_load(f)
        if not isinstance(overrides, dict) or any(
            not isinstance(key, str) for key in overrides
        ):
            raise ValueError("Apo configuration must be a mapping with string keys.")
        if unknown := overrides.keys() - {item.name for item in fields(cls)}:
            raise ValueError(f"Unknown apo settings: {sorted(unknown)}.")
        # Apply each model's overrides to its defaults and validate the result.
        defaults = cls()
        for model in ("monomer", "multimer"):
            settings = overrides.get(model, {})
            if not isinstance(settings, dict) or any(
                not isinstance(key, str) for key in settings
            ):
                raise ValueError(f"{model} settings must be a mapping with string keys.")
            if unknown := settings.keys() - {
                item.name for item in fields(AtlasFoldConfig)
            }:
                raise ValueError(f"Unknown {model} settings: {sorted(unknown)}.")
            try:
                overrides[model] = replace(getattr(defaults, model), **settings)
            except ValueError as error:
                raise ValueError(f"{model}: {error}") from error
        return cls(**overrides)


@dataclass
class ApoChain:
    sequence: str
    coordinates: NDArray[np.float32]  # (L, 37, 3)
    bb_tokens: NDArray[np.int64] | None = None  # (L,)
    fa_tokens: NDArray[np.int64] | None = None  # (L,)


@dataclass
class ApoMultimer:
    chain1: ApoChain
    chain2: ApoChain


@dataclass
class ApoPrediction:
    """Ranked prediction ensembles for one named apo target."""

    name: str
    apos: list[ApoChain] | list[ApoMultimer]
    priors: list[ApoChain] | list[ApoMultimer]


def apo_output_prefix(sequence_index: int, *, multimer: bool) -> str:
    """Return a monomer/multimer filename prefix for a 1-based query position."""
    if sequence_index < 1:
        raise ValueError("sequence_index must be a positive, 1-based query position.")
    return f"{'multimer' if multimer else 'monomer'}-{sequence_index}"


def save_apo_and_prior(
    apos: list[ApoChain] | list[ApoMultimer],
    priors: list[ApoChain] | list[ApoMultimer],
    out_dir: Path,
    sequence_index: int,
) -> tuple[Path, Path]:
    """Save one entry's apo and prior ensembles and return their absolute paths.

    Parameters
    ----------
    apos : list[ApoChain] | list[ApoMultimer]
        Apo predictions in model order.
    priors : list[ApoChain] | list[ApoMultimer]
        Prior predictions in model order.
    out_dir : Path
        Query/seed output directory.
    sequence_index : int
        One-based position in query.sequences, shared across monomers and multimers.

    Returns
    -------
    tuple[Path, Path]
        Absolute paths to the saved apo and prior PDB files.
    """
    # Validate ensembles and resolve their shared output naming.
    if not apos or not priors:
        raise ValueError("Cannot save an empty apo/prior ensemble.")
    prefix = apo_output_prefix(sequence_index, multimer=isinstance(apos[0], ApoMultimer))
    apo_dir = out_dir.resolve() / "apo"
    apo_dir.mkdir(parents=True, exist_ok=True)
    paths = (apo_dir / f"{prefix}_apo.pdb", apo_dir / f"{prefix}_prior.pdb")
    # Convert candidates into PDB models, preserving paired chains together.
    for predictions, path in zip((apos, priors), paths, strict=True):
        models = []
        for prediction in predictions:
            if isinstance(prediction, ApoMultimer):
                chains = (prediction.chain1, prediction.chain2)
            else:
                chains = (prediction,)
            models.append([(chain.sequence, chain.coordinates) for chain in chains])
        write_protein_pdbs(models, path)
    return paths


def _load_chain(raw_chain, sequence: str) -> ApoChain:
    """Read a chain and align its atom37 coordinates to the query sequence."""
    source_sequence, coordinates = _read_protein_chain(raw_chain)
    target_indices, source_indices = align_sequences(sequence, source_sequence)
    aligned = np.full((len(sequence), 37, 3), np.nan, dtype=np.float32)
    aligned[target_indices] = coordinates[source_indices]
    return ApoChain(sequence, aligned)


def load_monomer_apo(path: Path, sequence: str) -> list[ApoChain]:
    return [
        _load_chain(model.subchains()[0], sequence)
        for model in read_gemmi_structure(path)
    ]


def load_multimer_apo(path: Path, sequence1: str, sequence2: str) -> list[ApoMultimer]:
    predictions = []
    for model in read_gemmi_structure(path):
        chain1, chain2 = model
        predictions.append(
            ApoMultimer(_load_chain(chain1, sequence1), _load_chain(chain2, sequence2))
        )
    return predictions


class ApoRunner:
    def __init__(
        self,
        device: torch.device | str = "cuda",
        *,
        kernel_backend: str = "auto",
        config: ApoConfig | None = None,
        cache_dir: str | Path | None = None,
        lm: AtlasLM | None = None,
        verbose: bool = True,
    ) -> None:
        """K-Fold apo structure generator using AtlasFold.

        Parameters
        ----------
        device : torch.device | str
            Model device.
        kernel_backend : str
            AtlasFold kernel backend: auto (default), torch, cuequiv, or triton.
            With auto, use AtlasFold automatic selection. None is not accepted.
        config : ApoConfig | None
            Apo generation settings.
        cache_dir : str | Path | None
            Hugging Face cache directory.
        lm : AtlasLM | None
            Existing AtlasLM to share; otherwise load it.
        verbose : bool
            Emit model-loading logs.
        """
        self.device = torch.device(device)
        if kernel_backend not in ("auto", "torch", "triton", "cuequiv"):
            raise ValueError(
                f"Unknown kernel_backend {kernel_backend!r}. "
                "Expected 'auto', 'torch', 'triton', or 'cuequiv'."
            )
        self.kernel_backend = kernel_backend
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.config: ApoConfig = config if config is not None else ApoConfig()
        self.verbose: bool = verbose

        # Translate settings into the native AtlasFold runner arguments.
        self.monomer_config = {
            "num_recycles": self.config.monomer.num_recycles,
            "mlm_prob": self.config.monomer.mlm_prob,
            "num_samples": self.config.monomer.num_samples,
            "sampling_config": SamplingConfig(num_steps=self.config.monomer.num_steps)
            if self.config.monomer.num_steps is not None
            else None,
            "max_tokens_per_batch": self.config.max_tokens_per_batch,
        }
        self.multimer_config = {
            "num_recycles": self.config.multimer.num_recycles,
            "mlm_prob": self.config.multimer.mlm_prob,
            "num_samples": self.config.multimer.num_samples,
            "sampling_config": SamplingConfig(num_steps=self.config.multimer.num_steps)
            if self.config.multimer.num_steps is not None
            else None,
            "max_tokens_per_batch": self.config.max_tokens_per_batch,
        }

        # Share one AtlasLM; folding models are loaded on their first use.
        if lm is None:
            self._log("Loading AtlasLM.")
            lm = load_atlaslm(device=self.device, cache_dir=self.cache_dir)
        self.lm = lm
        self._runners: dict[str, MonomerRunner | MultimerRunner] = {}

    def _log(self, message: str, *args: object) -> None:
        if self.verbose:
            logger.info(message, *args)

    def load_model(self, *, multimer: bool = False) -> MonomerRunner | MultimerRunner:
        """Load the requested folding model, reusing it across prediction calls.

        Each model is loaded once and retained until explicitly unloaded.
        AtlasLM is shared across both types.

        Parameters
        ----------
        multimer : bool
            Load AtlasFold-Multimer instead of AtlasFold.

        Returns
        -------
        MonomerRunner | MultimerRunner
            Runner for the loaded model.
        """
        # K-Fold may have offloaded the shared AtlasLM since the last prediction.
        self.lm.to(self.device)
        model_name = "atlasfold-m" if multimer else "atlasfold"
        if model_name not in self._runners:
            self._log("Loading %s.", "AtlasFold-Multimer" if multimer else "AtlasFold")
            model = load_atlasfold(
                model_name,
                self.device,
                kernel=self.kernel_backend,
                cache_dir=self.cache_dir,
                lm=self.lm,
            )
            runner_class = MultimerRunner if multimer else MonomerRunner
            self._runners[model_name] = runner_class(model)
        return self._runners[model_name]

    def unload_models(self) -> None:
        """Release cached folding models and unused CUDA memory, retaining AtlasLM.

        Prediction iterators must be exhausted or closed before unloading.
        """
        if self._runners:
            self._runners.clear()
            gc.collect()
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

    def predict(
        self,
        name: str,
        sequence: str | tuple[str, str],
        seeds: list[int],
    ) -> ApoPrediction:
        """Generate ranked apo and prior predictions for one target.

        Parameters
        ----------
        name : str
            Target name passed to AtlasFold.
        sequence : str | tuple[str, str]
            A monomer sequence or two protein sequences.
        seeds : list[int]
            Generation seeds in output order.

        Returns
        -------
        ApoPrediction
            Best apo per seed and all prior predictions, ordered by seed and
            descending ranking score within each seed.
        """
        return next(self.predict_iter([(name, sequence)], seeds))

    def predict_iter(
        self,
        inputs: list[tuple[str, str]] | list[tuple[str, tuple[str, str]]],
        seeds: list[int],
    ) -> Iterator[ApoPrediction]:
        """Yield one target's ranked apo and prior predictions at a time.

        Parameters
        ----------
        inputs : list[tuple[str, str | tuple[str, str]]]
            Named targets of the same type: all monomers or all protein pairs.
            AtlasFold processes length buckets in ascending order.
        seeds : list[int]
            Generation seeds shared by every target in this call.

        Yields
        ------
        ApoPrediction
            Ranked apo and prior predictions for one target.
        """
        for batch in self.predict_iter_batch(inputs, seeds):
            yield from batch

    def predict_iter_batch(
        self,
        inputs: list[tuple[str, str]] | list[tuple[str, tuple[str, str]]],
        seeds: list[int],
    ) -> Iterator[list[ApoPrediction]]:
        """Yield predictions together for each AtlasFold model batch.

        Parameters
        ----------
        inputs : list[tuple[str, str | tuple[str, str]]]
            Named targets of the same type: all monomers or all protein pairs.
        seeds : list[int]
            Generation seeds shared by every target.

        Yields
        ------
        list[ApoPrediction]
            Ranked apo and prior ensembles for one batch.
        """
        if len(inputs) == 0:
            raise ValueError("Empty inputs list.")

        multimer = isinstance(inputs[0][1], tuple)

        # Select the folding model and sampling settings for this input type.
        runner = self.load_model(multimer=multimer)
        config = self.multimer_config if multimer else self.monomer_config
        # Predict native batches, then collect candidates for each target and seed.
        for outputs in runner.fold_iter_batch(inputs, seeds=seeds, **config):
            batch = []
            for out in outputs:
                apos, priors = [], []
                # Preserve seed order and rank candidates within each seed.
                for seed in seeds:
                    samples = [
                        sample for (s, _), sample in out.outputs.items() if s == seed
                    ]
                    samples.sort(key=lambda sample: -float(sample.ranking_score))
                    # Convert native atom14 predictions into K-Fold atom37 candidates.
                    predictions = []
                    for sample in samples:
                        if multimer:
                            chain1, chain2 = sample.chains
                            prediction = ApoMultimer(
                                ApoChain(
                                    chain1.sequence,
                                    atom14_to_atom37(chain1.sequence, chain1.coordinates),
                                ),
                                ApoChain(
                                    chain2.sequence,
                                    atom14_to_atom37(chain2.sequence, chain2.coordinates),
                                ),
                            )
                        else:
                            prediction = ApoChain(
                                sample.sequence,
                                atom14_to_atom37(sample.sequence, sample.coordinates),
                            )
                        predictions.append(prediction)
                    # Keep the best candidate as apo and all candidates as priors.
                    apos.append(predictions[0])
                    priors.extend(predictions)
                batch.append(ApoPrediction(out.name, apos, priors))
            yield batch
