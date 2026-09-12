"""Prepare query structures and generate K-Fold predictions."""

import copy
import gc
import json
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from time import perf_counter

import gemmi
import numpy as np
import torch
from atlasfold.pretrained import load_model as load_atlasfold
from atlasfold.runner import FoldingRunner as MonomerRunner
from atlasfold.runner import ProteinOutput as MonomerOutput
from atlasfold.runner_multimer import MultimerFoldingRunner as MultimerRunner
from atlasfold.runner_multimer import ProteinMultimerOutput as MultimerOutput
from atlasfold.runner_multimer import SamplingConfig
from atlaslm import AtlasLM
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
from numpy.typing import NDArray

from kfold.data.types.ccd import CCD
from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.structure import _read_protein_chain, read_gemmi_structure
from kfold.data.utils.writer import KFoldWriter
from kfold.inference.data_pipeline import InferenceInput, InputDataPipeline
from kfold.inference.query import (
    LigandSequence,
    PolymerSequence,
    ProteinPair,
    ProteinSequence,
    Query,
)
from kfold.inference.utils import align_sequences, atom14_to_atom37
from kfold.model import KFold
from kfold.utils import confidence_metrics

ASSETS_REPO_ID = "SeonghwanSeo/kfold-assets"
ATLASLM_REPO_ID = "SeonghwanSeo/atlaslm-3b-base"
ATLASFOLD_REPO_ID = "SeonghwanSeo/atlasfold-260703"
ATLASFOLD_M_REPO_ID = "SeonghwanSeo/atlasfold-m-260725"
logger = logging.getLogger(__name__)


@dataclass
class ApoChain:
    """One candidate chain aligned to the full query sequence.

    Coordinates and tokens follow sequence positions, with NaN coordinates
    and masked structure tokens for missing residues.
    """

    sequence: str
    coordinates: NDArray[np.float32]  # (L, 37, 3)
    bb_tokens: NDArray[np.int64] | None = None  # (L,)
    fa_tokens: NDArray[np.int64] | None = None  # (L,)

    def __post_init__(self) -> None:
        length = len(self.sequence)
        if self.coordinates.shape != (length, 37, 3):
            raise ValueError("Apo coordinates must have shape (sequence length, 37, 3).")
        for tokens in (self.bb_tokens, self.fa_tokens):
            if tokens is not None and tokens.shape != (length,):
                raise ValueError("Apo tokens must have shape (sequence length,).")


@dataclass
class ApoStructure:
    """One apo/prior candidate with chains in component order.

    chains holds aligned coordinates and tokens; pdb contains only ATOM, HETATM,
    ANISOU, and TER records for saving.
    """

    chains: list[ApoChain]
    pdb: str

    def __post_init__(self) -> None:
        self.pdb = "".join(
            line + "\n"
            for line in self.pdb.splitlines()
            if line[:6].strip() in {"ATOM", "HETATM", "ANISOU", "TER"}
        )

    @classmethod
    def load(cls, paths: list[Path], sequences: list[str]) -> list["ApoStructure"]:
        """Read and align apo/prior candidates without tokenizing them.

        Parameters
        ----------
        paths : list[Path]
            Structure files in candidate order. All models are loaded.
        sequences : list[str]
            One query sequence for a monomer or two for a pair.
            Monomers use the first subchain; pairs use two chains in file order.

        Returns
        -------
        list[ApoStructure]
            Aligned candidates, ordered by file then model.
        """
        structures = []
        for path in paths:
            source = read_gemmi_structure(path)
            if not len(source):
                raise ValueError(f"Structure file contains no models: {path}")
            for model in source:
                if not len(model):
                    raise ValueError(f"Empty model {model.num} in {path}.")
                if len(sequences) == 1:
                    raw_chains = [model.subchains()[0]]
                else:
                    if len(model) != len(sequences) or any(not len(c) for c in model):
                        raise ValueError(
                            f"Expected two nonempty multimer chains in {path}."
                        )
                    raw_chains = list(model)
                chains = []
                for raw_chain, target_sequence in zip(raw_chains, sequences, strict=True):
                    sequence, coords = _read_protein_chain(raw_chain)
                    target_indices, source_indices = align_sequences(
                        target_sequence, sequence
                    )
                    aligned_coords = np.full(
                        (len(target_sequence), 37, 3), np.nan, dtype=np.float32
                    )
                    aligned_coords[target_indices] = coords[source_indices]
                    chains.append(ApoChain(target_sequence, aligned_coords))

                # Preserve all original chains and atom records for saving.
                single = gemmi.Structure()
                single.add_model(model.clone())
                structures.append(cls(chains, single.make_pdb_string()))
        return structures


@dataclass
class FoldingResult:
    """Predictions, confidence metrics, and the apo/prior candidates used as inputs.

    Embeddings and distograms are shared across prediction samples.
    """

    query: Query
    seed: int
    structure: RefStructure
    coordinates: list[np.ndarray]
    confidence_summary: list[dict]
    confidence_scores: list[dict]
    # Candidate ensembles in query.protein_entries order.
    apos: list[list[ApoStructure]]
    priors: list[list[ApoStructure]]
    embeddings: dict[str, np.ndarray] | None = None
    distogram: dict[str, np.ndarray] | None = None
    trajectory: np.ndarray | None = None

    @property
    def num_samples(self) -> int:
        """Return the number of predicted coordinate samples."""
        return len(self.coordinates)

    def save(
        self,
        out_dir: str | Path,
        *,
        save_query: bool = True,
        save_confidence: bool = True,
        save_embeddings: bool = False,
        save_distogram: bool = False,
        save_trajectory: bool = False,
    ) -> None:
        """Write prepared inputs and predictions to an output directory.

        Predictions and confidence summaries are always written. Existing files
        with matching names are replaced; other files in the directory are retained.

        Parameters
        ----------
        out_dir : str | Path
            Destination directory, created if needed.
        save_query : bool
            Write query.json and the prepared apo/prior PDB files. Defaults to true.
        save_confidence : bool
            Write per-sample plddt, pae, and pde arrays as NPZ.
        save_embeddings : bool
            Write shared embeddings if present in this result.
        save_distogram : bool
            Write the shared distogram if present in this result.
        save_trajectory : bool
            Write per-sample trajectories if present in this result.
        """
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        name, seed = self.query.name, self.seed

        def write_pdbs(structures: list[ApoStructure], filename: str) -> None:
            with (out_dir / filename).open("w") as f:
                for model_id, structure in enumerate(structures, start=1):
                    f.write(f"MODEL     {model_id:4d}\n{structure.pdb}ENDMDL\n")
                f.write("END\n")

        # Save relative structure paths without changing the result's query.
        if save_query:
            document = self.query.to_dict()
            (out_dir / "apo").mkdir(exist_ok=True)
            ensembles = iter(zip(self.apos, self.priors, strict=True))
            for index, entry in enumerate(self.query.sequences):
                if not isinstance(entry, (ProteinSequence, ProteinPair)):
                    continue
                apos, priors = next(ensembles)
                entity = document["sequences"][index][entry.kind]
                apo_filename = f"apo/seq-{index}-apo.pdb"
                prior_filename = f"apo/seq-{index}-prior.pdb"
                write_pdbs(apos, apo_filename)
                write_pdbs(priors, prior_filename)
                entity["apo"] = [apo_filename]
                entity["prior"] = [prior_filename]
            (out_dir / "query.json").write_text(json.dumps(document, indent=2) + "\n")

        # Save each sample
        for i in range(self.num_samples):
            prefix = f"{name}_seed-{seed}_sample-{i}"
            coords = self.coordinates[i]
            confidence_summary = self.confidence_summary[i]
            confidence_scores = self.confidence_scores[i]

            structure = self.structure.copy_with_new_coords(
                coords, b_factors=confidence_scores["plddt"]
            )

            # Save structure
            KFoldWriter.write_mmcif(structure, out_dir / f"{prefix}_model.cif")

            # Save confidence summary
            with open(out_dir / f"{prefix}_confidence.json", "w") as f:
                json.dump(confidence_summary, f, indent=2)

            # Save raw confidence scores
            if save_confidence:
                np.savez_compressed(
                    out_dir / f"{prefix}_confidence.npz",
                    **{key: confidence_scores[key] for key in ("plddt", "pae", "pde")},
                )
            # Save trajectory
            if save_trajectory and self.trajectory is not None:
                KFoldWriter.write_trajectory(
                    self.structure,
                    self.trajectory[i],
                    out_dir / f"{prefix}_trajectory.cif",
                )

        if save_embeddings and self.embeddings is not None:
            np.savez_compressed(
                out_dir / f"{name}_seed-{seed}_embeddings.npz", **self.embeddings
            )

        # Save distogram (shared across all samples)
        if save_distogram and self.distogram is not None:
            np.savez_compressed(
                out_dir / f"{name}_seed-{seed}_distogram.npz", **self.distogram
            )


class KFoldRunner:
    """Prepare query structures and generate K-Fold predictions."""

    def __init__(
        self,
        model: KFold,
        *,
        ccd: CCD | None = None,
        lazy_load: bool = False,
        cache_dir: str | Path | None = None,
        share_atlaslm: bool = True,
        verbose: bool = True,
    ):
        """Create a runner for a loaded K-Fold model.

        Parameters
        ----------
        model : KFold
            KFold model already placed on a CUDA device.
        ccd : CCD | None
            Chemical component dictionary. If omitted, load the release CCD.
        share_atlaslm : bool
            Reuse KFold's AtlasLM for apo sampling. If false, the
            apo samplers share a separately loaded AtlasLM.
        cache_dir : str | Path | None
            Hugging Face cache directory for model and CCD downloads.
        lazy_load : bool
            If true, lazy-load the apo samplers on first access. If false,
            load them immediately.
        verbose : bool
            Emit INFO logs for initialization, prediction stages, and elapsed times.

        Raises
        ------
        NotImplementedError
            The model is not on a CUDA device.
        """
        self.verbose = verbose
        self.model: KFold = model
        self.device: torch.device = self.model.device
        if self.device.type != "cuda":
            raise NotImplementedError("KFoldRunner requires a CUDA device.")

        # Download the release CCD and AtlasFold weights.
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        self.download_assets()

        # Load the CCD
        self.ccd: CCD = ccd if ccd is not None else self.load_default_ccd()

        # Initialize the input pipeline.
        self.input_pipeline: InputDataPipeline = InputDataPipeline(self.ccd)

        # Apo sampler setup.
        self.shared_lm: AtlasLM | None = (
            self.model.prot_seq_encoder.lm if share_atlaslm else None
        )
        if not lazy_load:
            # Load the apo samplers
            _ = self.apo_runner
            _ = self.apo_m_runner

    def _log(self, message: str, *args: object) -> None:
        """Emit an INFO message when verbose output is enabled."""
        if self.verbose:
            logger.info(message, *args)

    def download_assets(self) -> None:
        """Download the release CCD and AtlasFold weights to the cache directory."""
        self._log("Checking release assets and downloading missing files.")
        with disable_progress_bars():
            snapshot_download(ASSETS_REPO_ID, cache_dir=self.cache_dir)
            snapshot_download(ATLASLM_REPO_ID, cache_dir=self.cache_dir)
            snapshot_download(ATLASFOLD_REPO_ID, cache_dir=self.cache_dir)
            snapshot_download(ATLASFOLD_M_REPO_ID, cache_dir=self.cache_dir)

    def load_default_ccd(self) -> CCD:
        """Load the release CCD, downloading it if needed.

        Returns
        -------
        CCD
            The loaded chemical component dictionary.
        """
        self._log("Loading CCD from release assets.")
        with disable_progress_bars():
            assets_dir = Path(snapshot_download(ASSETS_REPO_ID, cache_dir=self.cache_dir))
        return CCD.load(assets_dir / "assets/ccd.pkl")

    # === AtlasFold models === #
    @cached_property
    def apo_runner(self) -> MonomerRunner:
        """Load and cache the AtlasFold monomer runner on this runner's device."""
        self._log("Loading AtlasFold monomer model.")
        with disable_progress_bars():
            model = load_atlasfold(
                "atlasfold", self.device, cache_dir=self.cache_dir, lm=self.shared_lm
            )
        self.shared_lm = model.lm
        return MonomerRunner(model)

    @cached_property
    def apo_m_runner(self) -> MultimerRunner:
        """Load and cache the AtlasFold multimer runner on this runner's device."""
        self._log("Loading AtlasFold multimer model.")
        with disable_progress_bars():
            model = load_atlasfold(
                "atlasfold-m", self.device, cache_dir=self.cache_dir, lm=self.shared_lm
            )
        self.shared_lm = model.lm
        return MultimerRunner(model)

    def run_apo_sampler(
        self,
        name: str,
        sequence: str | list[str],
        seeds: list[int],
        num_samples: int = 5,
    ) -> tuple[list[ApoStructure], list[ApoStructure]]:
        """Generate and rank apo candidates without tokenizing them.

        Parameters
        ----------
        name : str
            Target name passed to AtlasFold.
        sequence : str | list[str]
            A monomer sequence or a list of two protein sequences.
        seeds : list[int]
            Generation seeds, processed in the supplied order.
        num_samples : int
            Number of candidates generated per seed.

        Returns
        -------
        apos : list[ApoStructure]
            Best candidate per seed, in seed order.
        priors : list[ApoStructure]
            All candidates, ordered by seed then descending ranking score.
            Includes the same candidate objects selected as apos.
        """
        if isinstance(sequence, list):
            out = self.apo_m_runner.fold(
                name,
                sequence,
                seeds=seeds,
                num_samples=num_samples,
                num_recycles=4,
                mlm_prob=0.15,
                sampling_config=SamplingConfig(num_steps=100),
            )
        else:
            out = self.apo_runner.fold(
                name,
                sequence,
                seeds=seeds,
                num_samples=num_samples,
                num_recycles=4,
                mlm_prob=0.15,
                sampling_config=None,  # dynamic scheduling
            )
        # Group outputs by seed and sort by ranking score.
        apos, priors = [], []
        for seed in seeds:
            group: list[MonomerOutput | MultimerOutput] = sorted(
                (o for (sample_seed, _), o in out.outputs.items() if sample_seed == seed),
                key=lambda output: -float(output.ranking_score),
            )
            candidates: list[ApoStructure] = []
            for output in group:
                multimer = isinstance(output, MultimerOutput)
                sources = output.chains if multimer else [output]
                chains: list[ApoChain] = [
                    ApoChain(
                        source.sequence,
                        atom14_to_atom37(source.sequence, source.coordinates),
                    )
                    for source in sources
                ]
                candidates.append(ApoStructure(chains, output.to_pdb()))
            apos.append(candidates[0])  # Best candidate per seed
            priors.extend(candidates)
        return apos, priors

    def unload_apo_models(self) -> None:
        """Free GPU memory used by the apo samplers."""
        loaded = "apo_runner" in self.__dict__ or "apo_m_runner" in self.__dict__
        self.__dict__.pop("apo_runner", None)
        self.__dict__.pop("apo_m_runner", None)
        if loaded:
            self._log("Releasing apo sampler models.")
            gc.collect()
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

    @torch.inference_mode()
    def tokenize_apo(self, apo: ApoStructure) -> None:
        """Populate an apo candidate's BB/FA structure tokens in place.

        If the model has no protein structure encoder, the candidate is unchanged.

        Parameters
        ----------
        apo : ApoStructure
            Aligned candidate whose chain token fields will be replaced.
        """
        encoder = self.model.prot_struct_encoder
        if encoder is not None:
            for chain in apo.chains:
                encoded = encoder.tokenize(chain.sequence, chain.coordinates)
                chain.bb_tokens = encoded["bb_token_id"].cpu().numpy().copy()
                chain.fa_tokens = encoded["fa_token_id"].cpu().numpy().copy()

    # == Query preparation and validation === #
    def validate_query(self, query: Query) -> None:
        """Check explicit ligand and modification CCD codes before preparation.

        Parameters
        ----------
        query : Query
            Parsed query whose explicit component codes will be checked.

        Raises
        ------
        ValueError
            An explicit component code is absent from this runner's CCD.
        """
        valid_codes = self.ccd.keys()

        ccd_codes = set()
        for entry in query.sequences:
            if isinstance(entry, PolymerSequence):
                ccd_codes.update(modification.ccd for modification in entry.modifications)
            elif isinstance(entry, LigandSequence) and entry.ccd is not None:
                ccd_codes.update(entry.ccd)
            elif isinstance(entry, ProteinPair):
                ccd_codes.update(
                    modification.ccd for modification in entry.modifications1
                )
                ccd_codes.update(
                    modification.ccd for modification in entry.modifications2
                )
        if not ccd_codes.issubset(valid_codes):
            missing = ccd_codes - valid_codes
            raise ValueError(
                f"Query {query.name}: {', '.join(missing)} are missing from CCD."
            )

    @torch.inference_mode()
    def prepare_query(
        self, query: Query, seed: int, num_apos: int = 1
    ) -> tuple[list[list[ApoStructure]], list[list[ApoStructure]]]:
        """Load or generate candidates, then encode apos without changing the query.

        Parameters
        ----------
        query : Query
            Query describing protein entries and optional structure paths.
        seed : int
            Positive base seed used to derive apo generation seeds.
        num_apos : int
            Number of generation seeds per entry without supplied apos, from 1 to 5.
            Each contributes one ranked apo and five prior candidates.

        Returns
        -------
        apos : list[list[ApoStructure]]
            Tokenized apo candidates for each entry in query.protein_entries order.
        priors : list[list[ApoStructure]]
            Prior candidates in the same entry order. Supplied apos are reused
            when no prior paths are given.

        Raises
        ------
        ValueError
            seed is nonpositive, num_apos is outside 1–5, required CCD codes
            are missing, or supplied structures cannot be aligned.
        """
        if seed <= 0:
            raise ValueError(f"Seed must be positive: {seed}.")
        if not 1 <= num_apos <= 5:
            raise ValueError(f"Num_apos must be between 1 and 5: {num_apos}.")

        # Validate query
        self.validate_query(query)

        seeds = [seed * 10 + i for i in range(1, num_apos + 1)]
        apo_ensembles: list[list[ApoStructure]] = []
        prior_ensembles: list[list[ApoStructure]] = []
        for entry_i, entry in enumerate(query.protein_entries):
            multimer = isinstance(entry, ProteinPair)
            sequences = (
                [entry.sequence1, entry.sequence2] if multimer else [entry.sequence]
            )
            if entry.apo:
                apos = ApoStructure.load(entry.apo, sequences)
                priors = (
                    ApoStructure.load(entry.prior, sequences)
                    if entry.prior is not None
                    else apos
                )
            else:
                apos, priors = self.run_apo_sampler(
                    f"{query.name}-{entry_i}",
                    sequences if multimer else entry.sequence,
                    seeds=seeds,
                    num_samples=5,
                )
            for apo in apos:
                self.tokenize_apo(apo)
            apo_ensembles.append(apos)
            prior_ensembles.append(priors)
        return apo_ensembles, prior_ensembles

    @torch.inference_mode()
    def fold(
        self,
        query: Query,
        seed: int,
        *,
        num_apos: int = 1,
        num_samples: int = 5,
        num_recycles: int = 10,
        num_steps: int = 100,
        return_embeddings: bool = False,
        return_trajectory: bool = False,
        return_distogram: bool = False,
    ) -> FoldingResult:
        """Generate missing apos, encode structures, and predict one query.

        Supplied structures are retained, and the caller's query is unchanged.

        Parameters
        ----------
        query : Query
            Parsed query to prepare and predict.
        seed : int
            Positive seed for preparation and model sampling.
        num_apos : int
            Number of generated apos per entry without supplied apos, from 1 to 5.
        num_samples : int
            Number of predictions to generate.
        num_recycles : int
            Number of model recycling iterations.
        num_steps : int
            Number of diffusion sampling steps.
        return_embeddings : bool
            Include shared single and pair embeddings.
        return_trajectory : bool
            Include diffusion trajectories for each sample.
        return_distogram : bool
            Include shared distance logits and bin metadata.

        Returns
        -------
        FoldingResult
            Predictions, confidence metrics, and prepared apo/prior candidates,
            plus the requested optional outputs.

        Raises
        ------
        ValueError
            seed or an inference count is nonpositive, num_apos exceeds 5,
            or query preparation fails validation.
        """
        # Validate inputs and device
        if seed <= 0 or min(num_apos, num_samples, num_recycles, num_steps) < 1:
            raise ValueError("seed and inference counts must be positive.")

        fold_start = perf_counter()
        query = copy.deepcopy(query)
        # Prepare structures and tokens separately from the query.
        prepare_start = perf_counter()
        apos, priors = self.prepare_query(query, seed, num_apos=num_apos)
        generated_entries = [entry for entry in query.protein_entries if not entry.apo]
        if generated_entries:
            num_multimers = sum(
                isinstance(entry, ProteinPair) for entry in generated_entries
            )
            self._log(
                "AtlasFold predict: %.2f s (%d monomer, %d multimer entries).",
                perf_counter() - prepare_start,
                len(generated_entries) - num_multimers,
                num_multimers,
            )
        else:
            self._log(
                "AtlasFold predict: skipped; supplied apos prepared in %.2f s.",
                perf_counter() - prepare_start,
            )

        # Build model features from the query and prepared structures.
        input_start = perf_counter()
        item = self.input_pipeline.build_input(
            query, seed, num_samples, apos=apos, priors=priors
        )
        self._log(
            "Build input: %.2f s (%d chains, %d tokens).",
            perf_counter() - input_start,
            item.ref_struct.num_chains,
            item.ref_struct.num_tokens,
        )

        # Run model
        prediction_start = perf_counter()
        result = self.fold_input(
            item,
            seed,
            num_samples=num_samples,
            num_recycles=num_recycles,
            num_steps=num_steps,
            return_embeddings=return_embeddings,
            return_trajectory=return_trajectory,
            return_distogram=return_distogram,
        )
        self._log(
            "K-Fold predict: %.2f s.",
            perf_counter() - prediction_start,
        )

        self._log("Total pipeline: %.2f s.", perf_counter() - fold_start)
        return result

    @torch.inference_mode()
    def fold_input(
        self,
        item: InferenceInput,
        seed: int,
        *,
        num_samples: int = 5,
        num_recycles: int = 10,
        num_steps: int = 100,
        return_embeddings: bool = False,
        return_distogram: bool = False,
        return_trajectory: bool = False,
    ) -> FoldingResult:
        """Generate predictions from a prepared input.

        Parameters
        ----------
        item : InferenceInput
            Prepared input containing an unbatched FoldingInput.
        seed : int
            Positive model sampling seed; use the input construction seed
            to match the behavior of fold.
        num_samples : int
            Number of diffusion samples. Use the input construction
            count to match fold; a different count reuses or truncates priors.
        num_recycles : int
            Number of model recycling iterations.
        num_steps : int
            Number of diffusion sampling steps.
        return_embeddings : bool
            Include shared single and pair embeddings.
        return_distogram : bool
            Include shared distance logits and bin metadata.
        return_trajectory : bool
            Include diffusion trajectories for each sample.

        Returns
        -------
        FoldingResult
            Predictions, confidence metrics, and prepared apo/prior candidates,
            plus the requested optional outputs.

        Raises
        ------
        ValueError
            The input is batched, or seed or an inference count is
            nonpositive.
        """
        query, struct, f_input = item.query, item.ref_struct, item.f_input
        if seed <= 0:
            raise ValueError(f"Seed must be positive: {seed}.")
        if num_samples <= 0:
            raise ValueError(f"Num_samples must be positive: {num_samples}.")
        if num_recycles <= 0:
            raise ValueError(f"Num_recycles must be positive: {num_recycles}.")
        if num_steps <= 0:
            raise ValueError(f"Num_steps must be positive: {num_steps}.")
        if f_input.is_batched:
            raise ValueError("fold_input expects an unbatched FoldingInput.")

        f_input = f_input.to(self.device)
        n_atoms = struct.num_atoms
        n_tokens = struct.num_tokens

        # Run the model
        with (
            torch.random.fork_rng(devices=[self.device]),
            torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ),
        ):
            torch.manual_seed(seed)
            out = self.model.inference(
                f_input,
                num_recycles=num_recycles,
                num_steps=num_steps,
                num_samples=num_samples,
                return_embeddings=return_embeddings,
                return_distogram=return_distogram,
                return_traj=return_trajectory,
            )

        # Remove padding atoms and convert to numpy array
        coords = list(out["diffusion"]["coordinates"][:, :n_atoms].float().cpu().numpy())

        # Summarize confidence metrics
        summaries, scores = confidence_metrics.summarize_confidence_metrics(
            f_input, struct, out
        )

        embeddings = None
        # Padding is trailing; slices avoid full pair-tensor copies on the GPU.
        if return_embeddings and "trunk" in out:
            embeddings = {
                "s_inputs": out["trunk"]["s_inputs"][:n_tokens].half().cpu().numpy(),
                "s_lm": out["trunk"]["s_lm"][:n_tokens].half().cpu().numpy(),
                "z": out["trunk"]["z"][:n_tokens, :n_tokens].half().cpu().numpy(),
            }

        # Return distogram if requested
        if return_distogram and "distogram" in out:
            dgram_out = out["distogram"]
            distogram = {
                "logits": dgram_out["logits"][:n_tokens, :n_tokens].half().cpu().numpy(),
                "bin_edges": out["distogram"]["bin_boundaries"].float().cpu().numpy(),
                "asym_ids": f_input.token.asym_id[:n_tokens].int().cpu().numpy(),
                "res_ids": f_input.token.residue_index[:n_tokens].int().cpu().numpy(),
            }
        else:
            distogram = None

        trajectory = None
        if return_trajectory:
            trajectory = out["diffusion"]["traj"][:, :, :n_atoms].float().cpu().numpy()

        result = FoldingResult(
            query=query,
            seed=seed,
            structure=struct,
            coordinates=coords,
            confidence_summary=summaries,
            confidence_scores=scores,
            apos=item.apos,
            priors=item.priors,
            embeddings=embeddings,
            distogram=distogram,
            trajectory=trajectory,
        )
        return result
