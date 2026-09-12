"""Python inference API with one shared AtlasLM and CCD per runner."""

import copy
import gc
import json
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import gemmi
import numpy as np
import torch
from atlasfold.common import residue_constants
from atlasfold.pretrained import load_model as load_atlasfold
from atlasfold.runner import FoldingRunner as MonomerRunner
from atlasfold.runner import ProteinOutput as MonomerOutput
from atlasfold.runner_multimer import MultimerFoldingRunner as MultimerRunner
from atlasfold.runner_multimer import ProteinMultimerOutput as MultimerOutput
from atlaslm import AtlasLM
from huggingface_hub import snapshot_download
from numpy.typing import NDArray

from kfold.constants.atom import protein_atom37_order
from kfold.data.types.ccd import CCD
from kfold.data.types.structure import RefStructure
from kfold.data.utils.io.structure import _read_protein_chain, read_gemmi_structure
from kfold.data.utils.writer import KFoldWriter
from kfold.inference.data_pipeline import InferenceInput, InputDataPipeline
from kfold.inference.query import (
    LigandSequence,
    PolymerSequence,
    ProteinMultimerSequence,
    ProteinSequence,
    Query,
)
from kfold.inference.utils import align_sequences
from kfold.model import KFold
from kfold.utils import confidence_metrics

ASSETS_REPO_ID = "SeonghwanSeo/kfold-assets"
logger = logging.getLogger(__name__)


def load_ccd(
    ccd_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> CCD:
    """Load a supplied CCD or download the release CCD."""
    if ccd_path is None:
        directory = Path(snapshot_download(repo_id=ASSETS_REPO_ID, cache_dir=cache_dir))
        ccd_path = directory / "assets/ccd.pkl"
    logger.info("Loading CCD data from: %s", ccd_path)
    return CCD.load(ccd_path)


@dataclass
class ApoChain:
    """One aligned source chain, before placement on the query sequence.

    ``sequence`` and ``coordinates`` contain only the source overlap.
    ``target`` locates that overlap in the query; tokens follow source residues.
    """

    sequence: str
    coordinates: NDArray[np.float32]  # (L, 37, 3)
    target: slice
    bb_tokens: NDArray[np.int64] | None = None  # (L,)
    fa_tokens: NDArray[np.int64] | None = None  # (L,)


@dataclass
class ApoStructure:
    """One apo/prior candidate, retaining multimer chains in component order.

    PDB text preserves the original model for saving only. Input processing uses
    the chain arrays directly and never parses this text.
    """

    chains: list[ApoChain]
    pdb: str

    @classmethod
    def from_atlasfold(cls, output: MonomerOutput | MultimerOutput) -> "ApoStructure":
        """Convert AtlasFold atom14 arrays directly to KFold atom37 arrays."""
        multimer = isinstance(output, MultimerOutput)
        sources = output.chains if multimer else [output]
        chains = []
        for source in sources:
            coords = np.full((len(source.sequence), 37, 3), np.nan, dtype=np.float32)
            for residue_i, aa in enumerate(source.sequence):
                residue = residue_constants.restype_1to3[aa]
                for atom14_i, atom in enumerate(residue_constants.residue_atoms[residue]):
                    coords[residue_i, protein_atom37_order[atom]] = source.coordinates[
                        residue_i, atom14_i
                    ]
            chains.append(
                ApoChain(source.sequence, coords, slice(0, len(source.sequence)))
            )
        return cls(chains, output.to_pdb(model="multimer" if multimer else "monomer"))

    @classmethod
    def load(cls, paths: list[str], sequences: list[str]) -> list["ApoStructure"]:
        """Read models in file order and align each component to its query sequence."""
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
                    target, overlap = align_sequences(target_sequence, sequence)
                    chains.append(ApoChain(sequence[overlap], coords[overlap], target))

                # Preserve all original chains and atom records for saving.
                single = gemmi.Structure()
                single.add_model(model.clone())
                structures.append(cls(chains, single.make_pdb_string()))
        return structures


@dataclass
class FoldingResult:
    """CPU results: FP16 embeddings/logits, FP32 coordinates/confidence.

    Embeddings contain unpadded token-level ``single`` (s_inputs) and ``pair`` (z)
    arrays, shared across all diffusion samples.
    """

    query: Query
    seed: int
    structure: RefStructure
    coordinates: list[np.ndarray]
    confidence_summary: list[dict]
    confidence_scores: list[dict]
    # Protein entries, then multimer entries; each list contains their candidates.
    apos: list[list[ApoStructure]]
    priors: list[list[ApoStructure]]
    embeddings: dict[str, np.ndarray] | None = None
    distogram: dict[str, np.ndarray] | None = None
    trajectory: np.ndarray | None = None

    @property
    def num_samples(self) -> int:
        return len(self.coordinates)

    def save(
        self,
        out_dir: str | Path,
        *,
        save_confidence: bool = True,
        save_embeddings: bool = False,
        save_distogram: bool = False,
        save_trajectory: bool = False,
    ) -> None:
        """Write prepared inputs and predictions, including confidence summary JSON.

        ``save_confidence`` controls only the raw confidence NPZ file.
        """
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        name, seed = self.query.name, self.seed

        def write_ensemble(structures: list[ApoStructure], filename: str) -> None:
            with (out_dir / filename).open("w") as handle:
                for model_id, structure in enumerate(structures, start=1):
                    lines = [
                        line
                        for line in structure.pdb.splitlines()
                        if line[:6].strip() in {"ATOM", "HETATM", "ANISOU", "TER"}
                    ]
                    handle.write(f"MODEL     {model_id:4d}\n")
                    handle.write("\n".join(lines) + "\nENDMDL\n")
                handle.write("END\n")

        # Canonicalize a separate document; keep the result's in-memory sources.
        document = self.query.to_dict()
        (out_dir / "apo").mkdir(exist_ok=True)
        entry_i = 0
        for section, prefix, entries in (
            ("sequences", "seq", self.query.sequences),
            ("multimer_sequences", "multimer", self.query.multimer_sequences),
        ):
            for index, entry in enumerate(entries):
                if not isinstance(entry, (ProteinSequence, ProteinMultimerSequence)):
                    continue
                entity = document[section][index]["protein"]
                apo_filename = f"apo/{prefix}-{index}-apo.pdb"
                prior_filename = f"apo/{prefix}-{index}-prior.pdb"
                write_ensemble(self.apos[entry_i], apo_filename)
                write_ensemble(self.priors[entry_i], prior_filename)
                entry_i += 1
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
    """K-Fold runner main class"""

    def __init__(
        self,
        model: KFold,
        *,
        ccd: CCD | None = None,
        share_atlaslm: bool = True,
        cache_dir: str | Path | None = None,
    ):
        self.model: KFold = model
        self.device: torch.device = self.model.device
        if self.device.type != "cuda":
            raise NotImplementedError("KFoldRunner requires a CUDA device.")

        self.ccd: CCD = ccd if ccd is not None else load_ccd(cache_dir=cache_dir)
        self.input_pipeline: InputDataPipeline = InputDataPipeline(self.ccd)
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        self.shared_lm: AtlasLM | None = (
            self.model.prot_seq_encoder.lm if share_atlaslm else None
        )

    # === Apo folding heads === #
    @cached_property
    def apo_runner(self) -> MonomerRunner:
        model = load_atlasfold(
            "atlasfold", self.device, cache_dir=self.cache_dir, lm=self.shared_lm
        )
        self.shared_lm = model.lm
        return MonomerRunner(model)

    @cached_property
    def apo_m_runner(self) -> MultimerRunner:
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
        """Return one tokenized apo per seed and all ranked candidates as priors."""
        runner = self.apo_m_runner if isinstance(sequence, list) else self.apo_runner
        out = runner.fold(
            name, sequence, seeds=seeds, num_samples=num_samples, num_recycles=4
        )
        # Group outputs by seed and sort by ranking score.
        apos, priors = [], []
        for seed in seeds:
            group = sorted(
                (o for (sample_seed, _), o in out.outputs.items() if sample_seed == seed),
                key=lambda output: -float(output.ranking_score),
            )
            candidates = [ApoStructure.from_atlasfold(output) for output in group]
            self.tokenize_apo(candidates[0])
            apos.append(candidates[0])
            priors.extend(candidates)
        return apos, priors

    @torch.inference_mode()
    def tokenize_apo(self, apo: ApoStructure) -> None:
        """Encode each component once and keep BB/FA token arrays on CPU."""
        encoder = self.model.prot_struct_encoder
        if encoder is None:
            return
        for chain in apo.chains:
            encoded = encoder.tokenize(chain.sequence, chain.coordinates)
            chain.bb_tokens = encoded["bb_token_id"].cpu().numpy().copy()
            chain.fa_tokens = encoded["fa_token_id"].cpu().numpy().copy()

    def release_apo_models(self) -> None:
        """Release AtlasFold heads."""
        loaded = "apo_runner" in self.__dict__ or "apo_m_runner" in self.__dict__
        self.__dict__.pop("apo_runner", None)
        self.__dict__.pop("apo_m_runner", None)
        if loaded:
            gc.collect()
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

    # == Query preparation and validation === #
    def validate_query(self, query: Query) -> None:
        """Check query sequences and required components against this runner's CCD."""
        valid_codes = self.ccd.keys()

        ccd_codes = set()
        for entry in [*query.sequences, *query.multimer_sequences]:
            if isinstance(entry, PolymerSequence):
                ccd_codes.update(entry.modifications.values())
            elif isinstance(entry, LigandSequence) and entry.ccd_ids is not None:
                ccd_codes.update(entry.ccd_ids)
            elif isinstance(entry, ProteinMultimerSequence):
                ccd_codes.update(entry.modifications1.values())
                ccd_codes.update(entry.modifications2.values())
        ccd_codes = {code.upper() for code in ccd_codes}
        if not ccd_codes.issubset(valid_codes):
            missing = ccd_codes - valid_codes
            raise ValueError(
                f"Query {query.name}: {', '.join(missing)} are missing from CCD."
            )

        for entry in [*query.sequences, *query.multimer_sequences]:
            if not isinstance(entry, (ProteinSequence, ProteinMultimerSequence)):
                continue
            if entry.prior is not None and not entry.apo:
                raise ValueError(
                    f"Query {query.name!r}, chains {entry.ids}: "
                    "'prior' requires supplied 'apo' structures. "
                    "Leave both unset to generate structures."
                )

    @torch.inference_mode()
    def prepare_query(
        self, query: Query, seed: int, num_apos: int = 1
    ) -> tuple[list[list[ApoStructure]], list[list[ApoStructure]]]:
        """Return apo and prior candidates per protein entry without changing Query."""
        if seed <= 0:
            raise ValueError(f"Seed must be positive: {seed}.")
        if num_apos < 1:
            raise ValueError(f"Num_apos must be positive: {num_apos}.")

        # Validate query
        self.validate_query(query)

        seeds = [seed * 10 + i for i in range(1, num_apos + 1)]
        entries = [
            e for e in query.sequences if isinstance(e, ProteinSequence)
        ] + query.multimer_sequences

        apo_ensembles: list[list[ApoStructure]] = []
        prior_ensembles: list[list[ApoStructure]] = []
        for entry_i, entry in enumerate(entries):
            multimer = isinstance(entry, ProteinMultimerSequence)
            sequences = (
                [entry.sequence1, entry.sequence2] if multimer else [entry.sequence]
            )
            if entry.apo:
                apo_structures = ApoStructure.load(entry.apo, sequences)
                for apo in apo_structures:
                    self.tokenize_apo(apo)
                prior_structures = (
                    ApoStructure.load(entry.prior, sequences)
                    if entry.prior is not None
                    else apo_structures
                )
            else:
                apo_structures, prior_structures = self.run_apo_sampler(
                    f"{query.name}-{entry_i}",
                    sequences if multimer else entry.sequence,
                    seeds=seeds,
                )
            apo_ensembles.append(apo_structures)
            prior_ensembles.append(prior_structures)
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

        ``num_apos`` controls generation only for entries without apos. Each
        generation seed contributes one ranked apo and five prior candidates.
        Supplied structures are retained, and the caller's query is unchanged.
        """
        # Validate inputs and device
        if seed <= 0 or min(num_apos, num_samples, num_recycles, num_steps) < 1:
            raise ValueError("seed and inference counts must be positive.")

        query = copy.deepcopy(query)
        # Prepare structures and tokens separately from the query.
        apos, priors = self.prepare_query(query, seed, num_apos=num_apos)

        # Place coordinates and tokens on the query and build model features.
        item = self.input_pipeline.build_input(
            query, seed, num_samples, apos=apos, priors=priors
        )

        # Run model
        return self.fold_input(
            item,
            seed,
            num_samples=num_samples,
            num_recycles=num_recycles,
            num_steps=num_steps,
            return_embeddings=return_embeddings,
            return_trajectory=return_trajectory,
            return_distogram=return_distogram,
        )

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
        """Run KFold on an already constructed input and return CPU results."""
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
        num_atoms = struct.num_atoms
        num_tokens = struct.num_tokens

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
        coords = list(
            out["diffusion"]["coordinates"][:, :num_atoms].float().cpu().numpy()
        )

        # Summarize confidence metrics
        summaries, scores = confidence_metrics.summarize_confidence_metrics(
            f_input, struct, out
        )

        embeddings = None
        # Padding is trailing; slices avoid full pair-tensor copies on the GPU.
        if return_embeddings:
            embeddings = {
                "single": out["trunk"]["s_inputs"][:num_tokens].half().cpu().numpy(),
                "pair": out["trunk"]["z"][:num_tokens, :num_tokens].half().cpu().numpy(),
            }

        # Return distogram if requested
        if return_distogram:
            distogram = {
                "logits": out["distogram"]["logits"][:num_tokens, :num_tokens]
                .half()
                .cpu()
                .numpy(),
                "bin_edges": out["distogram"]["bin_boundaries"].float().cpu().numpy(),
                "asym_ids": f_input.token.asym_id[:num_tokens].int().cpu().numpy(),
                "res_ids": f_input.token.residue_index[:num_tokens].int().cpu().numpy(),
            }
        else:
            distogram = None

        trajectory = None
        if return_trajectory:
            trajectory = out["diffusion"]["traj"][:, :, :num_atoms].float().cpu().numpy()

        return FoldingResult(
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
