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
from atlasfold.pretrained import load_model as load_atlasfold
from atlasfold.runner import FoldingRunner as MonomerRunner
from atlasfold.runner import ProteinOutput as MonomerOutput
from atlasfold.runner_multimer import MultimerFoldingRunner as MultimerRunner
from atlasfold.runner_multimer import ProteinMultimerOutput as MultimerOutput
from atlaslm import AtlasLM
from huggingface_hub import snapshot_download

from kfold.data.types.ccd import CCD
from kfold.data.types.structure import RefStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.inference.data_pipeline import InferenceInput, InputDataPipeline
from kfold.inference.query import (
    LigandSequence,
    PolymerSequence,
    ProteinMultimerSequence,
    ProteinSequence,
    Query,
    validate_input_sequences,
)
from kfold.inference.utils import (
    align_structures,
    encode_apo_tokens,
    read_pdbs,
    resolve_structure_chains,
)
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
class FoldingResult:
    query: Query
    seed: int
    structure: RefStructure
    coordinates: list[np.ndarray]
    confidence_summary: list[dict]
    confidence_scores: list[dict]
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
        save_distogram: bool = False,
        save_trajectory: bool = False,
    ) -> None:
        """Write prepared inputs and predictions, including confidence summary JSON.

        ``save_confidence`` controls only the raw confidence NPZ file.
        """
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        name, seed = self.query.name, self.seed

        def write_ensemble(pdbs: list[str], filename: str) -> None:
            with (out_dir / filename).open("w") as handle:
                for model_id, pdb in enumerate(pdbs, start=1):
                    lines = [
                        line
                        for line in pdb.splitlines()
                        if line[:6].strip() in {"ATOM", "HETATM", "ANISOU", "TER"}
                    ]
                    handle.write(f"MODEL     {model_id:4d}\n")
                    handle.write("\n".join(lines) + "\nENDMDL\n")
                handle.write("END\n")

        # Canonicalize a separate document; keep the result's in-memory sources.
        document = self.query.to_dict()
        (out_dir / "apo").mkdir(exist_ok=True)
        for section, prefix, entries in (
            ("sequences", "seq", self.query.sequences),
            ("multimer_sequences", "multimer", self.query.multimer_sequences),
        ):
            for index, entry in enumerate(entries):
                if not isinstance(entry, (ProteinSequence, ProteinMultimerSequence)):
                    continue
                entity = document[section][index]["protein"]
                if entry._apo_pdb is None or entry._prior_pdb is None:
                    raise ValueError(
                        f"Missing prepared apo/prior structures for {prefix}-{index}."
                    )
                apo_filename = f"apo/{prefix}-{index}-apo.pdb"
                prior_filename = f"apo/{prefix}-{index}-prior.pdb"
                write_ensemble(entry._apo_pdb, apo_filename)
                write_ensemble(entry._prior_pdb, prior_filename)
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
        self.ccd: CCD = ccd if ccd is not None else load_ccd(cache_dir=cache_dir)
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
    ) -> list[list[MonomerOutput]] | list[list[MultimerOutput]]:
        """Generate ranked AtlasFold candidates for one protein entry."""
        runner = self.apo_m_runner if isinstance(sequence, list) else self.apo_runner
        out = runner.fold(
            name, sequence, seeds=seeds, num_samples=num_samples, num_recycles=4
        )
        # Group outputs by seed and sort by ranking score.
        grouped = []
        for seed in seeds:
            group = sorted(
                (o for (sample_seed, _), o in out.outputs.items() if sample_seed == seed),
                key=lambda output: -float(output.ranking_score),
            )
            grouped.append(group)
        return grouped

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
        validate_input_sequences(query.sequences, query.multimer_sequences)
        components = []
        for sequence in query.sequences:
            if isinstance(sequence, PolymerSequence):
                components.append((sequence.ids, sequence.ccd_sequence))
            elif isinstance(sequence, LigandSequence) and sequence.ccd_ids is not None:
                components.append((sequence.ids, sequence.ccd_ids))
        for sequence in query.multimer_sequences:
            components.extend(
                (
                    ([pair[0] for pair in sequence.ids], sequence.ccd_sequence1),
                    ([pair[1] for pair in sequence.ids], sequence.ccd_sequence2),
                )
            )
        for chain_ids, codes in components:
            for residue_index, code in enumerate(codes, start=1):
                if code.upper() not in self.ccd.component_bytes:
                    raise ValueError(
                        f"Query {query.name!r}, chains {', '.join(chain_ids)}, "
                        f"residue {residue_index}: component {code!r} "
                        "is missing from CCD."
                    )

    @torch.inference_mode()
    def prepare_query(self, query: Query, seed: int, num_apos: int = 1) -> Query:
        """Load or generate structures, align them, and encode apo tokens on a copy."""
        if seed < 0 or num_apos < 1:
            raise ValueError("seed must be nonnegative and num_apos must be positive.")
        for entry in [*query.sequences, *query.multimer_sequences]:
            if not isinstance(entry, (ProteinSequence, ProteinMultimerSequence)):
                continue
            if entry.prior is not None and not entry.apo:
                raise ValueError(
                    f"Query {query.name!r}, chains {entry.ids}: "
                    "'prior' requires supplied 'apo' structures. "
                    "Leave both unset to generate structures."
                )
            if entry._apo_pdb is not None or entry._prior_pdb is not None:
                raise ValueError(
                    "prepare_query expects an unprepared query: "
                    "_apo_pdb and _prior_pdb must both be None."
                )
        self.validate_query(query)

        query = copy.deepcopy(query)
        seeds = [seed * 10 + i for i in range(1, num_apos + 1)]
        entries = [seq for seq in query.sequences if isinstance(seq, ProteinSequence)]
        entries += query.multimer_sequences
        structures = []

        # All supplied and generated structures use the same in-memory input path.
        for entry_i, entry in enumerate(entries):
            if entry.apo:
                entry._apo_pdb, apo_structures = read_pdbs(entry.apo)
                if entry.prior is not None:
                    entry._prior_pdb, prior_structures = read_pdbs(entry.prior)
                else:
                    entry._prior_pdb, prior_structures = entry._apo_pdb, apo_structures
                structures.append((apo_structures, prior_structures))
                continue
            multimer = isinstance(entry, ProteinMultimerSequence)
            sequence = [entry.sequence1, entry.sequence2] if multimer else entry.sequence
            samples_by_seed = self.run_apo_sampler(
                f"{query.name}-{entry_i}", sequence, seeds=seeds
            )
            apo_pdbs, prior_pdbs = [], []
            apo_structures, prior_structures = [], []
            for samples in samples_by_seed:
                pdbs = [
                    sample.to_pdb(model="multimer" if multimer else "monomer")
                    for sample in samples
                ]
                models = [gemmi.read_pdb_string(pdb) for pdb in pdbs]
                apo_pdbs.append(pdbs[0])
                prior_pdbs.extend(pdbs)
                apo_structures.append(models[0])
                prior_structures.extend(models)
            entry._apo_pdb = apo_pdbs
            entry._prior_pdb = prior_pdbs
            structures.append((apo_structures, prior_structures))

        chains = resolve_structure_chains(query, structures)
        tokens = None
        if self.model.prot_struct_encoder is not None:
            with (
                torch.random.fork_rng(
                    devices=[self.device] if self.device.type == "cuda" else []
                ),
                torch.autocast(
                    self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ),
            ):
                torch.manual_seed(seed)
                tokens = encode_apo_tokens(chains, self.model.prot_struct_encoder)
        align_structures(query, chains, tokens)
        return query

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
        return_trajectory: bool = False,
        return_distogram: bool = False,
    ) -> FoldingResult:
        """Generate missing apos, encode structures, and predict one query.

        ``num_apos`` controls generation only for entries without apos. Each
        generation seed contributes one ranked apo and five prior candidates.
        Supplied structures are retained, and the caller's query is unchanged.
        """
        if seed < 0 or min(num_apos, num_samples, num_recycles, num_steps) < 1:
            raise ValueError("seed must be nonnegative and inference counts positive.")
        if self.device.type != "cuda":
            raise NotImplementedError("KFold prediction requires a CUDA device.")

        # Prepare the query with any missing apo structures and tokenize them
        prepared_query = self.prepare_query(query, seed, num_apos=num_apos)

        pipeline = InputDataPipeline(self.ccd, num_prior_samples=num_samples)
        item = pipeline.build_input(prepared_query, seed)
        return self.fold_input(
            item,
            seed,
            num_samples=num_samples,
            num_recycles=num_recycles,
            num_steps=num_steps,
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
        return_trajectory: bool = False,
        return_distogram: bool = False,
    ) -> FoldingResult:
        """Run KFold on an already constructed input and return CPU results."""
        query, struct, f_input = item
        if seed < 0 or min(num_samples, num_recycles, num_steps) < 1:
            raise ValueError("seed must be nonnegative and inference counts positive.")
        if self.device.type != "cuda":
            raise NotImplementedError("KFold prediction requires a CUDA device.")
        if f_input.is_batched:
            raise ValueError("fold_input expects an unbatched FoldingInput.")
        model = self.model
        f_input = f_input.to(self.device)
        num_atoms = struct.num_atoms
        with torch.random.fork_rng(devices=[self.device]), torch.cuda.device(self.device):
            torch.manual_seed(seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.inference(
                    f_input,
                    num_recycles=num_recycles,
                    num_steps=num_steps,
                    num_samples=num_samples,
                    return_traj=return_trajectory,
                )
        summaries, scores = confidence_metrics.summarize_confidence_metrics(
            f_input, struct, out
        )
        coords = list(out["diffusion"]["coordinates"][:, :num_atoms].cpu().numpy())
        distogram = None
        if return_distogram:
            mask = f_input.token.pad_mask
            distogram = {
                "logits": out["distogram"]["logits"][mask][:, mask].half().cpu().numpy(),
                "bin_edges": out["distogram"]["bin_boundaries"].float().cpu().numpy(),
                "asym_ids": f_input.token.asym_id[mask].int().cpu().numpy(),
                "res_ids": f_input.token.residue_index[mask].int().cpu().numpy(),
            }
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
            distogram=distogram,
            trajectory=trajectory,
        )
