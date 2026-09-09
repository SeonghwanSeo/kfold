"""CPU prediction results for Python callers and file output."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kfold.data.types.structure import RefStructure
from kfold.data.utils.writer import KFoldWriter
from kfold.inference.query import Query


@dataclass
class FoldingResult:
    query: Query
    structure: RefStructure
    coordinates: np.ndarray
    confidence_summary: list[dict]
    confidence_scores: list[dict]
    distogram: dict[str, np.ndarray] | None = None
    trajectory: np.ndarray | None = None

    def save(self, out_dir: str | Path, *, save_confidence: bool = False) -> Path:
        """Write samples and mark complete only after all outputs are saved."""
        name, seed = self.query.name, self.query.seed
        directory = Path(out_dir) / name / f"{name}_seed-{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        done = directory / "done.txt"
        done.unlink(missing_ok=True)
        self.query.save(directory.parent / f"{name}_seed-{seed}_query.yaml")
        for i, coords in enumerate(self.coordinates):
            prefix = directory / f"{name}_seed-{seed}_sample-{i}"
            score = self.confidence_scores[i]
            structure = self.structure.copy_with_new_coords(
                coords, b_factors=score["plddt"]
            )
            # Use the format-specific writer so write errors propagate to callers.
            KFoldWriter.write_mmcif(structure, Path(f"{prefix}.cif"))
            Path(f"{prefix}_confidences.json").write_text(
                json.dumps(self.confidence_summary[i], indent=2) + "\n"
            )
            if save_confidence:
                np.savez_compressed(
                    f"{prefix}_confidences.npz",
                    **{key: score[key] for key in ("plddt", "pae", "pde")},
                )
            if self.trajectory is not None:
                KFoldWriter.write_trajectory(
                    self.structure, self.trajectory[i], Path(f"{prefix}_traj.pdb")
                )
        if self.distogram is not None:
            np.savez_compressed(
                directory / f"{name}_seed-{seed}_distogram.npz", **self.distogram
            )
        done.touch()
        return directory
