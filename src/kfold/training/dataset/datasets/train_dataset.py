""" """

import dataclasses
from pathlib import Path

import lmdb
import numpy as np
import torch

from kfold.data.pipelines import featurization, prior_sampling, tokenization
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.training.dataset.cropper import BaseCropper
from kfold.training.dataset.sampler import BaseSampler, Sample
from kfold.training.dataset.utils import constraint_sampling, pre_crop
from kfold.training.utils.permutation_alignment.symmetry import get_symmetries
from kfold.utils.registry import Registry

from .base import BaseLMDBDataset, DatasetConfig

StructInfo = dict


@dataclasses.dataclass(kw_only=True)
class TrainingDatasetConfig(DatasetConfig):
    """Configuration for training dataset.

    Attributes
    ----------
    type: str
        Type of the training dataset, e.g. "monomer-distillation".
    weight : float
        Weight of the dataset during training.
    prob_drop_apo : float
        Probability of dropping apo structure for each chain.
    prob_drop_struct_token : float
        Probability of dropping structure tokens for each chain.
    sampler : BaseSampler.Config | None
        Sampler configuration for generating samples.
    cropper : BaseCropper.Config | None
        Cropper configuration for cropping structures.
    """

    type: str
    weight: float = 1.0
    prob_drop_apo: float = 0.0  # probability of dropping apo structure
    prob_drop_struct_token: float = 0.0  # probability of dropping structure tokens
    sampler: BaseSampler.Config | None
    cropper: BaseCropper.Config | None


# === Helper functions === #
def _open_lmdb(lmdb_path: str | Path) -> lmdb.Environment:
    if not Path(lmdb_path).exists():
        raise FileNotFoundError(f"LMDB file {lmdb_path} not found.")
    return lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False
    )


def parse_residue_map(residue_map: str) -> tuple[int, int, int, int]:
    """Parse residue map string into start and end indices.
    Example: "1:100->5:104" -> (0, 100, 4, 104)
    """
    res_range, apo_range = residue_map.split("->")
    res_st, res_end = map(int, res_range.split(":"))
    apo_st, apo_end = map(int, apo_range.split(":"))
    if (res_end - res_st) != (apo_end - apo_st):
        return -1, -1, -1, -1  # invalid mapping
    # Convert to 0-based indexing
    # 1:100 means residues 1 to 100 inclusive -> coords[0:100]
    return res_st - 1, res_end, apo_st - 1, apo_end


# === Training Dataset === #
class TrainingDataset(BaseLMDBDataset):
    """Training dataset with AF3-style sampling and cropping."""

    def __init__(
        self,
        config: TrainingDatasetConfig,
        ccd: CCD,
        tokenizer: tokenization.Tokenizer,
        featurizer: featurization.InputFeaturizer,
        prior_sampler: prior_sampling.PriorSampler | None,
        safe_load: bool,
        max_chains: int,
        max_tokens: int,
        max_sequence_tokens: int,
    ) -> None:
        """
        Parameters
        ----------
        config : TrainingDatasetConfig
            Dataset configuration.
        ccd: CCD
            CCD database
        tokenizer: tokenization.Tokenizer
            Tokenizer for tokenizing structures.
        featurizer: featurization.InputFeaturizer
            Featurizer for featurizing tokenized structures.
        prior_sampler: prior_sampling.PriorSampler | None
            Prior sampler for sampling prior coordinates (optional).
        max_chains : int
            Maximum number of chains per sample.
        max_tokens : int
            Maximum number of tokens per sample.
        max_sequence_tokens : int
            Maximum number of sequence tokens per sample,
            limiting the entire input size of PLM module.
        Notes
        -----
        This dataset implements AF3-style sampling (chain/interface-based).
        1. Samples are generated based on chains/interfaces in the structures.
        2. During data loading, samples are cropped to fit within `max_tokens`
           using the provided `cropper`.
        """
        super().__init__(
            config,
            ccd,
            tokenizer,
            featurizer,
            prior_sampler,
            safe_load=safe_load,
            train=True,
        )
        self.config: TrainingDatasetConfig = config
        if self.seed is not None:
            # Warn about fixed seed affecting randomness
            self.logger.warning(
                "Seed is set for TrainingDataset, which may affect randomness."
            )

        # For pre-cropping (RefStructure)
        self.max_chains: int = max_chains
        # For main cropping (TokenizedStructure)
        self.max_tokens: int = max_tokens
        self.max_sequence_tokens: int = max_sequence_tokens

        assert max_sequence_tokens >= max_tokens + (max_chains * 2), (
            f"max_sequence_tokens should be greater than max_tokens to accommodate "
            f"additional sequence tokens for PLM input."
            f" (max_sequence_tokens={max_sequence_tokens}, max_tokens={max_tokens}, "
            f"max_chains={max_chains})"
        )  # +2 tokens per chain for [CLS] and [SEP]

        assert config.cropper is not None, "Cropper config must be provided."
        self.cropper: BaseCropper = Registry.instantiate(config.cropper)

        if config.sampler is None:
            # Uniform sampling (complex-level)
            self.sampler: BaseSampler = BaseSampler()
        else:
            # AF3-style sampling (chain/interface-based)
            self.sampler: BaseSampler = Registry.instantiate(config.sampler)

        samples, weights = self.sampler.get_samples(self.metadatas)
        self.samples: list[Sample] = samples
        self.weights: np.ndarray = weights

        # Constraint sampling for training
        # TODO: configurize the parameters
        self.max_constraints = 0
        self.constraint_sampling = constraint_sampling.ConstraintSampling(
            min_dist=2.0,
            max_dist=22.0,
            prob_constraint=0.05,
            max_constraints=self.max_constraints,
        )

    def sanity_check(self) -> None:
        """Perform sanity checks on the dataset."""
        cfg = self.config
        if cfg.apo_init is None:
            self.logger.warning("Apo initialization is disabled for training set.")
            return
        if cfg.apo_init.perturbation is None:
            self.logger.warning("Protein perturbation is disabled for training set.")

    def __len__(self) -> int:
        return len(self.samples)

    def get_item_safe(
        self, index: int, num_trials: int = 100
    ) -> tuple[FoldingInput, StructInfo]:
        """Get the folding input for the given index, with retry on failure.
        NOTE: This is overridden to use `self.samples` instead of `self.metadatas`.
        """
        trials = []
        for _ in range(num_trials):
            sample = self.samples[index]
            metadata = Metadata.from_dict(sample.metadata)
            try:
                return self.get_item(metadata, asym_ids=sample.asym_id)
            except (KeyboardInterrupt, SystemExit) as e:
                raise e
            except Exception as e:
                sample_id = sample.metadata["id"]
                self.logger.error(
                    f"Error loading index {sample_id}({index}): {e}. Retrying..."
                )
                if not self.safe_load:
                    raise e
                index = np.random.randint(0, len(self))
                trials.append(sample)
        raise RuntimeError(
            f"Failed to load data after {num_trials} attempts. Tried: {trials}"
        )

    def get_item(self, metadata: Metadata, **kwargs) -> tuple[FoldingInput, StructInfo]:
        """Get the folding input for the given sample."""
        metadata_id: str = metadata.id

        # Initialize random number generator (create new rng based on metadata_id)
        rng = np.random.default_rng()

        # Load structure (NOTE: ref_struct.metadata == metadata)
        ref_struct: RefStructure = self.load_ref_structure(metadata)

        # Sub-complex structure extraction for large complex (>20 chains)
        # This is the on-the-fly pipeline of AlphaFold3 SI Section 2.5.4
        ref_struct = self.extract_substructure(ref_struct, rng=rng, **kwargs)
        metadata = ref_struct.metadata  # update metadata after extraction

        # Get apo lookup for the structure
        apo_lookup = self.get_apo_lookup(ref_struct, rng)

        # Fetch apo structure
        apo_dict = self.fetch_apo_structures(ref_struct, apo_lookup, rng)

        prior_coords = self.sample_prior_coords(ref_struct, apo_dict, rng)

        # Tokenization
        tokenized = self.tokenize(ref_struct, apo_dict, prior_coords, rng)

        # Populate structure tokens for apo structure (in-place)
        self.populate_structure_tokens(tokenized, apo_lookup)

        # Optionally drop apo structure for trunk input during training
        self.drop_apo_structure(tokenized, rng)

        # Cropping
        cropped, crop_mode = self.crop_structure(tokenized, metadata, rng=rng, **kwargs)

        # Featurization
        f_input = self.featurize(cropped)
        f_input = f_input.copy_with(crop_mode=torch.tensor(crop_mode, dtype=torch.long))

        # Pad the features.
        f_input = self.pad_input(f_input)

        # Determine whether to train confidence head
        train_confidence = self.determine_confidence_train_data(metadata)

        struct_info = {
            "id": metadata_id,
            "structure": ref_struct,
            "symmetry": get_symmetries(ref_struct, self.ccd),
            "train_confidence_head": train_confidence,
        }
        return f_input, struct_info

    def pad_input(self, f_input: FoldingInput) -> FoldingInput:
        max_chains = self.max_chains
        max_tokens = self.max_tokens
        max_sequence_tokens = self.max_sequence_tokens
        num_constraints = self.max_constraints
        max_atoms = max_tokens * 24  # max 24 atoms per token
        max_bonds = max_tokens * 10  # max 10 bonds per token
        return f_input.pad(
            max_tokens=max_tokens,
            max_chains=max_chains,
            max_atoms=max_atoms,
            max_bonds=max_bonds,
            max_sequence_tokens=max_sequence_tokens,
            max_constraints=num_constraints,
        )

    def tokenize(
        self,
        ref_struct: RefStructure,
        apo_dict: dict[int, np.ndarray],
        prior_coords: np.ndarray,
        rng: np.random.Generator,
    ) -> TokenizedStructure:
        """Tokenize the given structure."""
        # Sample the constraints
        constraints = self.constraint_sampling(ref_struct, rng)
        # Tokenize the structure
        tokenized = self.tokenizer(
            ref_struct,
            rng,
            apo_coords=apo_dict,
            prior_coords=prior_coords,
            constraints=constraints,
        )
        return tokenized

    # === Utility methods for training dataset === #
    def extract_substructure(
        self,
        ref_struct: RefStructure,
        rng: np.random.Generator,
        **kwargs,
    ) -> RefStructure:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: int | tuple[int, int] | None = kwargs["asym_ids"]
        if self.max_chains < ref_struct.num_chains:
            # Get sub-complex with limited number of chains
            ref_struct = pre_crop.extract_substructure(
                ref_struct,
                max_chains=self.max_chains,
                bias_asym_id=asym_ids,
                rng=rng,
            )
        return ref_struct

    def crop_structure(
        self,
        tokenized: TokenizedStructure,
        metadata: Metadata,
        rng: np.random.Generator,
        **kwargs,
    ) -> tuple[TokenizedStructure, int]:
        assert "asym_ids" in kwargs, "asym_ids must be provided for cropping."
        asym_ids: int | tuple[int, int] | None = kwargs["asym_ids"]
        crop_mode = 0
        if self.max_tokens < tokenized.num_tokens:
            # Crop the tokenized structure
            tokenized = self.cropper.crop(
                tokenized,
                metadata,
                max_tokens=self.max_tokens,
                max_sequence_tokens=self.max_sequence_tokens,
                bias_asym_id=asym_ids,
                rng=rng,
            )
            crop_mode = int(getattr(self.cropper, "last_crop_mode", 0))
        return tokenized, crop_mode

    def drop_apo_structure(
        self, tokenized: TokenizedStructure, rng: np.random.Generator
    ) -> None:
        """Drop the apo coords / structure tokens"""
        # Optionally drop apo structure for trunk input during training
        if rng.random() < self.config.prob_drop_apo:
            tokenized.token.apo_center_coords.fill(np.nan)
            tokenized.token.apo_repr_coords.fill(np.nan)
            tokenized.token.apo_frame_coords.fill(np.nan)
            tokenized.token.apo_center_mask.fill(False)
            tokenized.token.apo_repr_mask.fill(False)
            tokenized.token.apo_frame_mask.fill(False)
            tokenized.atom.apo_coords.fill(np.nan)
            tokenized.atom.apo_mask.fill(False)

        if rng.random() < self.config.prob_drop_struct_token:
            tokenized.sequence.bb_struct_token_id.fill(-1)
            tokenized.sequence.fa_struct_token_id.fill(-1)

    def determine_confidence_train_data(self, metadata: Metadata) -> bool:
        return False
