import itertools
import logging
import pathlib

import numpy as np

import kfold.constants as C
from kfold.data.pipelines import (
    apo_initialization,
    featurization,
    prior_sampling,
    sequence_masking,
    structure_preparation,
    tokenization,
)
from kfold.data.types.ccd import CCD
from kfold.data.types.metadata import ChainInfo, Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import Chain, RefStructure
from kfold.data.types.tokenized import TokenizedStructure

from . import query

logger = logging.getLogger(__name__)


class InputDataPipeline:
    def __init__(
        self,
        ccd: CCD,
        num_samples: int = 5,
        use_sequence_masking: bool = False,
        seed: int = 1,
    ) -> None:
        """Initialize the input data pipeline.

        Parameters
        ----------
        ccd : CCD
            The chemical component dictionary for residue information.
        num_samples : int, optional
            The number of samples to generate for prior sampling.
            Default is 5.
        use_sequence_masking : bool, optional
            Whether to apply sequence masking for sample diversity. Default is False.
        seed : int, optional
            The random seed for reproducibility. Default is 1.
        """

        self.ccd: CCD = ccd
        self.seed: int = seed

        # Initialize apo initializer
        self.apo_initializer = apo_initialization.ApoInitializer.inference_mode(ccd)
        self.prior_sampler = prior_sampling.PriorSampler.inference_mode(ccd, num_samples)

        # Initialize tokenizer
        self.tokenizer = tokenization.Tokenizer(self.ccd, self.prior_sampler)

        # Initialize sequence masking (0.0-0.15 masking ratio if enabled)
        mask_prob = 1.0 if use_sequence_masking else 0.0
        self.sequence_masking = sequence_masking.SequenceMasking(mask_prob)

        # Initialize featurizer
        self.featurizer: featurization.InputFeaturizer = featurization.InputFeaturizer()

    def process_query(
        self, input: query.Query
    ) -> tuple[RefStructure, TokenizedStructure, FoldingInput]:
        """Process an Query into model-ready inputs.

        Parameters
        ----------
        input : Query
            The input file containing sequences and metadata.

        Returns
        -------
        ref_struct : RefStructure
            The reference structure representation.
        tokenized_struct : TokenizedStructure
            The tokenized structure representation.
        f_input : FoldingInput
            The featurized model input.
        """
        rng = np.random.default_rng(self.seed)

        # Prepare structure from input file
        ref_struct = self.prepare_structure_from_query(input)

        # Populate apo structure
        self.populate_apo_structure(ref_struct, input, rng=rng)

        # Tokenize structure
        tok_struct = self.tokenizer.tokenize(ref_struct)

        # Apply sequence masking for sample diversity (only if enabled)
        self.sequence_masking(tok_struct, rng)

        # Featurize input
        f_input = self.featurizer(tok_struct, rng)
        return ref_struct, tok_struct, f_input

    def prepare_structure_from_query(self, input: query.Query) -> RefStructure:
        """Prepare the reference structure from the input file.

        Parameters
        ----------
        input : Query
            The input query file.

        Returns
        -------
        ref_struct : RefStructure
            The reference structure representation.
        """
        chain_metas: list[ChainInfo] = []
        chains: list[Chain] = []
        asym_id_iter = itertools.count(1)

        # TODO: add constraints if needed (covalent ligands)
        # This should be conducted here to property assign
        # covalent flags during ligand parsing.

        for entity_id, seq in enumerate(input.sequences, start=1):
            # Prepare chain ids
            chain_names: list[str] = seq.ids
            num_chains = len(chain_names)
            asym_ids: list[int] = [next(asym_id_iter) for _ in range(num_chains)]
            sym_ids: list[int] = [i for i in range(1, num_chains + 1)]

            num_residues = len(seq)

            # Parse sequence
            entity_chain: Chain
            if isinstance(seq, query.LigandSequence):
                entity_chain = self.parse_ligand_sequence(seq)
            else:
                entity_chain = self.parse_polymer_sequence(seq)

            # Create copies for multiple chains
            for i in range(num_chains):
                # Assign chain ids
                chain_name: str = chain_names[i]
                asym_id: int = asym_ids[i]
                sym_id: int = sym_ids[i]

                # Create chain copy
                chain = entity_chain.copy_with(
                    entity_id=entity_id,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    deepcopy=(i > 0),  # deep copy only for additional chains
                )
                chains.append(chain)

                # Add chain metadata
                chain_meta = ChainInfo(
                    type=entity_chain.ctype,
                    name=chain_name,
                    entity_id=entity_id,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    num_residues=num_residues,
                    num_tokens=chain.num_tokens,
                    num_atoms=chain.num_atoms,
                    description=seq.description,
                )
                chain_metas.append(chain_meta)

        # Prepare metadata
        metadata = Metadata(
            id=input.name,
            source="query",
            chains=chain_metas,
        )

        # Return RefStructure
        return structure_preparation.prepare_structure(
            chains=chains,
            connections=[],  # No connections for now
            metadata=metadata,
        )

    def populate_apo_structure(
        self,
        ref_struct: RefStructure,
        input: query.Query,
        rng: np.random.Generator,
    ) -> None:
        """Populate apo structure in-place.

        Parameters
        ----------
        ref_struct : RefStructure
            The reference structure to populate.
        input : Query
            The input query file.
        rng : np.random.Generator | None, optional
            Random number generator for any stochastic processes. Default is None.
        """
        # Prepare lookup (apo initializer input)
        lookup: dict[int, dict] = {}
        for entity_id, seq in enumerate(input.sequences, start=1):
            if isinstance(seq, query.LigandSequence):
                # Ligands do not have apo structures
                continue
            if seq.apo is None:
                # No apo structure provided
                continue
            # Use provided apo structure
            apo_path = pathlib.Path(seq.apo)
            lookup[entity_id] = {
                "path": apo_path,
                "source": "query",  # dummy
            }
        # Populate apo structure
        self.apo_initializer(ref_struct, lookup=lookup, rng=rng)

    # ================================================================================
    # Chain Parsing Functions
    # ================================================================================

    def parse_polymer_sequence(
        self,
        seq: query.PolymerSequence,
    ) -> Chain:
        """Parse a polymer chain from the sequence input.

        Parameters
        ----------
        seq : PolymerSequence
            The polymer sequence input.

        Returns
        -------
        chain: Chain
            The chain representation.

        Notes
        -----
        The chain ids (entity_id, asym_id, sym_id) are all set to placeholder (zero)
        """
        # Load sequence and modifications
        sequence: str = seq.sequence
        modifications: dict[int, str] = {int(k): v for k, v in seq.modifications.items()}

        # Get CCD sequences (three-letter codes) from one-letter sequence
        ccd_sequences: list[str] = [
            C.residue.map_one_letter_to_residue_name(aa, seq.ctype).name
            for aa in sequence
        ]
        # Apply modifications
        for res_idx, ccd_code in modifications.items():
            ccd_sequences[res_idx - 1] = ccd_code  # res_idx is 1-based

        # Prepare reference chain
        chain = structure_preparation.prepare_ref_chain(
            chain_type=seq.ctype,
            ccd_sequences=ccd_sequences,
            ccd=self.ccd,
        )
        return chain

    def parse_ligand_sequence(
        self,
        seq: query.LigandSequence,
        is_covalent: bool = False,
    ) -> Chain:
        """Parse a ligand chain from the sequence input.

        Parameters
        ----------
        seq : LigandSequence
            The ligand sequence input.
        is_covalent : bool, optional
            Whether the ligand is covalently bound. Default is False.

        Returns
        -------
        chain: Chain
            The chain representation.

        Notes
        -----
        The chain ids (entity_id, asym_id, sym_id) are all set to placeholder (zero)
        """
        # Load ccd or smiles
        if seq.ccd_ids is not None:
            ctype = C.ChainType.LIGAND
            return structure_preparation.prepare_ref_chain(
                chain_type=ctype,
                ccd_sequences=seq.ccd_ids,
                ccd=self.ccd,
            )
        else:
            assert seq.smiles is not None, "Either CCD code or SMILES must be provided."
            # NOTE: Using "LIG" as a placeholder code for ligands from SMILES
            # This will be replaced later during mmcif writing.
            raise NotImplementedError(
                "Parsing ligands from SMILES is not implemented yet."
            )
