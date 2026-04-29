import itertools
import logging
import pathlib
from collections import defaultdict

import numpy as np
import torch

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
from kfold.data.types.structure import Chain, CovalentConnection, RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.data.utils.io.structure import read_protein_structure

from . import query


# === Helper functions ===
def parse_residue_map(residue_map: str) -> tuple[int, int, int, int]:
    """Parse residue map string into start and end indices.
    Example:
        "1:100->5:104" -> (0, 100, 4, 104)
    """
    res_range, apo_range = residue_map.split("->")
    res_st, res_end = map(int, res_range.split(":"))
    apo_st, apo_end = map(int, apo_range.split(":"))
    if (res_end - res_st) != (apo_end - apo_st):
        raise ValueError(f"Residue range length mismatch: {residue_map}")
    # Convert to 0-based indexing
    # 1:100 means residues 1 to 100 inclusive -> coords[0:100]
    return res_st - 1, res_end, apo_st - 1, apo_end


def align_sequence(query: str, target: str, margin: int = 5) -> tuple[int, int, int, int]:
    # Return default 0s if either sequence is empty
    if len(query) == 0 or len(target) == 0:
        return -1, -1, -1, -1  # No matches found

    # Read raw memory directly (no slow Python loops)
    q_arr = np.frombuffer(query.encode("ascii"), dtype=np.uint8)
    t_arr = np.frombuffer(target.encode("ascii"), dtype=np.uint8)

    # Define shift range (k = target_index - query_index)
    m, n = len(query), len(target)
    base_shift = n - m
    k_min = min(-margin, base_shift - margin)
    k_max = max(margin, base_shift + margin)
    n_shifts = k_max - k_min + 1
    n_match = min(m, n)

    # Compute alignment matches
    align = np.zeros((n_shifts, n_match), dtype=bool)
    for i, k in enumerate(range(k_min, k_max + 1)):
        q_st = max(0, -k)
        t_st = max(0, k)
        overlap_len = min(m - q_st, n - t_st)
        if overlap_len > 0:
            q_end = q_st + overlap_len
            t_end = t_st + overlap_len
            align[i, :overlap_len] = q_arr[q_st:q_end] == t_arr[t_st:t_end]

    # Find the shift index with the most matches
    best_idx = np.argmax(align.sum(axis=1))
    best_k = k_min + best_idx

    # Recalculate the base bounds for the best shift
    base_q_st = max(0, -best_k)
    base_t_st = max(0, best_k)
    overlap_len = min(m - base_q_st, n - base_t_st)

    # Extract the exact boundaries of the matched region
    match_indices = np.where(align[best_idx, :overlap_len])[0]

    if len(match_indices) == 0:
        return -1, -1, -1, -1  # No matches found

    # Get the first and last actual match indices within the evaluated window
    first_match = match_indices[0]
    last_match = match_indices[-1]

    q_st_final = base_q_st + first_match
    q_end_final = base_q_st + last_match + 1
    t_st_final = base_t_st + first_match
    t_end_final = base_t_st + last_match + 1

    return int(q_st_final), int(q_end_final), int(t_st_final), int(t_end_final)


class InputDataPipeline:
    def __init__(
        self,
        ccd: CCD,
        num_samples: int = 5,
        use_sequence_masking: bool = False,
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
        """

        self.ccd: CCD = ccd

        # Initialize apo initializer
        self.apo_initializer = apo_initialization.ApoInitializer.inference_mode(ccd)
        self.prior_sampler = prior_sampling.PriorSampler.inference_mode()
        self.num_samples = num_samples

        # Initialize tokenizer
        self.tokenizer = tokenization.Tokenizer(self.ccd)

        # Initialize sequence masking (0.0-0.15 masking ratio if enabled)
        self.use_sequence_masking = use_sequence_masking
        if use_sequence_masking:
            self.sequence_masking = sequence_masking.SequenceMasking(1.0, 0.15)

        # Initialize featurizer
        self.featurizer: featurization.InputFeaturizer = featurization.InputFeaturizer()

        self.logger = logging.getLogger("InputDataPipeline")
        self.logger.setLevel(logging.INFO)

    def __call__(
        self, input: query.Query
    ) -> tuple[RefStructure, TokenizedStructure, FoldingInput, dict]:
        """Process an Query into model-ready inputs.

        Parameters
        ----------
        input : Query
            The input file containing sequences and metadata.

        Returns
        -------
        ref_struct : RefStructure
            The reference structure.
        tokenized_struct : TokenizedStructure
            The tokenized structure.
        f_input : FoldingInput
            The featurized model input.
        struct_tok_input : dict[int, tuple[torch.Tensor, torch.Tensor]]
            A dictionary mapping entity_id to a tuple of (aatypes, coords) for
            apo structure tokenization.
        """
        return self.run(input)

    def run(
        self, input: query.Query
    ) -> tuple[RefStructure, TokenizedStructure, FoldingInput, dict]:
        """Process an Query into model-ready inputs.

        Parameters
        ----------
        input : Query
            The input file containing sequences and metadata.

        Returns
        -------
        ref_struct : RefStructure
            The reference structure.
        tokenized_struct : TokenizedStructure
            The tokenized structure.
        f_input : FoldingInput
            The featurized model input.
        struct_tok_input : dict[int, tuple[torch.Tensor, torch.Tensor]]
            A dictionary mapping entity_id to a tuple of (aatypes, coords) for
            apo structure tokenization.
        """
        rng = np.random.default_rng(input.seed)

        # Prepare structure from input file
        ref_struct: RefStructure = self.prepare_structure_from_query(input)

        # Populate apo structure
        apo_lookup = self.load_apo_structures(ref_struct, input)
        self.apo_initializer(ref_struct, lookup=apo_lookup, rng=rng)

        # Tokenize structure
        # NOTE: We feed apo structure tokens during model forward pass (gpu required).
        tokenized: TokenizedStructure = self.tokenizer(
            ref_struct, rng, num_priors=self.num_samples
        )

        # Sample prior coordinates for diffusion bridge modeling
        self.sample_prior_coords(ref_struct, tokenized, rng)

        # Apply sequence masking for sample diversity (only if enabled)
        if self.use_sequence_masking:
            self.sequence_masking(tokenized, rng)

        # Featurize input
        f_input: FoldingInput = self.featurizer(tokenized)

        # Prepare structure tokenization input for later use in model inference
        struct_tok_input = self.prepare_struct_tok_input(f_input, apo_lookup)
        return ref_struct, tokenized, f_input, struct_tok_input

    def prepare_structure_from_query(self, input: query.Query) -> RefStructure:
        """Prepare the reference structure from the input file.

        Parameters
        ----------
        input : Query
            The input query file.

        Returns
        -------
        ref_struct : RefStructure
            The reference structure.
        """
        chain_metas: list[ChainInfo] = []
        chains: list[Chain] = []
        asym_id_iter = itertools.count(1)
        chain_id_to_asym_id: dict[str, int] = {}

        # Collect bonded atoms
        chain_bonded_atoms: dict[str, dict[int, set[str]]] = defaultdict(dict)
        for (chain_id1, res_idx1, atom1), (chain_id2, res_idx2, atom2) in input.bonds:
            chain_bonded_atoms[chain_id1].setdefault(res_idx1, set()).add(atom1)
            chain_bonded_atoms[chain_id2].setdefault(res_idx2, set()).add(atom2)

        # TODO: add constraints if needed (covalent ligands)
        # This should be conducted here to property assign
        # covalent flags during ligand parsing.

        for entity_id, seq in enumerate(input.sequences, start=1):
            # Prepare chain ids
            chain_names: list[str] = seq.ids
            num_chains = len(chain_names)
            asym_ids: list[int] = [next(asym_id_iter) for _ in range(num_chains)]
            sym_ids: list[int] = [i for i in range(1, num_chains + 1)]
            for name, asym_id in zip(chain_names, asym_ids, strict=True):
                chain_id_to_asym_id[name] = asym_id

            num_residues = len(seq)

            # Parse sequence
            entity_chain: Chain = self.parse_sequence(seq, entity_id)

            # Create copies for multiple chains
            for i in range(num_chains):
                # Assign chain ids
                chain_name: str = chain_names[i]
                asym_id: int = asym_ids[i]
                sym_id: int = sym_ids[i]

                # Create chain copy
                if entity_chain.is_ligand and chain_name in chain_bonded_atoms:
                    # If it's a ligand with covalent bonds, create a new chain
                    bonded_atoms = chain_bonded_atoms[chain_name]
                    chain = self.parse_sequence(seq, entity_id, bonded_atoms).copy_with(
                        asym_id=asym_id, sym_id=sym_id
                    )
                else:
                    # Otherwise, create a copy of the original chain with new ids
                    chain = entity_chain.copy_with(
                        asym_id=asym_id, sym_id=sym_id, deepcopy=(i > 0)
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
                if entity_chain.smiles is not None:
                    chain_meta.smiles = entity_chain.smiles
                chain_metas.append(chain_meta)

        # Prepare metadata
        metadata = Metadata(
            id=input.name,
            source="query",
            chains=chain_metas,
        )

        # Add covalent bond
        connections: list[CovalentConnection] = []
        for (chain_id1, res_idx1, atom1), (chain_id2, res_idx2, atom2) in input.bonds:
            asym_id1 = chain_id_to_asym_id[chain_id1]
            asym_id2 = chain_id_to_asym_id[chain_id2]
            connections.append(
                CovalentConnection(
                    (asym_id1, asym_id2), (res_idx1, res_idx2), (atom1, atom2)
                )
            )

        # Return RefStructure
        return structure_preparation.prepare_structure(chains, connections, metadata)

    def load_apo_structures(
        self, ref_struct: RefStructure, input: query.Query
    ) -> dict[int, dict]:
        """Populate apo structure in-place.

        Parameters
        ----------
        ref_struct : RefStructure
            The reference structure.
        input : Query
            The input query file.

        Returns
        -------
        """
        entity_to_ref_chain: dict[int, Chain] = {}
        for chain in ref_struct.chains:
            eid = chain.entity_id
            if eid not in entity_to_ref_chain:
                entity_to_ref_chain[eid] = chain

        lookup: dict[int, dict] = {}
        for entity_id, seq in enumerate(input.sequences, start=1):
            if not isinstance(seq, query.ProteinSequence):
                continue
            seq_id = f"{input.name}:{list(seq.ids)}"

            path = pathlib.Path(seq.apo)
            sequence, coords = read_protein_structure(path)
            if seq.apo_range is not None:
                apo_range = seq.apo_range
            else:
                # If no residue_map is provided, check if sequence lengths match
                ref_chain = entity_to_ref_chain[entity_id]
                length = len(sequence)
                ref_seq = ref_chain.get_sequence(map_to_standard=True)
                seq_st, seq_end, apo_st, apo_end = align_sequence(ref_seq, sequence)
                if seq_st == -1:
                    self.logger.warning(
                        f"No apo_range provided for protein sequence {seq_id}, "
                        f"and no alignment found between reference and apo sequences. "
                        f"Skipping apo structure loading for this entity."
                    )
                    continue
                apo_range = f"{seq_st + 1}:{seq_end}->{apo_st + 1}:{apo_end}"
                if apo_range != f"1:{length}->1:{length}":
                    self.logger.warning(
                        f"No apo_range provided for protein sequence {seq_id}, "
                        f"but lengths do not match. Inferred apo_range: {apo_range}"
                    )

            lookup[entity_id] = {
                "path": path,
                "seq": sequence,
                "coords": coords,
                "residue_map": apo_range,
            }
        return lookup

    def sample_prior_coords(
        self,
        ref_struct: RefStructure,
        tokenized: TokenizedStructure,
        rng: np.random.Generator,
    ) -> None:
        """Populate the prior coordinates for the given reference structure."""
        prior_coords = np.full(
            (tokenized.num_tokens, 24, self.num_samples, 3), np.nan, dtype=np.float32
        )
        prior_coords[tokenized.atom.pad_mask] = self.prior_sampler(
            ref_struct, self.num_samples, rng=rng
        ).transpose(1, 0, 2)
        tokenized.atom.prior_coords[:] = prior_coords

    def prepare_struct_tok_input(
        self,
        f_input: FoldingInput,
        apo_lookup: dict[int, dict],
    ) -> dict[int, dict]:
        """Prepare the structure tokenization input for apo structures.

        Parameters
        ----------
        f_input : FoldingInput
            The featurized model input containing sequence and entity information.
        apo_lookup : dict[int, dict]
            The lookup dictionary containing apo structure information.

        Returns
        dict[int, dict]
            A dictionary mapping entity_id to a tuple of (aatypes, coords) for
            structure tokenization.
        """
        struct_tok_input: dict[int, dict] = {}
        for entity_id, info in apo_lookup.items():
            length = len(info["seq"])
            aatypes = C.sequence.encode_protein_sequence(info["seq"])
            aatypes = torch.tensor(aatypes, dtype=torch.long)
            coords = torch.from_numpy(info["coords"]).float()

            # Find the corresponding indices in the featurized input.
            indices = torch.where(f_input.sequence.entity_id == entity_id)[0]
            ref_length = len(indices)
            if ref_length == 0:
                raise ValueError(f"No sequence indices found for entity_id {entity_id}.")

            seq_base = indices[0].item() + 1  # Account for bos token at the start
            if "residue_map" in info:
                # If residue_map is provided, find the corresponding residue ranges
                res_st, res_end, apo_st, apo_end = parse_residue_map(info["residue_map"])
                seq_st, seq_end = seq_base + res_st, seq_base + res_end
            else:
                apo_st, apo_end = 0, length
                seq_st, seq_end = seq_base, seq_base + length

            if (seq_end - seq_st) != (apo_end - apo_st):
                raise ValueError(
                    f"Sequence range does not match apo range for entity {entity_id}: "
                    f"seq range ({seq_st}:{seq_end}) vs apo range ({apo_st}:{apo_end})"
                )

            struct_tok_input[entity_id] = {
                "aatypes": aatypes,
                "coords": coords,
                "mapping": (seq_st, seq_end, apo_st, apo_end),
            }
        return struct_tok_input

    # ================================================================================
    # Chain Parsing Functions
    # ================================================================================
    def parse_sequence(
        self,
        seq: query.BaseSequence,
        entity_id: int,
        bonded_atoms: dict[int, set[str]] | None = None,
    ) -> Chain:
        """Parse a chain from the sequence input.

        Parameters
        ----------
        seq : Sequence
            The sequence input.
        entity_id : int
            The entity_id to assign to the chain.
        bonded_atoms : dict[int, set[str]], optional
            A dictionary mapping residue index to a set of atom names that are involved
            in covalent bonds.

        Returns
        -------
        chain: Chain
            The reference chain.

        Notes
        -----
        The chain ids (entity_id, asym_id, sym_id) are all set to placeholder (zero)
        """
        if isinstance(seq, query.PolymerSequence):
            # Load sequence and modifications
            sequence: str = seq.sequence
            modifications: dict[int, str] = {
                int(k): v for k, v in seq.modifications.items()
            }

            # Get CCD sequences (three-letter codes) from one-letter sequence
            ccd_sequences: list[str] = [
                C.residue.map_one_letter_to_residue_name(aa, seq.ctype).name
                for aa in sequence
            ]
            # Apply modifications
            for res_idx, ccd_code in modifications.items():
                ccd_sequences[res_idx - 1] = ccd_code  # res_idx is 1-based

            # Prepare reference chain
            return structure_preparation.prepare_ref_chain(
                chain_type=seq.ctype,
                entity_id=entity_id,
                ccd_sequences=ccd_sequences,
                ccd=self.ccd,
                bonded_atoms=bonded_atoms,
            )
        elif isinstance(seq, query.LigandSequence):
            # Load ccd or smiles
            ctype = C.ChainType.LIGAND
            if seq.ccd_ids is not None:
                return structure_preparation.prepare_ref_chain(
                    chain_type=ctype,
                    ccd_sequences=seq.ccd_ids,
                    ccd=self.ccd,
                    bonded_atoms=bonded_atoms,
                )
            else:
                assert seq.smiles is not None, (
                    "Either CCD code or SMILES must be provided."
                )
                # NOTE: Using "LIG" as a placeholder code for ligands from SMILES
                # This will be replaced later during mmcif writing.
                code = f"LIG{entity_id}"
                return structure_preparation.prepare_ref_chain(
                    chain_type=ctype,
                    entity_id=entity_id,
                    ccd_sequences=[code],
                    smiles=seq.smiles,
                    ccd=self.ccd,
                    bonded_atoms=bonded_atoms,
                )
        else:
            raise ValueError(f"Unsupported sequence type: {type(seq)}")
