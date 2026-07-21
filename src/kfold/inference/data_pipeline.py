import itertools
import logging
import pathlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import kfold.constants as C
from kfold.data.pipelines import (
    featurization,
    prior_sampling,
    structure_preparation,
    tokenization,
)
from kfold.data.types.ccd import CCD, Component
from kfold.data.types.constraint import Constraint
from kfold.data.types.metadata import ChainInfo, Metadata
from kfold.data.types.model_input import FoldingInput
from kfold.data.types.structure import Chain, CovalentConnection, RefStructure
from kfold.data.types.tokenized import TokenizedStructure
from kfold.data.utils.io.structure import (
    read_protein_multimer_structure,
    read_protein_structure,
)

from . import query


@dataclass
class ResolvedStructureSources:
    """Normalized apo/prior inputs for one inference query."""

    num_apo: int
    apo_coords: dict[int, np.ndarray]
    prior_sources: list[dict[int, np.ndarray]]
    struct_token_records: list[list[dict]]


@dataclass
class _AlignedStructureSource:
    """One parsed structure aligned to its query sequence."""

    sequence: str
    coords: np.ndarray
    target_range: tuple[int, int]


def _best_sequence_mapping(
    target_sequence: str,
    source_sequence: str,
) -> tuple[tuple[int, int, int, int], int]:
    """Return the ungapped overlap with the most matching residues."""
    target_length = len(target_sequence)
    source_length = len(source_sequence)
    best_mapping = (0, 0, 0, 0)
    best_score = (-1, -1)

    for offset in range(1 - target_length, source_length):
        target_start = max(0, -offset)
        source_start = max(0, offset)
        overlap = min(
            target_length - target_start,
            source_length - source_start,
        )
        num_matches = sum(
            target_sequence[target_start + i] == source_sequence[source_start + i]
            for i in range(overlap)
        )
        score = (num_matches, overlap)
        if score > best_score:
            best_score = score
            best_mapping = (
                target_start,
                target_start + overlap,
                source_start,
                source_start + overlap,
            )

    return best_mapping, best_score[0]


class InputDataPipeline:
    def __init__(self, ccd: CCD, num_prior_samples: int = 5) -> None:
        """Initialize the input data pipeline.

        Parameters
        ----------
        ccd : CCD
            The chemical component dictionary for residue information.
        num_prior_samples : int, optional
            Number of heuristic DNA/ligand priors to create. Default is 5.
        """

        self.ccd: CCD = ccd

        self.prior_sampler = prior_sampling.PriorSampler.inference_mode()
        if num_prior_samples <= 0:
            raise ValueError("num_prior_samples must be positive.")
        self.num_prior_samples = num_prior_samples

        # Initialize tokenizer
        self.tokenizer = tokenization.Tokenizer(self.ccd)

        # Initialize featurizer
        self.featurizer: featurization.InputFeaturizer = featurization.InputFeaturizer()

        self.logger = logging.getLogger("InputDataPipeline")
        self.logger.setLevel(logging.INFO)

    def __call__(
        self, input: query.Query
    ) -> tuple[RefStructure, TokenizedStructure, FoldingInput, list[list[dict]]]:
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
        struct_token_records : list[list[dict]]
            The raw apo-token records for each chain.
        """
        return self.run(input)

    def run(
        self, input: query.Query
    ) -> tuple[RefStructure, TokenizedStructure, FoldingInput, list[list[dict]]]:
        """Convert one query into model input and raw apo-token records."""
        # Set up RNGs
        source_rng = np.random.default_rng(np.random.SeedSequence([input.seed, 0]))
        prior_rng = np.random.default_rng(np.random.SeedSequence([input.seed, 1]))
        tokenizer_rng = np.random.default_rng(np.random.SeedSequence([input.seed, 2]))

        # Read query and prepare reference structure
        ref_struct, constraints = self.read_query(input)

        # Read apo/prior structures
        sources = self.resolve_structure_sources(ref_struct, input, source_rng)

        # Each resolved source contributes exactly one prior. Perturbation,
        # relaxation, and rigid augmentation stay inside PriorSampler.
        prior_coords = np.stack(
            [
                self.prior_sampler.sample(ref_struct, source, 1, prior_rng)[0]
                for source in sources.prior_sources
            ],
            axis=0,
        )

        # Tokenize structure
        # Learned structure token IDs are added later on the model device.
        metadata_by_asym_id = {c.asym_id: c for c in ref_struct.metadata.chains}
        apo_uids = {
            asym_id: np.full(
                sources.num_apo,
                metadata_by_asym_id[asym_id].apo_uid,
                dtype=np.int64,
            )
            for asym_id in sources.apo_coords
        }
        tokenized = self.tokenizer(
            ref_struct,
            tokenizer_rng,
            apo_coords=sources.apo_coords,
            apo_uids=apo_uids,
            num_apo=sources.num_apo,
            prior_coords=prior_coords,
            constraints=constraints,
        )

        f_input = self.featurizer(tokenized)
        return ref_struct, tokenized, f_input, sources.struct_token_records

    def read_query(self, input: query.Query) -> tuple[RefStructure, list[Constraint]]:
        """Prepare the reference structure from the input file.

        Parameters
        ----------
        input : Query
            The input query file.

        Returns
        -------
        ref_struct : RefStructure
            The reference structure.
        constraints : list[Constraint]
            The list of distance constraints specified in the input.
        """
        chain_metas: list[ChainInfo] = []
        chains: list[Chain] = []
        entity_id_iter = itertools.count(1)
        asym_id_iter = itertools.count(1)
        chain_id_to_asym_id: dict[str, int] = {}

        # Collect bonded atoms
        chain_bonded_atoms: dict[str, dict[int, set[str]]] = defaultdict(dict)
        for constraint in input.constraints:
            if constraint.type == "bond":
                chain_id1, res_idx1, atom1 = constraint.atom1
                chain_id2, res_idx2, atom2 = constraint.atom2
                chain_bonded_atoms[chain_id1].setdefault(res_idx1, set()).add(atom1)
                chain_bonded_atoms[chain_id2].setdefault(res_idx2, set()).add(atom2)

        for seq in input.sequences:
            entity_id = next(entity_id_iter)
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

        # A multimer sequence describes two entities and one or more physical
        # copies of their shared-frame pair.
        for sequence_group in input.multimer_sequences:
            entity_id1 = next(entity_id_iter)
            entity_id2 = next(entity_id_iter)
            entity_chain1 = structure_preparation.prepare_ref_chain(
                chain_type=sequence_group.ctype,
                ccd_sequences=sequence_group.ccd_sequence1,
                ccd=self.ccd,
                entity_id=entity_id1,
            )
            entity_chain2 = structure_preparation.prepare_ref_chain(
                sequence_group.ctype,
                sequence_group.ccd_sequence2,
                ccd=self.ccd,
                entity_id=entity_id2,
            )

            for sym_id, (chain_name1, chain_name2) in enumerate(
                sequence_group.ids, start=1
            ):
                asym_id1 = next(asym_id_iter)
                asym_id2 = next(asym_id_iter)
                rigid_group_uid = asym_id1
                chain_id_to_asym_id[chain_name1] = asym_id1
                chain_id_to_asym_id[chain_name2] = asym_id2

                chain1 = entity_chain1.copy_with(
                    asym_id=asym_id1, sym_id=sym_id, deepcopy=(sym_id > 1)
                )
                chain2 = entity_chain2.copy_with(
                    asym_id=asym_id2, sym_id=sym_id, deepcopy=(sym_id > 1)
                )
                chains.extend((chain1, chain2))

                chain_metas.extend(
                    (
                        ChainInfo(
                            type=chain1.ctype,
                            name=chain_name1,
                            entity_id=entity_id1,
                            asym_id=asym_id1,
                            sym_id=sym_id,
                            num_residues=chain1.num_residues,
                            num_tokens=chain1.num_tokens,
                            num_atoms=chain1.num_atoms,
                            description=sequence_group.description,
                            apo_uid=rigid_group_uid,
                            prior_uid=rigid_group_uid,
                        ),
                        ChainInfo(
                            type=chain2.ctype,
                            name=chain_name2,
                            entity_id=entity_id2,
                            asym_id=asym_id2,
                            sym_id=sym_id,
                            num_residues=chain2.num_residues,
                            num_tokens=chain2.num_tokens,
                            num_atoms=chain2.num_atoms,
                            description=sequence_group.description,
                            apo_uid=rigid_group_uid,
                            prior_uid=rigid_group_uid,
                        ),
                    )
                )

        # Prepare metadata
        metadata = Metadata(id=input.name, source="query", chains=chain_metas)

        # Add covalent bond and constraint
        connections: list[CovalentConnection] = []
        constraints: list[Constraint] = []
        for constraint in input.constraints:
            if constraint.type == "bond":
                chain_id1, res_idx1, atom1 = constraint.atom1
                chain_id2, res_idx2, atom2 = constraint.atom2
                asym_id1 = chain_id_to_asym_id[chain_id1]
                asym_id2 = chain_id_to_asym_id[chain_id2]
                connections.append(
                    CovalentConnection(
                        (asym_id1, asym_id2), (res_idx1, res_idx2), (atom1, atom2)
                    )
                )
            elif constraint.type == "distance":
                raise NotImplementedError("Distance constraints are not yet supported.")
            else:
                raise ValueError(f"Unsupported constraint type: {constraint.type}")

        # Return RefStructure
        ref_struct = structure_preparation.prepare_structure(
            chains, connections, metadata
        )
        return ref_struct, constraints

    def resolve_structure_sources(
        self,
        ref_struct: RefStructure,
        input: query.Query,
        rng: np.random.Generator,
    ) -> ResolvedStructureSources:
        """Load custom sources and normalize them by asym_id."""
        metadata_by_name = {chain.name: chain for chain in ref_struct.metadata.chains}
        apo_groups: list[list[dict[int, np.ndarray]]] = []
        prior_groups: list[list[dict[int, np.ndarray]]] = []
        struct_token_records: list[list[dict]] = []

        for sequence in input.sequences:
            if not isinstance(sequence, query.ProteinSequence):
                continue
            assert sequence.apo is not None
            asym_ids = [metadata_by_name[name].asym_id for name in sequence.ids]
            apos, priors, records = self._load_monomer_sources(
                asym_ids,
                sequence.sequence,
                sequence.apo,
                sequence.prior or sequence.apo,
                f"{input.name}:{','.join(sequence.ids)}",
            )
            apo_groups.append(apos)
            prior_groups.append(priors)
            struct_token_records.extend(records)

        for sequence_group in input.multimer_sequences:
            assert sequence_group.apo is not None
            component_asym_ids: tuple[list[int], list[int]] = ([], [])
            for id_pair in sequence_group.ids:
                for component_i, chain_name in enumerate(id_pair):
                    component_asym_ids[component_i].append(
                        metadata_by_name[chain_name].asym_id
                    )

            physical_ids = [name for id_pair in sequence_group.ids for name in id_pair]
            apos, priors, records = self._load_multimer_sources(
                component_asym_ids,
                (sequence_group.sequence1, sequence_group.sequence2),
                sequence_group.apo,
                sequence_group.prior or sequence_group.apo,
                f"{input.name}:{','.join(physical_ids)}",
            )
            apo_groups.append(apos)
            prior_groups.append(priors)
            struct_token_records.extend(records)

        num_apo = min(max(map(len, apo_groups), default=1), 5)
        apo_coords: dict[int, np.ndarray] = {}
        for sources in apo_groups:
            for asym_id, first_coords in sources[0].items():
                apo_coords[asym_id] = np.stack(
                    [
                        sources[i][asym_id]
                        if i < len(sources)
                        else np.full_like(first_coords, np.nan)
                        for i in range(num_apo)
                    ]
                )

        prior_sources = self._sample_prior_sources(ref_struct, prior_groups, rng)
        return ResolvedStructureSources(
            num_apo=num_apo,
            apo_coords=apo_coords,
            prior_sources=prior_sources,
            struct_token_records=[records[:num_apo] for records in struct_token_records],
        )

    def _load_monomer_sources(
        self,
        asym_ids: list[int],
        target_sequence: str,
        apo_paths: list[str],
        prior_paths: list[str],
        label: str,
    ) -> tuple[
        list[dict[int, np.ndarray]],
        list[dict[int, np.ndarray]],
        list[list[dict]],
    ]:
        def read(path: str) -> _AlignedStructureSource:
            source_sequence, source_coords = read_protein_structure(path)
            return self._align_structure_source(
                path,
                label,
                target_sequence,
                source_sequence,
                source_coords,
            )

        def collect_coords(source: _AlignedStructureSource) -> dict[int, np.ndarray]:
            length = len(target_sequence)
            coords = np.full((length, 37, 3), np.nan, dtype=np.float32)
            target_start, target_end = source.target_range
            coords[target_start:target_end] = source.coords
            return {asym_id: coords.copy() for asym_id in asym_ids}

        apo_sources: list[dict[int, np.ndarray]] = []
        token_records: list[dict] = []
        for path in apo_paths:
            source = read(path)
            apo_sources.append(collect_coords(source))
            token_records.append(
                {
                    "seq": source.sequence,
                    "coords": source.coords,
                    "targets": [(asym_id, *source.target_range) for asym_id in asym_ids],
                }
            )

        prior_sources = [collect_coords(read(path)) for path in prior_paths]
        return apo_sources, prior_sources, [token_records]

    def _load_multimer_sources(
        self,
        component_asym_ids: tuple[list[int], list[int]],
        target_sequences: tuple[str, str],
        apo_paths: list[str],
        prior_paths: list[str],
        label: str,
    ) -> tuple[
        list[dict[int, np.ndarray]],
        list[dict[int, np.ndarray]],
        list[list[dict]],
    ]:
        def read(
            path: str,
        ) -> tuple[_AlignedStructureSource, _AlignedStructureSource]:
            chain_records = list(
                read_protein_multimer_structure(pathlib.Path(path)).values()
            )
            if len(chain_records) != 2:
                raise ValueError(
                    f"Protein multimer source for {label} in {path} must contain "
                    f"exactly two non-empty protein chains, found "
                    f"{len(chain_records)}."
                )
            components = tuple(
                self._align_structure_source(
                    path,
                    f"Protein multimer component {i} for {label}",
                    target_sequence,
                    record["seq"],
                    record["coords"],
                )
                for i, (record, target_sequence) in enumerate(
                    zip(chain_records, target_sequences, strict=True),
                    start=1,
                )
            )
            return components[0], components[1]

        def collect_coords(
            components: tuple[_AlignedStructureSource, _AlignedStructureSource],
        ) -> dict[int, np.ndarray]:
            source: dict[int, np.ndarray] = {}
            for target_sequence, asym_ids, component in zip(
                target_sequences,
                component_asym_ids,
                components,
                strict=True,
            ):
                length = len(target_sequence)
                coords = np.full((length, 37, 3), np.nan, dtype=np.float32)
                target_start, target_end = component.target_range
                coords[target_start:target_end] = component.coords
                source.update({asym_id: coords.copy() for asym_id in asym_ids})
            return source

        apo_sources: list[dict[int, np.ndarray]] = []
        token_records: list[list[dict]] = [[] for _ in target_sequences]
        for path in apo_paths:
            components = read(path)
            apo_sources.append(collect_coords(components))
            for component_i, (asym_ids, component) in enumerate(
                zip(component_asym_ids, components, strict=True)
            ):
                token_records[component_i].append(
                    {
                        "seq": component.sequence,
                        "coords": component.coords,
                        "targets": [
                            (asym_id, *component.target_range) for asym_id in asym_ids
                        ],
                    }
                )

        prior_sources = [collect_coords(read(path)) for path in prior_paths]
        return apo_sources, prior_sources, token_records

    def _align_structure_source(
        self,
        path: str,
        label: str,
        target_sequence: str,
        source_sequence: str,
        source_coords: np.ndarray,
    ) -> _AlignedStructureSource:
        source_length = len(source_sequence)
        assert source_coords.shape == (source_length, 37, 3)

        if target_sequence == source_sequence:
            return _AlignedStructureSource(
                sequence=source_sequence,
                coords=source_coords,
                target_range=(0, source_length),
            )

        mapping, num_matches = _best_sequence_mapping(target_sequence, source_sequence)
        if num_matches == 0:
            raise ValueError(
                f"Structure sequence mismatch for {label} in {path}: no matching "
                f"residues between expected {target_sequence} and got "
                f"{source_sequence}."
            )

        target_start, target_end, source_start, source_end = mapping
        num_mapped = target_end - target_start
        self.logger.warning(
            "Aligned structure sequence for %s in %s: query length %d, source "
            "length %d, %d/%d mapped residues match, %d substitutions, %d "
            "query residues without coordinates, and %d ignored source residues.",
            label,
            path,
            len(target_sequence),
            source_length,
            num_matches,
            num_mapped,
            num_mapped - num_matches,
            len(target_sequence) - num_mapped,
            source_length - num_mapped,
        )
        return _AlignedStructureSource(
            sequence=source_sequence[source_start:source_end],
            coords=source_coords[source_start:source_end],
            target_range=(target_start, target_end),
        )

    def _sample_prior_sources(
        self,
        ref_struct: RefStructure,
        prior_groups: list[list[dict[int, np.ndarray]]],
        rng: np.random.Generator,
    ) -> list[dict[int, np.ndarray]]:
        """Randomly select one source per rigid chain group for every prior."""
        metadata_by_asym_id = {
            chain.asym_id: chain for chain in ref_struct.metadata.chains
        }
        rigid_groups: list[tuple[list[int], list[dict[int, np.ndarray]]]] = []
        for candidates in prior_groups:
            asym_ids_by_uid: dict[int, list[int]] = defaultdict(list)
            for asym_id in candidates[0]:
                uid = metadata_by_asym_id[asym_id].prior_uid
                assert uid is not None
                asym_ids_by_uid[uid].append(asym_id)
            rigid_groups.extend(
                (asym_ids, candidates) for asym_ids in asym_ids_by_uid.values()
            )

        global_sources: list[dict[int, np.ndarray]] = []
        for _ in range(self.num_prior_samples):
            global_source: dict[int, np.ndarray] = {}
            for asym_ids, candidates in rigid_groups:
                candidate = candidates[int(rng.integers(len(candidates)))]
                global_source.update({i: candidate[i].copy() for i in asym_ids})

            for chain in ref_struct.chains:
                if chain.is_protein:
                    continue
                if chain.is_nucleic_acid:
                    global_source[chain.asym_id] = np.full(
                        (chain.num_residues, 29, 3),
                        np.nan,
                        dtype=np.float32,
                    )
                else:
                    global_source[chain.asym_id] = self._get_ligand_prior_source(
                        chain, rng
                    )
            global_sources.append(global_source)
        return global_sources

    def _get_ligand_prior_source(
        self,
        chain: Chain,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate one ligand conformer in residue-major prior-source format."""
        assert chain.is_ligand
        if chain.smiles is not None:
            ref_comp = Component.from_smiles("LIG", chain.smiles)
            coords = ref_comp.get_ref_conformer(rng, train=False)
        else:
            coords = np.full_like(chain.atom.coords, np.nan)
            ccd_sequence = chain.get_ccd_sequence()
            for res_i, code in enumerate(ccd_sequence):
                if code not in self.ccd:
                    self.logger.warning(
                        f"CCD code {code} not found for ligand chain "
                        f"{chain.asym_id}. Filling with NaN coordinates."
                    )
                    continue

                ref_comp = self.ccd[code]
                ref_pos = ref_comp.get_ref_conformer(rng, train=False)
                ref_atom_order = ref_comp.get_atom_index_map()
                src_atom_indices: list[int] = []
                dst_atom_indices: list[int] = []
                res_idx = res_i + 1
                for atom_i in chain.residue.iter_residue_atoms(res_idx):
                    atom_name = chain.atom.name[atom_i]
                    if atom_name in ref_atom_order:
                        src_atom_indices.append(ref_atom_order[atom_name])
                        dst_atom_indices.append(atom_i)
                coords[dst_atom_indices] = ref_pos[src_atom_indices]

        return np.expand_dims(coords, axis=1).astype(np.float32)

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
            return structure_preparation.prepare_ref_chain(
                seq.ctype,
                seq.ccd_sequence,
                ccd=self.ccd,
                entity_id=entity_id,
                bonded_atoms=bonded_atoms,
            )
        elif isinstance(seq, query.LigandSequence):
            # Load ccd or smiles
            ctype = C.ChainType.LIGAND
            if seq.ccd_ids is not None:
                ccd_ids = seq.ccd_ids
                smiles = None
            else:
                assert seq.smiles is not None, (
                    "Either CCD code or SMILES must be provided."
                )
                ccd_ids = [f"LIG{entity_id}"]
                smiles = seq.smiles

            return structure_preparation.prepare_ref_chain(
                ctype,
                ccd_ids,
                ccd=self.ccd,
                smiles=smiles,
                entity_id=entity_id,
                bonded_atoms=bonded_atoms,
            )
        else:
            raise ValueError(f"Unsupported sequence type: {type(seq)}")
