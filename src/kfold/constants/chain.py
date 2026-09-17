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

from enum import IntEnum


class ChainType(IntEnum):
    PROTEIN = 0
    DNA = 1
    RNA = 2
    LIGAND = 3

    @property
    def is_polymer(self) -> bool:
        return self in {ChainType.PROTEIN, ChainType.DNA, ChainType.RNA}

    @property
    def is_nonpolymer(self) -> bool:
        return self in {ChainType.LIGAND}

    @property
    def is_protein(self) -> bool:
        return self is ChainType.PROTEIN

    @property
    def is_nucleic_acid(self) -> bool:
        return self in {ChainType.DNA, ChainType.RNA}

    @property
    def is_dna(self) -> bool:
        return self is ChainType.DNA

    @property
    def is_rna(self) -> bool:
        return self is ChainType.RNA

    @property
    def is_ligand(self) -> bool:
        return self is ChainType.LIGAND

    def __str__(self) -> str:
        match self:
            case ChainType.PROTEIN:
                return "Protein"
            case ChainType.DNA:
                return "DNA"
            case ChainType.RNA:
                return "RNA"
            case ChainType.LIGAND:
                return "Ligand"


class SubChainType(IntEnum):
    PROTEIN = 0
    DNA = 1
    RNA = 2
    SMALL_MOLECULE = 3
    ION = 4
    PEPTIDE = 5
    GLYCAN = 6
    COVALENT_LIGAND = 7

    def __str__(self) -> str:
        match self:
            case SubChainType.PROTEIN:
                return "Protein"
            case SubChainType.DNA:
                return "DNA"
            case SubChainType.RNA:
                return "RNA"
            case SubChainType.SMALL_MOLECULE:
                return "SmallMolecule"
            case SubChainType.ION:
                return "Ion"
            case SubChainType.PEPTIDE:
                return "Peptide"
            case SubChainType.GLYCAN:
                return "Glycan"
            case SubChainType.COVALENT_LIGAND:
                return "CovalentLigand"


# TODO: may want to add some mmcif-related informations for data preprocessing
# e.g., mmcif chain type naming.
