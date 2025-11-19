import enum
from enum import IntEnum


# Same to Boltz's order
class ChainType(IntEnum):
    PROTEIN = 0
    DNA = 1
    RNA = 2
    LIGAND = 3


# TODO: may want to add some mmcif-related informations for data preprocessing
# e.g., mmcif chain type naming.

class OutType(str, enum.Enum): # TODO: add modified, Unresolved type
    DNA_PROTEIN = "dna_protein"
    RNA_PROTEIN = "rna_protein"
    LIGAND_PROTEIN = "ligand_protein"
    DNA_LIGAND = "dna_ligand"
    RNA_LIGAND = "rna_ligand"
    INTRA_LIGAND = "intra_ligand"
    INTRA_DNA = "intra_dna"
    INTRA_RNA = "intra_rna"
    INTRA_PROTEIN = "intra_protein"
    PROTEIN_PROTEIN = "protein_protein"
    # MODIFIED = "modified"
    # UNRESOLVED = "unresolved"

OutTypeWeightsBoltz = { # TODO: add modified, Unresolved type
    OutType.DNA_PROTEIN: 5.0,
    OutType.RNA_PROTEIN: 5.0,
    OutType.LIGAND_PROTEIN: 20.0,
    OutType.DNA_LIGAND: 2.0,
    OutType.RNA_LIGAND: 2.0,
    OutType.INTRA_LIGAND: 20.0,
    OutType.INTRA_DNA: 2.0,
    OutType.INTRA_RNA: 8.0,
    OutType.INTRA_PROTEIN: 20.0,
    OutType.PROTEIN_PROTEIN: 20.0,
}

OutTypeWeightsAF3Initial = {
    OutType.DNA_PROTEIN: 10.0,
    OutType.RNA_PROTEIN: 10.0,
    OutType.LIGAND_PROTEIN: 10.0,
    OutType.DNA_LIGAND: 5.0,
    OutType.RNA_LIGAND: 5.0,
    OutType.INTRA_LIGAND: 20.0,
    OutType.INTRA_DNA: 4.0,
    OutType.INTRA_RNA: 16.0,
    OutType.INTRA_PROTEIN: 20.0,
    OutType.PROTEIN_PROTEIN: 20.0,
    # OutType.MODIFIED: 10.0,
    # OutType.UNRESOLVED: 10.0,
}

OutTypeWeightsAF3Finetune = {
    OutType.DNA_PROTEIN: 10.0,
    OutType.RNA_PROTEIN: 2.0,
    OutType.LIGAND_PROTEIN: 10.0,
    OutType.DNA_LIGAND: 5.0,
    OutType.RNA_LIGAND: 2.0,
    OutType.INTRA_LIGAND: 20.0,
    OutType.INTRA_DNA: 4.0,
    OutType.INTRA_RNA: 16.0,
    OutType.INTRA_PROTEIN: 20.0,
    OutType.PROTEIN_PROTEIN: 20.0,
    # OutType.MODIFIED: 0.0,
    # OutType.UNRESOLVED: 10.0,
}