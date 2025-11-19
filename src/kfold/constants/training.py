import enum


# For model training
class LDDTType(enum.Enum):  # TODO: add modified, Unresolved type
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
    MODIFIED = "modified"


LDDTWeightsBoltz = {
    LDDTType.DNA_PROTEIN: 5.0,
    LDDTType.RNA_PROTEIN: 5.0,
    LDDTType.LIGAND_PROTEIN: 20.0,
    LDDTType.DNA_LIGAND: 2.0,
    LDDTType.RNA_LIGAND: 2.0,
    LDDTType.INTRA_LIGAND: 20.0,
    LDDTType.INTRA_DNA: 2.0,
    LDDTType.INTRA_RNA: 8.0,
    LDDTType.INTRA_PROTEIN: 20.0,
    LDDTType.PROTEIN_PROTEIN: 20.0,
    LDDTType.MODIFIED: 0.0,  # Not used
}

LDDTWeightsAF3 = {
    LDDTType.DNA_PROTEIN: 10.0,
    LDDTType.RNA_PROTEIN: 10.0,
    LDDTType.LIGAND_PROTEIN: 10.0,
    LDDTType.DNA_LIGAND: 5.0,
    LDDTType.RNA_LIGAND: 5.0,
    LDDTType.INTRA_LIGAND: 20.0,
    LDDTType.INTRA_DNA: 4.0,
    LDDTType.INTRA_RNA: 16.0,
    LDDTType.INTRA_PROTEIN: 20.0,
    LDDTType.PROTEIN_PROTEIN: 20.0,
    LDDTType.MODIFIED: 10.0,
}
