import enum


# For model training
# TODO: add modified and resolved
class LDDTType(enum.Enum):
    PROTEIN_PROTEIN = "protein_protein"
    DNA_PROTEIN = "dna_protein"
    RNA_PROTEIN = "rna_protein"
    DNA_LIGAND = "dna_ligand"
    LIGAND_PROTEIN = "ligand_protein"
    RNA_LIGAND = "rna_ligand"
    INTRA_PROTEIN = "intra_protein"
    INTRA_DNA = "intra_dna"
    INTRA_RNA = "intra_rna"
    INTRA_LIGAND = "intra_ligand"


LDDTWeightsAF3 = {
    LDDTType.PROTEIN_PROTEIN: 20.0,
    LDDTType.DNA_PROTEIN: 10.0,
    LDDTType.RNA_PROTEIN: 10.0,
    LDDTType.DNA_LIGAND: 5.0,
    LDDTType.LIGAND_PROTEIN: 10.0,
    LDDTType.RNA_LIGAND: 5.0,
    LDDTType.INTRA_PROTEIN: 20.0,
    LDDTType.INTRA_DNA: 4.0,
    LDDTType.INTRA_RNA: 16.0,
    LDDTType.INTRA_LIGAND: 20.0,
}
LDDTWeightsBoltz = {
    LDDTType.PROTEIN_PROTEIN: 20.0,
    LDDTType.DNA_PROTEIN: 5.0,
    LDDTType.RNA_PROTEIN: 5.0,
    LDDTType.DNA_LIGAND: 2.0,
    LDDTType.LIGAND_PROTEIN: 20.0,
    LDDTType.RNA_LIGAND: 2.0,
    LDDTType.INTRA_PROTEIN: 20.0,
    LDDTType.INTRA_DNA: 2.0,
    LDDTType.INTRA_RNA: 8.0,
    LDDTType.INTRA_LIGAND: 20.0,
}
