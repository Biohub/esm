"""Constants for the ESMFold2 input pipeline.

Includes molecule types, residue types, vocabularies, atom lists, and element data.
"""

# =============================================================================
# Molecule types
# =============================================================================

MOL_TYPE_PROTEIN = 0
MOL_TYPE_DNA = 1
MOL_TYPE_RNA = 2
MOL_TYPE_NONPOLYMER = 3

# =============================================================================
# Residue type indices
# =============================================================================

# Standard amino acids (indices 2-21), MSE mapped to MET
PROTEIN_RESIDUE_TO_RES_TYPE = {
    "ALA": 2,
    "ARG": 3,
    "ASN": 4,
    "ASP": 5,
    "CYS": 6,
    "GLN": 7,
    "GLU": 8,
    "GLY": 9,
    "HIS": 10,
    "ILE": 11,
    "LEU": 12,
    "LYS": 13,
    "MET": 14,
    "PHE": 15,
    "PRO": 16,
    "SER": 17,
    "THR": 18,
    "TRP": 19,
    "TYR": 20,
    "VAL": 21,
    "MSE": 14,  # Selenomethionine -> MET
}
PROTEIN_UNK_RES_TYPE = 22

# RNA nucleotides (indices 23-26, unknown=27)
RNA_RESIDUE_TO_RES_TYPE = {"A": 23, "G": 24, "C": 25, "U": 26}
RNA_UNK_RES_TYPE = 27

# DNA nucleotides (indices 28-31, unknown=32)
DNA_RESIDUE_TO_RES_TYPE = {"DA": 28, "DG": 29, "DC": 30, "DT": 31}
DNA_UNK_RES_TYPE = 32

GAP_RES_TYPE = 32

# =============================================================================
# Vocabularies
# =============================================================================

# 3-letter to 1-letter codes for proteins
PROTEIN_3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}

# 1-letter to 3-letter codes
PROTEIN_1TO3 = {v: k for k, v in PROTEIN_3TO1.items() if k != "MSE"}
PROTEIN_1TO3["X"] = "UNK"

# DNA 1-letter to CCD code
DNA_1TO3 = {"A": "DA", "T": "DT", "C": "DC", "G": "DG"}

# RNA 1-letter to CCD code
RNA_1TO3 = {"A": "A", "U": "U", "C": "C", "G": "G"}

# ESM-2 input_ids vocabulary for proteins
ESM_PROTEIN_VOCAB = {
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
    "X": 3,  # Unknown
}

# For DNA/RNA/ligands
DNA_RNA_LIGAND_INPUT_ID = 24

# MSA tokens
MSA_PAD_TOKEN_ID = 0
MSA_GAP_TOKEN_ID = 1  # Gap/insertion token for MSA

# res_type int -> CCD component ID (for conformer lookup)
RES_TYPE_TO_CCD = {
    # Proteins (2-22)
    2: "ALA",
    3: "ARG",
    4: "ASN",
    5: "ASP",
    6: "CYS",
    7: "GLN",
    8: "GLU",
    9: "GLY",
    10: "HIS",
    11: "ILE",
    12: "LEU",
    13: "LYS",
    14: "MET",
    15: "PHE",
    16: "PRO",
    17: "SER",
    18: "THR",
    19: "TRP",
    20: "TYR",
    21: "VAL",
    22: "UNK",
    # RNA (23-27)
    23: "A",
    24: "G",
    25: "C",
    26: "U",
    27: "N",
    # DNA (28-32)
    28: "DA",
    29: "DG",
    30: "DC",
    31: "DT",
    32: "DN",
}

# =============================================================================
# Charged atoms at physiological pH
# =============================================================================

CHARGED_ATOMS: dict[tuple[str, str], int] = {
    ("LYS", "NZ"): 1,
    ("ARG", "NH2"): 1,
    ("HIS", "ND1"): 1,
    ("PO4", "O2"): -1,
    ("PO4", "O3"): -1,
    ("PO4", "O4"): -1,
    ("SO4", "O3"): -1,
    ("SO4", "O4"): -1,
    ("MG", "MG"): 2,
    ("ZN", "ZN"): 2,
    ("CA", "CA"): 2,
    ("FE2", "FE"): 2,
    ("MN", "MN"): 2,
    ("CO", "CO"): 2,
    ("NCO", "CO"): 3,
    ("CU", "CU"): 2,
    ("NI", "NI"): 2,
    ("K", "K"): 1,
    ("NA", "NA"): 1,
    ("CD", "CD"): 2,
    ("CL", "CL"): -1,
    ("ACT", "OXT"): -1,
    ("NAD", "O2N"): -1,
    ("NAD", "N1N"): 1,
    ("NAP", "O2N"): -1,
    ("NAP", "N1N"): 1,
    ("IMD", "N3"): 1,
    ("SAM", "SD"): 1,
    ("FE", "FE"): 3,
    ("A1BH3", "N3"): 1,
}

# =============================================================================
# Element atomic numbers (Z=1 to 92)
# =============================================================================

ELEMENT_TO_ATOMIC_NUM = {
    "H": 1,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "K": 19,
    "CA": 20,
    "SC": 21,
    "TI": 22,
    "V": 23,
    "CR": 24,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "GA": 31,
    "GE": 32,
    "AS": 33,
    "SE": 34,
    "BR": 35,
    "KR": 36,
    "RB": 37,
    "SR": 38,
    "Y": 39,
    "ZR": 40,
    "NB": 41,
    "MO": 42,
    "TC": 43,
    "RU": 44,
    "RH": 45,
    "PD": 46,
    "AG": 47,
    "CD": 48,
    "IN": 49,
    "SN": 50,
    "SB": 51,
    "TE": 52,
    "I": 53,
    "XE": 54,
    "CS": 55,
    "BA": 56,
    "LA": 57,
    "CE": 58,
    "PR": 59,
    "ND": 60,
    "PM": 61,
    "SM": 62,
    "EU": 63,
    "GD": 64,
    "TB": 65,
    "DY": 66,
    "HO": 67,
    "ER": 68,
    "TM": 69,
    "YB": 70,
    "LU": 71,
    "HF": 72,
    "TA": 73,
    "W": 74,
    "RE": 75,
    "OS": 76,
    "IR": 77,
    "PT": 78,
    "AU": 79,
    "HG": 80,
    "TL": 81,
    "PB": 82,
    "BI": 83,
    "PO": 84,
    "AT": 85,
    "RN": 86,
    "FR": 87,
    "RA": 88,
    "AC": 89,
    "TH": 90,
    "PA": 91,
    "U": 92,
}

# Inverse mapping: atomic number → element symbol
ELEMENT_NUMBER_TO_SYMBOL = {v: k for k, v in ELEMENT_TO_ATOMIC_NUM.items()}

# =============================================================================
# Standard heavy atoms per residue type
# =============================================================================

PROTEIN_HEAVY_ATOMS = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "MSE": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "UNK": ["N", "CA", "C", "O"],
}

DNA_HEAVY_ATOMS = {
    "DA": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
    ],
    "DG": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
    ],
    "DC": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
    ],
    "DT": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C7",
        "C6",
    ],
}

RNA_HEAVY_ATOMS = {
    "A": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
    ],
    "G": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
    ],
    "C": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
    ],
    "U": [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C6",
    ],
}

# Unknown nucleotide backbone atoms
DNA_BACKBONE_ATOMS = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
]
RNA_BACKBONE_ATOMS = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
]
