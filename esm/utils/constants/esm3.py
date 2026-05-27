import os
from functools import cache
from pathlib import Path

from huggingface_hub import snapshot_download

SEQUENCE_BOS_TOKEN = 0
SEQUENCE_PAD_TOKEN = 1
SEQUENCE_EOS_TOKEN = 2
SEQUENCE_CHAINBREAK_TOKEN = 31
SEQUENCE_MASK_TOKEN = 32

VQVAE_CODEBOOK_SIZE = 4096
VQVAE_SPECIAL_TOKENS = {
    "MASK": VQVAE_CODEBOOK_SIZE,
    "EOS": VQVAE_CODEBOOK_SIZE + 1,
    "BOS": VQVAE_CODEBOOK_SIZE + 2,
    "PAD": VQVAE_CODEBOOK_SIZE + 3,
    "CHAINBREAK": VQVAE_CODEBOOK_SIZE + 4,
}
VQVAE_DIRECTION_LOSS_BINS = 16
VQVAE_PAE_BINS = 64
VQVAE_MAX_PAE_BIN = 31.0
VQVAE_PLDDT_BINS = 50

STRUCTURE_MASK_TOKEN = VQVAE_SPECIAL_TOKENS["MASK"]
STRUCTURE_BOS_TOKEN = VQVAE_SPECIAL_TOKENS["BOS"]
STRUCTURE_EOS_TOKEN = VQVAE_SPECIAL_TOKENS["EOS"]
STRUCTURE_PAD_TOKEN = VQVAE_SPECIAL_TOKENS["PAD"]
STRUCTURE_CHAINBREAK_TOKEN = VQVAE_SPECIAL_TOKENS["CHAINBREAK"]
STRUCTURE_UNDEFINED_TOKEN = 955

SASA_PAD_TOKEN = 0

SS8_PAD_TOKEN = 0

INTERPRO_PAD_TOKEN = 0

RESIDUE_PAD_TOKEN = 0

CHAIN_BREAK_STR = "|"

SEQUENCE_BOS_STR = "<cls>"
SEQUENCE_EOS_STR = "<eos>"

MASK_STR_SHORT = "_"
SEQUENCE_MASK_STR = "<mask>"
SASA_MASK_STR = "<unk>"
SS8_MASK_STR = "<unk>"

# fmt: off
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]
# fmt: on

SEQUENCE_STANDARD_AA_MIN_TOKEN = 4  # L
SEQUENCE_STANDARD_AA_MAX_TOKEN = 24  # X (exclusive)

SSE_8CLASS_VOCAB = "GHITEBSC"
SSE_3CLASS_VOCAB = "HEC"
SSE_8CLASS_TO_3CLASS_MAP = {
    "G": "H",
    "H": "H",
    "I": "H",
    "T": "C",
    "E": "E",
    "B": "E",
    "S": "C",
    "C": "C",
}

SASA_DISCRETIZATION_BOUNDARIES = [
    0.8,
    4.0,
    9.6,
    16.4,
    24.5,
    32.9,
    42.0,
    51.5,
    61.2,
    70.9,
    81.6,
    93.3,
    107.2,
    125.4,
    151.4,
]

MAX_RESIDUE_ANNOTATIONS = 16


TFIDF_VECTOR_SIZE = 58641

FUNCTION_TOKENS_DEPTH = 8


@staticmethod
@cache
def data_root(model: str):
    if "INFRA_PROVIDER" in os.environ:
        return Path("")
    # Try to download from huggingface if it doesn't exist
    if model.startswith("esm3"):
        path = Path(snapshot_download(repo_id="biohub/esm3-sm-open-v1"))
    elif model.startswith("esmc-300"):
        path = Path(snapshot_download(repo_id="biohub/esmc-300m-2024-12"))
    elif model.startswith("esmc-600"):
        path = Path(snapshot_download(repo_id="biohub/esmc-600m-2024-12"))
    elif model.startswith("esmc-6b"):
        path = Path(snapshot_download(repo_id="biohub/esmc-6b-2024-12"))
    else:
        raise ValueError(f"{model=} is an invalid model name.")
    return path


IN_REPO_DATA_FOLDER = Path(__file__).parents[2] / "data"

INTERPRO_ENTRY = IN_REPO_DATA_FOLDER / "entry_list_safety_29026.list"
INTERPRO_HIERARCHY = IN_REPO_DATA_FOLDER / "ParentChildTreeFile.txt"
INTERPRO2GO = IN_REPO_DATA_FOLDER / "ParentChildTreeFile.txt"
INTERPRO_2ID = "data/tag_dict_4_safety_filtered.json"

LSH_TABLE_PATHS = {"8bit": "data/hyperplanes_8bit_58641.npz"}

KEYWORDS_VOCABULARY = (
    IN_REPO_DATA_FOLDER / "keyword_vocabulary_safety_filtered_58641.txt"
)
KEYWORDS_IDF = IN_REPO_DATA_FOLDER / "keyword_idf_safety_filtered_58641.npy"

RESID_CSV = "data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv"
INTERPRO2KEYWORDS = IN_REPO_DATA_FOLDER / "interpro_29026_to_keywords_58641.csv"
