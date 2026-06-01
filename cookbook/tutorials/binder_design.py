# /// script
# requires-python = "<=3.13"
# dependencies = [
#     "abnumber",
#     "esm@git+https://github.com/Biohub/esm.git@main",
#     "modal",
# ]
# ///
"""
Code for binder design with ESMFold2 and ESMC.

As described in [Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf).
"""

import logging
import math
import os
import random
import string
from dataclasses import dataclass
from functools import cache, partial
from typing import Any

import biotite.structure
import modal
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
from transformers.models.esmc.modeling_esmc import (
    UnifiedTransformerBlock as TransformerBlock,
)
from transformers.models.esmc.tokenization_esmc import ESMCTokenizer
from transformers.models.esmfold2.modeling_esmfold2_common import (
    CUE_AVAILABLE,
    PairUpdateBlock,
)
from transformers.models.esmfold2.modeling_esmfold2_common import (
    _seed_context as seed_context,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    MSAEncoder as ESMFold2MSAEncoder,
)

from esm.models.esmfold2 import (
    ELEMENT_NUMBER_TO_SYMBOL,
    ProteinInput,
    StructurePredictionInput,
    load_ccd,
    prepare_esmfold2_input,
)
from esm.models.esmfold2.constants import (
    MOL_TYPE_NONPOLYMER,
    PROTEIN_1TO3,
    PROTEIN_3TO1,
    RES_TYPE_TO_CCD,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---- Constants ----


# General
TOKENS = ["<pad>", "-"] + [RES_TYPE_TO_CCD[i] for i in range(2, 33)]
ELEMENTS = ["X"] * (max(ELEMENT_NUMBER_TO_SYMBOL) + 1)
ELEMENTS[0] = "<pad>"
for _atomic_num, _symbol in ELEMENT_NUMBER_TO_SYMBOL.items():
    ELEMENTS[_atomic_num] = _symbol[:1] + _symbol[1:].lower()
TOKEN_IDS = {token: idx for idx, token in enumerate(TOKENS)}
AA_DIMS = 20
# Cysteine index in the 20-dim AA space (TOKEN_IDS are offset by 2 for <pad> and -)
CYS_IDX = TOKEN_IDS[PROTEIN_1TO3["C"]] - 2
MUTABLE_TOKEN = "#"
# Contains AA chars at fixed positions and MUTABLE_TOKEN at mutable positions.
BinderPromptStr = str

# Design
LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2}
STEPS = 150
LOG_INTERVAL = 5
LEARNING_RATE = 0.1
TEMPERATURE_MIN = 1e-2
ESMC_MASK_FRACTION = 0.15
CHECKPOINT_LM = False
COMPILE = False
# NOTE - This significantly reduces VRAM usage.
# On config (target_name=cd45", binder_name="trastuzumab_framework_vhvl, batch_size=1)
# this reduces VRAM from 51GB -> 27GB.  And enables increasing batch size up to 6.
# We are testing this setting in silico, and may change the default to True, in the future.
REUSE_ESMC = False


# ---- Prompts ----


@dataclass(frozen=True)
class PromptFactory:
    """A simple factory for making binder prompt strings."""

    name: str
    template: str  # string with format fields
    length_ranges: dict[str, tuple[int, int]]  # map from field name tp length range
    is_antibody: bool  # Used to set LM loss weight for antibodies.

    def sample(self, seed: int) -> BinderPromptStr:
        random.seed(seed)
        return self.template.format(
            **{
                key: MUTABLE_TOKEN * random.randint(low, high)
                for key, (low, high) in self.length_ranges.items()
            }
        )


# fmt: off
BINDER_PROMPT_FACTORIES = {
    "minibinder": PromptFactory(name="minibinder", template="{seq}", length_ranges={"seq": (60, 200)}, is_antibody=False),
    "trastuzumab_framework_vhvl": PromptFactory(
        name="trastuzumab_framework_vhvl",
        template="EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}YIHWVRQAPGKGLEWVARI{hcdr2}TRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    ),
    "atezolizumab_framework_vhvl": PromptFactory(
        name="atezolizumab_framework_vhvl",
        template="EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}WIHWVRQAPGKGLEWVAWI{hcdr2}TYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    ),
    "ocankitug_framework_vhvl": PromptFactory(
        name="ocankitug_framework_vhvl",
        template="QVQLVQSGAEVKKPGSSVKVSCKAS{hcdr1}WMHWVRQAPGQGLEWMGII{hcdr2}TSLNQKFQGRVTITADTSTSTAYMELSSLRSEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (8, 14), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    )
}


TARGET_SEQUENCES = {
    # https://www.uniprot.org/uniprotkb/P08575  389-574
    "cd45": "GSPGEPQIIFCRSEAAHQGVITWNPPQRSFHNFTLCYIKETEKDCLNLDKNLIKYDLQNLKPYTKYVLSLHAYIIAKVQRNGSAAMCHFTTKSAPPSQVWNMTVSMTSDNSMHVKCRPPRDRNGPHERYHLEVEAGNTLVRNESHKNCDFRVKDLQYSTDYTFKAYFHNGDYPGEPFILHHSTSY",
    # https://www.uniprot.org/uniprotkb/P16410  37-155
    "ctla4": "MHVAQPAVVLASSRGIASFVCEYASPGKATEVRVTVLRQADSQVTEVCAATYMMGNELTFLDDSICTGTSSGNQVNLTIQGLRAMDTGLYICKVELMYPPPYYLGIGNGTQIYVIDPE",
    # https://www.uniprot.org/uniprotkb/P00533  333-524
    "egfr": "RKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCV",
    # https://www.uniprot.org/uniprotkb/Q9NZQ7  17-132
    "pd-l1": "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA",
    # https://www.uniprot.org/uniprotkb/P09619  125-312
    "pdgfr": "GFLPNDAEELFIFLTEITEITIPCRVTDPQLVVTLHEKKGDVALPVPYDHQRGFSGIFEDRSYICKTTIGDREVDSDAYYVYRLQVSSINVSVNAVQTVVRQGENITLMCIVIGNEVVNFEWTYPRKESGRLVEPVTDFLLDMPYHIRSILHIPSAELEDSGTYTCNVTESVNDHQDEKAINITVVE",
}
# fmt: on


# ---- Helper functions ----


def build_initial_soft_sequence_logits(sequence: str, batch_size: int) -> torch.Tensor:
    """
    Initialize logits with:
    - High confidence (10.0) for fixed positions
    - Random (~0) for mutable positions
    - -1e6 for cysteines
    """
    if all(aa == MUTABLE_TOKEN for aa in sequence):
        logits = 0.01 * torch.randn([batch_size, len(sequence), AA_DIMS])
        logits[:, :, CYS_IDX] = -1e6  # remove cysteines
    else:
        logits = torch.zeros([batch_size, len(sequence), AA_DIMS])
        for i, aa in enumerate(sequence):
            if aa == MUTABLE_TOKEN:  # mutable position - random
                logits[:, i, :] = 0.01 * torch.randn(batch_size, AA_DIMS)
                logits[:, i, CYS_IDX] = -1e6
            else:  # fixed position
                assert aa in PROTEIN_1TO3, aa
                token_id = TOKEN_IDS[PROTEIN_1TO3[aa]]
                logits[:, i, token_id - 2] = 10.0

    return logits.requires_grad_(True)


def build_gradient_mask(sequence: str, batch_size: int) -> torch.Tensor:
    """
    Build gradient mask [B, L, V]:
    - 0 for fixed (all amino acids)
    - 0 for cysteine at all positions
    - 1 for non-cysteine amino acids at mutable positions
    """
    mask = torch.ones([batch_size, len(sequence), AA_DIMS])
    fixed_positions = [i for i, aa in enumerate(sequence) if aa != MUTABLE_TOKEN]
    mask[:, fixed_positions, :] = 0.0
    mask[:, :, CYS_IDX] = 0.0
    return mask


def sequence_to_one_hot(sequence: str, device="cuda") -> torch.Tensor:
    """Convert target string to one-hot tensor [1, L_target, num_tokens]."""

    const_dict = {token: i for i, token in enumerate(TOKENS)}
    target_index = [const_dict[PROTEIN_1TO3[letter]] for letter in sequence]
    one_hot = F.one_hot(torch.tensor(target_index), num_classes=len(TOKENS))
    return one_hot.to(device).unsqueeze(0).float()


def get_mid_points() -> torch.Tensor:
    """128 distance bin midpoints (2p-52 Angstrom range)."""

    boundaries = torch.linspace(2, 52.0, 127)
    lower = torch.tensor([1.0])
    upper = torch.tensor([52.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    return (exp_boundaries[:-1] + exp_boundaries[1:]) / 2


def binned_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Entropy of distance distribution within cutoff (design losses only)."""

    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Mean of the smallest k values in x under mask along the last dimension."""

    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean along last axis."""

    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


# ---- Loss functions ----


def compute_contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    """Algorithm 12 Contact Losses.

    Entropy-based contact loss with sequence separation constraint."""

    con_loss = binned_entropy(distogram_logits, bin_distance, cutoff)
    position = torch.arange(distogram_logits.shape[1])
    p_dist = position[:, None] - position[None, :]
    if min_sep > 0:
        separation_mask = (torch.abs(p_dist) >= min_sep).to(distogram_logits.device)
        binder_mask = torch.logical_and(separation_mask, binder_mask)
    per_residue = masked_min_k(con_loss, mask=binder_mask, k=num_contacts).to(
        distogram_logits.device
    )
    return masked_average(per_residue, mask=chain_mask).to(distogram_logits.device)


def compute_intra_contact_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    """Binder internal contacts (k=2, min_sep=9, cutoff=14A)."""

    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=2,
        min_sep=9,
        cutoff=14.0,
        chain_mask=is_binder,
        binder_mask=is_binder,
    )


def compute_inter_contact_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    """Binder-target interface (k=1, min_sep=0, cutoff=22A)."""

    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=1 - is_binder,
        binder_mask=is_binder,
    )


def compute_globularity_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    """Algorithm 13 Globularity Loss.

    Radius of gyration vs theoretical packed protein."""

    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    bin_distance = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(bin_distance), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n**0.365)
    return F.elu(rg_term - rg_th)


def compute_structure_losses(
    distogram_logits: torch.Tensor, binder_length: int
) -> dict[str, torch.Tensor]:
    """Compute structural losses and a weighted total."""

    bin_distance = get_mid_points().to(distogram_logits.device)
    losses: dict[str, torch.Tensor] = {}
    losses["intra_contact_loss"] = compute_intra_contact_loss(
        distogram_logits, binder_length, bin_distance
    )
    losses["inter_contact_loss"] = compute_inter_contact_loss(
        distogram_logits, binder_length, bin_distance
    )
    losses["glob_loss"] = compute_globularity_loss(
        distogram_logits, binder_length, bin_distance
    )
    B = distogram_logits.size(0)
    total = torch.tensor([0.0] * B, device=distogram_logits.device, requires_grad=True)
    total = total + LOSS_WEIGHTS["intra_contact"] * losses["intra_contact_loss"]
    total = total + LOSS_WEIGHTS["inter_contact"] * losses["inter_contact_loss"]
    total = total + LOSS_WEIGHTS["glob"] * losses["glob_loss"]
    losses["total_loss"] = total
    return losses


# ---- Distogram iptm proxy ----


def _binding_confidence_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Pair entropy within cutoff."""

    probs = torch.softmax(dgram, dim=-1)
    cutoff_mask = bin_distance < cutoff
    p_cut = probs[..., cutoff_mask]
    p_cut = p_cut / (p_cut.sum(-1, keepdim=True) + 1e-8)
    return -(p_cut * torch.log(p_cut + 1e-10)).sum(-1)


def _entropy_to_confidence(mean_entropy: float) -> float:
    """Map mean pair entropy to [0, 1]; lower entropy → higher score."""
    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def _cdr_indices(binder_sequence: str) -> list[int]:
    """0-based binder indices for all Chothia CDRs."""
    from abnumber import Chain
    from abnumber.common import _anarci_align

    result = _anarci_align(
        sequences=[binder_sequence], scheme="chothia", allowed_species=None
    )[0]
    chains = [
        Chain("".join(result[i][0].values()), scheme="chothia")
        for i in range(len(result))
    ]
    if len(chains) == 2 and not chains[0].is_heavy_chain():
        chains.reverse()
    indices: list[int] = []
    for chain in chains:
        for cdr in (chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq):
            start = binder_sequence.find(cdr)
            assert start >= 0
            indices.extend(range(start, start + len(cdr)))
    return indices


def compute_distogram_iptm_proxy(
    distogram_logits: torch.Tensor,
    target_length: int,
    binder_sequence: str,
    is_antibody: bool,
) -> dict[str, float]:
    """Algorithm 15 Distogram ipTM Proxy.

    Distogram iptm proxy for a target|binder complex (binder at suffix).

    Returns distogram_iptm_proxy for all designs and
    cdr_distogram_iptm_proxy when the binder can be annotated as an
    antibody; otherwise the CDR score is NaN.
    """
    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]

    binder_length = len(binder_sequence)
    assert distogram_logits.shape[0] == target_length + binder_length

    bin_distance = get_mid_points().to(distogram_logits.device)
    binder_start = target_length

    def _mean_lowest_k(entropies: torch.Tensor, k: int) -> float:
        sorted_entropies, _ = torch.sort(entropies.reshape(-1))
        k = min(k, sorted_entropies.numel())
        return float(sorted_entropies[:k].mean())

    binder_to_target_entropy = _binding_confidence_entropy(
        distogram_logits[binder_start:, :target_length, :], bin_distance, cutoff=22.0
    )
    distogram_iptm_proxy = _entropy_to_confidence(
        _mean_lowest_k(binder_to_target_entropy, k=binder_length)
    )

    if not is_antibody:
        cdr_distogram_iptm_proxy = float("nan")
    else:
        cdr_indices = _cdr_indices(binder_sequence)
        cdr_rows = [binder_start + i for i in cdr_indices]
        cdr_to_target_entropy = _binding_confidence_entropy(
            distogram_logits[cdr_rows, :target_length, :], bin_distance, cutoff=22.0
        )
        cdr_distogram_iptm_proxy = _entropy_to_confidence(
            _mean_lowest_k(cdr_to_target_entropy, k=len(cdr_indices))
        )

    return {
        "distogram_iptm_proxy": distogram_iptm_proxy,
        "cdr_distogram_iptm_proxy": cdr_distogram_iptm_proxy,
    }


# ---- Folding ----


def _resize_tensor(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    current = tensor.shape[dim]
    if current >= size:
        return tensor.narrow(dim, 0, size)

    pad_shape = list(tensor.shape)
    pad_shape[dim] = size - current
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=dim)


_ATOM_FEATURE_DIMS = {
    "ref_pos": 0,
    "ref_element": 0,
    "ref_charge": 0,
    "ref_atom_name_chars": 0,
    "ref_space_uid": 0,
    "atom_attention_mask": 0,
    "atom_to_token": 0,
    "is_resolved": 0,
    "gt_coords": 1,
}


@cache
def _ensure_ccd_loaded() -> None:
    load_ccd()


def prepare_esmfold2_tensors(
    input: StructurePredictionInput,
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    max_seqs: int = 16384,
    pad_to_max_seqs: bool = False,
    seed: int | None = None,
    use_vectorized_msa_assembly: bool = True,
) -> dict[str, torch.Tensor]:
    del max_tokens, max_seqs, pad_to_max_seqs, use_vectorized_msa_assembly
    _ensure_ccd_loaded()
    features, _ = prepare_esmfold2_input(input, seed=seed)
    if max_atoms is not None:
        for key, dim in _ATOM_FEATURE_DIMS.items():
            if key in features:
                features[key] = _resize_tensor(features[key], dim=dim, size=max_atoms)
    return features


def fold_and_get_distogram(
    model: ESMFold2ExperimentalModel,
    target_seq: str,
    target_one_hot: torch.Tensor,
    design: torch.Tensor,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
) -> dict:
    """Prepare inputs, run model forward, return distogram_logits + raw output."""
    padding = (2, 11)
    padded_design = F.pad(design, padding, mode="constant", value=0)

    # Argmax to get the designed sequence string.
    token_lists = torch.argmax(padded_design, dim=-1)
    designed_seq = [
        [PROTEIN_3TO1[TOKENS[int(tkn.item())]] for tkn in token_list]
        for token_list in token_lists
    ]
    seq_list = [target_seq + "|" + "".join(seq) for seq in designed_seq]
    max_atoms = None if len(seq_list) == 1 else ((len(seq_list[0]) - 1) * 14) // 32 * 32

    inputs_list = []
    for seq in seq_list:
        sequences = {
            sequence: [str(idx)] for idx, sequence in enumerate(seq.split("|"))
        }
        inputs_raw = StructurePredictionInput(
            sequences=[
                ProteinInput(id=chain_id, sequence=sequence, msa=None)
                for sequence, chain_id in sequences.items()
            ]
        )
        inputs_list.append(prepare_esmfold2_tensors(inputs_raw, max_atoms=max_atoms))

    inputs = {
        key: torch.stack([inp[key] for inp in inputs_list], dim=0).cuda()
        for key in inputs_list[0]
    }
    inputs["res_type_soft"] = torch.cat(
        (target_one_hot.repeat(design.size(0), 1, 1), padded_design), dim=1
    )

    with seed_context(seed):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=num_sampling_steps,
            num_loops=num_loops,
            calculate_confidence=calculate_confidence,
            seed=seed,
        )

    result: dict = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "inputs_list": inputs_list,
        "output": output,
        "seq_list": seq_list,
    }
    if calculate_confidence:
        result.update(
            {
                "ptm": output.get("ptm"),
                "iptm": output.get("iptm"),
                "plddt": output.get("plddt"),
            }
        )
    return result


_CHAIN_ID_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _asym_id_to_chain_label(asym_id: int) -> str:
    if asym_id < 0:
        raise ValueError(f"asym_id must be >= 0, got {asym_id}")
    label = ""
    n = len(_CHAIN_ID_ALPHABET)
    while True:
        label = _CHAIN_ID_ALPHABET[asym_id % n] + label
        asym_id = asym_id // n - 1
        if asym_id < 0:
            return label


def to_atom_array(
    coords: np.ndarray,
    atom_to_token: np.ndarray,
    res_type: np.ndarray,
    residue_index: np.ndarray,
    asym_id: np.ndarray,
    mol_type: np.ndarray,
    ref_atom_name_chars: np.ndarray,
    ref_element: np.ndarray,
    atom_attention_mask: np.ndarray,
    plddt_per_atom: np.ndarray | None = None,
) -> biotite.structure.AtomArray:
    atoms = []
    for atom_i, (
        atom_coord,
        token_idx,
        atom_name_chars,
        element_idx,
        is_not_pad,
    ) in enumerate(
        zip(
            coords, atom_to_token, ref_atom_name_chars, ref_element, atom_attention_mask
        )
    ):
        if not is_not_pad:
            continue
        atoms.append(
            biotite.structure.Atom(
                coord=atom_coord,
                chain_id=_asym_id_to_chain_label(int(asym_id[token_idx])),
                res_id=residue_index[token_idx] + 1,
                res_name=TOKENS[res_type[token_idx]],
                atom_name="".join(chr(c + 32) for c in atom_name_chars if c != 0),
                element=ELEMENTS[element_idx],
                ins_code=" ",
                hetero=mol_type[token_idx] == MOL_TYPE_NONPOLYMER,
                b_factor=float(plddt_per_atom[atom_i])
                if plddt_per_atom is not None
                else 0.0,
            )
        )
    return biotite.structure.array(atoms)


def build_complex(
    inputs: dict[str, torch.Tensor], output: dict[str, Any]
) -> ProteinComplex:
    """Build ProteinComplex from model output."""
    atom_arr = to_atom_array(
        coords=output["sample_atom_coords"][0].cpu().numpy(),
        atom_to_token=inputs["atom_to_token"][0].cpu().numpy(),
        res_type=inputs["res_type"][0].cpu().numpy(),
        residue_index=inputs["token_index"][0].cpu().numpy(),
        asym_id=inputs["asym_id"][0].cpu().numpy(),
        mol_type=inputs["mol_type"][0].cpu().numpy(),
        ref_atom_name_chars=inputs["ref_atom_name_chars"][0].cpu().numpy(),
        ref_element=inputs["ref_element"][0].cpu().numpy(),
        atom_attention_mask=inputs["atom_attention_mask"][0].cpu().numpy(),
    )
    return ProteinComplex.from_chains(
        [ProteinChain.from_atomarray(a) for a in biotite.structure.chain_iter(atom_arr)]
    )


# ---- LM loss ----


@cache
def _folding_trunk_to_lm_aa_vocab_matrix(device: torch.device) -> torch.Tensor:
    """Build a matrix of shape [ft_aas=20, lm_aas=20]."""
    three_to_one_map = {v: k for k, v in PROTEIN_1TO3.items()}
    ft_aas = [three_to_one_map[tok_3letter] for tok_3letter in TOKENS[2:22]]

    lm_vocab = sorted(ESMCTokenizer().vocab.items(), key=lambda x: x[1])
    lm_aas = [lm_vocab[i][0] for i in range(4, 24)]

    ft_to_lm_aa_matrix = torch.zeros(20, 20)
    for ft_idx, ft_aa in enumerate(ft_aas):
        lm_idx = lm_aas.index(ft_aa)
        ft_to_lm_aa_matrix[ft_idx, lm_idx] = 1

    return ft_to_lm_aa_matrix.to(device=device)


def _one_hot_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.size(-1)).to(
        probs.dtype
    )


def _straight_through(discrete: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    return continuous + (discrete - continuous).detach()


def compute_esmc_pseudoperplexity_nll(
    esmc_model: ESMCForMaskedLM,
    binder_design: torch.Tensor,
    score_mask: torch.Tensor,
    batch_size: int = 4,
    n_passes: int = 4,
) -> torch.Tensor:
    """Algorithm 14 ESMC Pseudo-perplexity Sequence Regularization.

    Approximate pseudoperplexity NLL via multiple sampled masks."""
    device = binder_design.device
    lm_vocab_size = esmc_model.config.vocab_size
    model_dtype = esmc_model.esmc.embed.weight.dtype

    target_esm = binder_design @ _folding_trunk_to_lm_aa_vocab_matrix(device)
    input_esm = _straight_through(_one_hot_from_probs(target_esm), target_esm)
    input_ids = torch.zeros(
        (binder_design.size(0), binder_design.size(1) + 2, lm_vocab_size),
        dtype=model_dtype,
        device=device,
    )
    tokenizer = ESMCTokenizer()
    input_ids[:, 0, tokenizer.cls_token_id] = 1
    input_ids[:, -1, tokenizer.eos_token_id] = 1
    input_ids[:, 1:-1, 4:24] = input_esm.to(model_dtype)

    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
    elif score_mask.shape != binder_design.shape[:2]:
        raise ValueError(
            f"Expected score_mask with shape {(binder_design.size(0), binder_design.size(1))}, "
            f"got {tuple(score_mask.shape)}"
        )
    score_mask = score_mask.to(device=device, dtype=torch.bool)

    mask_token = torch.zeros(lm_vocab_size, dtype=model_dtype, device=device)
    mask_token[esmc_model.config.mask_token_id] = 1
    esmc = esmc_model.esmc

    losses = []
    for batch_idx in range(binder_design.size(0)):
        position_indices = score_mask[batch_idx].nonzero(as_tuple=False).flatten()
        num_positions = int(position_indices.numel())
        if num_positions == 0:
            raise ValueError(
                "ESMC pseudoperplexity score mask selected zero positions."
            )

        num_masked = max(1, math.ceil(ESMC_MASK_FRACTION * num_positions))
        random_scores = torch.rand((n_passes, num_positions), device=device)
        masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
        pass_masks = torch.zeros(
            (n_passes, binder_design.size(1)), dtype=torch.bool, device=device
        )
        pass_masks[
            torch.arange(n_passes, device=device)[:, None],
            position_indices[masked_offsets],
        ] = True

        masked_sequences = input_ids[batch_idx : batch_idx + 1].repeat(n_passes, 1, 1)
        mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
        masked_sequences[mask_rows, mask_cols + 1] = mask_token

        target_weights = target_esm[batch_idx]
        masked_nlls = []
        for start in range(0, n_passes, batch_size):
            stop = min(start + batch_size, n_passes)
            chunk = masked_sequences[start:stop]
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                hidden, *_ = esmc.transformer(
                    chunk @ esmc.embed.weight.to(chunk.dtype),
                    sequence_id=None,
                    layers_to_collect=[],
                    output_attentions=False,
                )
                logits = esmc_model.lm_head(hidden)
            log_probs = logits.log_softmax(dim=-1)[:, 1:-1, 4:24]
            nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(
                dim=-1
            )
            masked_nlls.append(nlls[pass_masks[start:stop]])

        losses.append(torch.cat(masked_nlls, dim=0).mean())

    return torch.stack(losses, dim=0)


# ---- Design ----


def normalized_gradient_tensor(
    grad: torch.Tensor, gradient_mask: torch.Tensor
) -> torch.Tensor:
    masked_grad = grad * gradient_mask
    index_has_nonzero_grad = torch.square(masked_grad).sum(-1) > 0  # (B, L)
    eff_L = index_has_nonzero_grad.sum(-1)  # (B,)
    grad_norm = torch.linalg.norm(masked_grad, axis=(-1, -2))  # (B,)
    normalized_grad = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(
        eff_L[:, None, None]
    )
    return normalized_grad * gradient_mask


def design_binder(
    inversion_models: dict[str, ESMFold2ExperimentalModel],
    hf_critic_models: dict[str, ESMFold2ExperimentalModel],
    esmc_model: ESMCForMaskedLM,
    target_name: str | None,
    target_sequence: str | None,
    binder_name: str | None,
    binder_sequence: str | None,
    is_antibody: bool | None,
    seed: int,
    batch_size: int = 1,
) -> tuple[list[str], dict[int, dict[str, torch.Tensor]], list[dict]]:
    """
    Algorithm 11 Gradient-Guided Binder Sequence Optimization.

    Run the full optimization loop.
    Returns dict with designed_sequence, complex, and trajectory.

    Every critic is folded once on the best designed sequence via HF ESMFold2.
    Hero critics expose iPTM; scaling critics contribute distogram scores only.
    ``distogram_binding_confidence`` / ``cdr_distogram_binding_confidence`` come
    from the distogram in all cases.
    """
    # Vet inputs
    assert (target_name is None) ^ (
        target_sequence is None
    ), "Provide either target name or sequence."
    assert (binder_name is None) ^ (
        binder_sequence is None
    ), "Provide either binder name or sequence."

    # Setup
    device = "cuda"
    if target_name is not None:
        target_sequence = TARGET_SEQUENCES[target_name]
    else:
        assert target_sequence is not None
    target_one_hot = sequence_to_one_hot(target_sequence, device=device)

    if binder_name is None:
        assert binder_sequence is not None
        # If no binder_name and is_antibody is not specified, assume False.
        if is_antibody is None:
            is_antibody = False
    else:
        binder_prompt_factor = BINDER_PROMPT_FACTORIES[binder_name]
        if is_antibody is not None:
            assert (
                binder_prompt_factor.is_antibody == is_antibody
            ), "Conflict in is_antibody settings."
        is_antibody = binder_prompt_factor.is_antibody
        binder_sequence = binder_prompt_factor.sample(seed=seed)

    binder_length = len(binder_sequence)

    # By default, we only support single binder and target chains.
    # To support this case, remove the asserts below and check that losses
    # and selection metrics are appropriate for your multi-chain case.
    assert "|" not in target_sequence
    assert "|" not in binder_sequence

    with seed_context(seed), torch.device(device):
        logits = build_initial_soft_sequence_logits(
            binder_sequence, batch_size=batch_size
        )
        gradient_mask = build_gradient_mask(binder_sequence, batch_size=batch_size)

    # step -> {loss_name: [B] tensor on CPU}
    trajectory: dict[int, dict[str, torch.Tensor]] = {}
    global_step = 0

    def run_step(
        logits: torch.Tensor,
        optimizer: optim.Optimizer,
        temperature: float,
        calculate_confidence: bool,
    ) -> tuple[torch.Tensor, list[str], list[float] | None]:
        nonlocal global_step
        optimizer.zero_grad()

        random.seed(seed + global_step)
        replicate_choice = random.randint(0, len(inversion_models) - 1)
        inversion_model = list(inversion_models.values())[replicate_choice]
        design = F.softmax(logits / temperature, dim=-1)

        fold_result = fold_and_get_distogram(
            inversion_model,
            target_sequence,
            target_one_hot,
            design,
            num_loops=1,
            num_sampling_steps=50 if calculate_confidence else 1,
            calculate_confidence=calculate_confidence,
            seed=seed + global_step,
        )
        sequences: list[str] = fold_result["seq_list"]
        losses = compute_structure_losses(
            fold_result["distogram_logits"], binder_length
        )
        structure_loss = losses["total_loss"]
        structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]

        # Recompute the logits -> design transform for a fresh graph.
        design = F.softmax(logits / temperature, dim=-1)
        score_mask = gradient_mask.sum(dim=-1) > 0
        with seed_context(seed + global_step):
            plm_loss = compute_esmc_pseudoperplexity_nll(
                esmc_model=esmc_model,
                binder_design=design,
                score_mask=score_mask,
                batch_size=4,
                n_passes=4,
            )
        plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]

        logits.grad = normalized_gradient_tensor(structure_grad, gradient_mask) + (
            0.05 if is_antibody else 0.15
        ) * normalized_gradient_tensor(plm_grad, gradient_mask)

        for g in optimizer.param_groups:
            g["lr"] = LEARNING_RATE * temperature

        optimizer.step()

        step = global_step
        step_losses = {k: v.detach().cpu() for k, v in losses.items()}
        step_losses["plm_loss"] = plm_loss.detach().cpu()
        step_losses["total_loss"] = (structure_loss + plm_loss).detach().cpu()
        trajectory[step] = step_losses
        loss_str = "  ".join(
            f"{k}={v.mean().item():.4f}" for k, v in step_losses.items()
        )
        if step % LOG_INTERVAL == 0:
            logger.info(f"  step {step:3d}  |  {loss_str}  T={temperature:.4f}")
        global_step += 1
        return logits, sequences, fold_result.get("iptm", None)

    # Optimize
    optimizer = optim.SGD([logits], lr=LEARNING_RATE)
    best_iptm: list[float] = [-1.0] * batch_size
    best_sequences: list[str] = [""] * batch_size
    for step in range(STEPS):
        # Cosine schedule
        t = (step + 1) / STEPS
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMPERATURE_MIN + (1 - TEMPERATURE_MIN) * remaining
        logits, sequences, iptm = run_step(
            logits,
            optimizer,
            temperature=temperature,
            calculate_confidence=temperature < 0.05,
        )
        if iptm is not None:
            for b in range(batch_size):
                if iptm[b] is not None and iptm[b] > best_iptm[b]:
                    best_iptm[b] = iptm[b]
                    best_sequences[b] = sequences[b]
    assert all(seq != "" for seq in best_sequences)

    # Score
    critic_results: list[dict] = []
    target_length = len(target_sequence.replace("|", ""))
    for batch_idx in range(batch_size):
        best_seq = best_sequences[batch_idx]
        binder_seq = best_seq.split("|")[-1]
        binder_design = sequence_to_one_hot(binder_seq)[..., 2:22]
        for critic_name, critic_model in hf_critic_models.items():
            is_scaling_critic = "ESMFold2-Experimental-Fast-base" in critic_name
            if is_scaling_critic:
                critic_model.cuda()
            final_fold = fold_and_get_distogram(
                critic_model,
                target_sequence,
                target_one_hot,
                binder_design,
                num_loops=3,
                num_sampling_steps=200,
                calculate_confidence=True,
                seed=seed,
            )
            if is_scaling_critic:
                critic_model.cpu()
            pred_complex = build_complex(final_fold["inputs"], final_fold["output"])
            iptm_proxy_scores = compute_distogram_iptm_proxy(
                final_fold["distogram_logits"], target_length, binder_seq, is_antibody
            )
            iptm = final_fold["iptm"].item() if final_fold["iptm"] is not None else None
            critic_results.append(
                {
                    "is_antibody": is_antibody,
                    "critic_name": critic_name,
                    "batch_idx": batch_idx,
                    "designed_sequence": best_seq,
                    "complex": pred_complex,
                    "final_loss": trajectory[global_step - 1]["total_loss"][
                        batch_idx
                    ].item(),
                    "iptm": iptm,
                    "logits": logits[batch_idx].detach().cpu(),
                    **iptm_proxy_scores,
                }
            )

    if not critic_results:
        for batch_idx in range(batch_size):
            critic_results.append(
                {
                    "is_antibody": is_antibody,
                    "batch_idx": batch_idx,
                    "designed_sequence": best_sequences[batch_idx],
                    "final_loss": trajectory[global_step - 1]["total_loss"][
                        batch_idx
                    ].item(),
                    "logits": logits[batch_idx].detach().cpu(),
                }
            )

    return best_sequences, trajectory, critic_results


# ---- Model Loading ----

_ESMC = None


def _load_hf_model(
    critic_name: str, lm_dropout: float, cache_esmc: bool, device: str
) -> Any:
    """Loads ESMFold2 from huggingface.  Will cache ESMC-6B among
    all non-scaling checkpoints, to save on VRAM and load time."""
    global _ESMC
    repo_id = f"biohub/{critic_name}"
    model = ESMFold2ExperimentalModel.from_pretrained(repo_id, load_esmc=not cache_esmc)
    if cache_esmc:
        if _ESMC is None:
            model.load_esmc(model.config.esmc_id)
            _ESMC = model._esmc
        else:
            model._esmc = _ESMC
    model.configure_lm_dropout(lm_dropout, force_lm_dropout_during_inference=True)
    model.set_kernel_backend("cuequivariance" if CUE_AVAILABLE else None)
    return model.to(device=device).eval().requires_grad_(False)


def _apply_torch_compile(model: torch.nn.Module) -> None:
    """A helper for torch compiling the model."""
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 512

    compile_targets = (ESMFold2MSAEncoder, PairUpdateBlock, TransformerBlock)

    def _maybe_compile_module(module: torch.nn.Module) -> None:
        if not isinstance(module, compile_targets):
            return
        module.forward = torch.compile(module.forward)  # pyright: ignore

    model.apply(_maybe_compile_module)


class ESMFold2Design:
    lm_name = "biohub/ESMC-6B"
    inversion_model_names: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
    ]
    hero_critic_hf_paths: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
        "ESMFold2-Experimental",
        "ESMFold2-Experimental-Cutoff2025",
    ]
    scaling_critic_hf_paths: list[str] = []

    def load(self, use_scaling_critics: bool):
        if use_scaling_critics:
            self.scaling_critic_hf_paths = [
                f"ESMFold2-Experimental-Fast-base{size}-step{step}k"
                for size in ("300M", "600M", "6B")
                for step in ("250", "500", "750", "1000", "1500")
            ]

        self.inversion_models = {
            model_name: _load_hf_model(
                model_name, lm_dropout=0.5, cache_esmc=True, device="cuda"
            )
            for model_name in self.inversion_model_names
        }
        if COMPILE:
            for model in self.inversion_models.values():
                _apply_torch_compile(model)

        self.hf_critic_models: dict[str, Any] = {}
        for name in self.hero_critic_hf_paths:
            self.hf_critic_models[name] = _load_hf_model(
                name, lm_dropout=0.25, cache_esmc=True, device="cuda"
            )
        for name in self.scaling_critic_hf_paths:
            self.hf_critic_models[name] = _load_hf_model(
                name, lm_dropout=0.25, cache_esmc=False, device="cpu"
            )

        self.esmc_model = ESMCForMaskedLM.from_pretrained(
            self.lm_name, torch_dtype=torch.float32
        )
        if REUSE_ESMC:
            del self.esmc_model.esmc
            torch.cuda.empty_cache()
            self.esmc_model.esmc = self.inversion_models[
                "ESMFold2-Experimental-Fast"
            ]._esmc
        self.esmc_model = self.esmc_model.cuda().eval().requires_grad_(False)

        if CHECKPOINT_LM:
            apply_activation_checkpointing(
                self.esmc_model,
                checkpoint_wrapper_fn=partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                check_fn=lambda module: isinstance(module, TransformerBlock),
            )

    def design(
        self,
        target_name: str | None = None,
        target_sequence: str | None = None,
        binder_name: str | None = None,
        binder_sequence: str | None = None,
        is_antibody: bool | None = None,
        seed: int = 0,
        batch_size: int = 1,
    ) -> tuple[list[str], dict[int, dict[str, torch.Tensor]], list[dict]]:
        return design_binder(
            self.inversion_models,
            self.hf_critic_models,
            self.esmc_model,
            target_name=target_name,
            target_sequence=target_sequence,
            binder_name=binder_name,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            seed=seed,
            batch_size=batch_size,
        )


# ---- Modal ----


def get_base_image():
    return (
        modal.Image.micromamba(python_version="3.12")
        .run_commands("apt update && apt install -y git build-essential")
        .micromamba_install(
            "anarci>=2020.04.03", "hmmer=3.4", channels=["conda-forge", "bioconda"]
        )
        .pip_install("abnumber", "esm@git+https://github.com/Biohub/esm.git@main")
        .env({"HF_HOME": "/models", "HF_XET_HIGH_PERFORMANCE": "1"})
    )


app = modal.App(
    name="esmfold2-design",
    image=get_base_image(),
    volumes={
        "/models": modal.Volume.from_name("esmfold2-models", create_if_missing=True)
    },
)


# If use_scaling_checkpoints is True, `memory` should be increased to 60 * 1024.
@app.cls(gpu="H100", timeout=60 * 60, cpu=16, memory=10 * 1024)
class ESMFold2DesignModal(ESMFold2Design):
    """Modal entrypoint. Hero critics are HF experimental exports with
    confidence heads. Set ``use_scaling_critics=True`` to also load the
    15-checkpoint scaling-experiment ensemble (distogram binding confidence only).
    """

    use_scaling_critics: bool = modal.parameter(default=False)

    @modal.enter()
    def load(self):
        return super().load(self.use_scaling_critics)

    @modal.method()
    def design(self, *args, **kws):
        return super().design(*args, **kws)


@app.local_entrypoint()
def main(
    target_name: str | None = None,
    target_sequence: str | None = None,
    binder_name: str | None = None,
    binder_sequence: str | None = None,
    use_scaling_critics: bool = False,
    is_antibody: bool | None = None,
    local: bool = False,
    seed: int = 0,
    batch_size: int = 1,
):
    if local:
        assert not use_scaling_critics, (
            "'abnumber' will fail if running this script with uv run. "
            "It requires conda packages.  To be addressed soon."
        )
        app = ESMFold2Design()
        app.load(use_scaling_critics)
        run_fn = app.design
    else:
        app = ESMFold2DesignModal(use_scaling_critics=use_scaling_critics)
        run_fn = app.design.remote

    seq, trajectory, results = run_fn(
        target_name=target_name,
        target_sequence=target_sequence,
        binder_name=binder_name,
        binder_sequence=binder_sequence,
        is_antibody=is_antibody,
        seed=seed,
        batch_size=batch_size,
    )

    avg_final_loss = sum(r["final_loss"] for r in results) / len(results)
    logger.info(f"\nDesigned sequence: {seq}")
    logger.info(f"Trajectory length: {len(trajectory)} steps")
    logger.info(f"Average final loss: {avg_final_loss:.4f}")


if __name__ == "__main__":
    # Run a single local design.
    main(
        # Example case 1
        target_name="pd-l1",
        binder_name="minibinder",
        is_antibody=False,
        # Example case 2
        # target_name="cd45",
        # binder_name="trastuzumab_framework_vhvl",
        # is_antibody=True,
        # Common settings
        seed=0,
        batch_size=1,
        local=True,
        use_scaling_critics=False,
    )
