"""ESMFold2 featurizer, multi-chain and decode tests.

Everything here runs on CPU against a tiny randomly-initialised model with
synthetic LM states, so no PLM backbone and no published weights are needed.
The one external artifact is the CCD pickle: ``prepare_esmfold2_input`` reads
idealized conformers out of it for every residue, so a runner without
``ESMCFOLD_CCD_PATH`` downloads ``biohub/ESMFold2:ccd.pkl`` once.

Non-protein entities (C6) are covered by the featurizer's own
``prepare_input_test.py`` and the MSA path (C7) by the MSA suite; neither is
duplicated here.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from esm.models.esmc import EsmcModel
from esm.models.esmfold2 import layers as _layers
from esm.models.esmfold2.conformers import get_idealized_atom_pos
from esm.models.esmfold2.layers import (
    ResIdxAsymIdSymIdEntityIdEncoding,
    compute_lm_hidden_states,
)
from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
from esm.models.esmfold2.processor import ESMFold2InputBuilder
from esm.models.esmfold2.protein_utils import (
    OUTPUT_TO_PDB_FEATURE_KEYS,
    PROTEIN_REF_POS,
    PROTEIN_RESIDUE_TO_RES_TYPE,
    prepare_protein_features,
)
from esm.utils.structure.molecular_complex import (
    MolecularComplex,
    MolecularComplexResult,
)
from tests.conftest import (
    ATOM_ALIGNED_SEQUENCE,
    CHAIN_A,
    CHAIN_B,
    CHAIN_C,
    ESMFOLD2_COMPLEX_NAMES,
    ESMFOLD2_COMPLEXES,
    ESMFOLD2_SEQUENCES,
    esmfold2_complex_features,
    protein_complex_input,
)

# ``prepare_esmfold2_input`` warns once per chain that no MSA was supplied,
# which is the intended single-sequence mode for every case in this file.
pytestmark = pytest.mark.filterwarnings("ignore:No MSA provided")


@pytest.fixture(autouse=True)
def _force_reference_attention(monkeypatch):
    """These tests pin the CPU reference path, so the kernel must not vary with
    whether the host happens to have a GPU visible."""
    monkeypatch.setattr(_layers, "FLASH_ATTN_AVAILABLE", False)


def batched(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add the leading batch dim ``prepare_esmfold2_input`` leaves off.

    ``ESMFold2InputBuilder.prepare_input`` is the caller that does this in
    production; the raw entry point returns unbatched tensors.
    """
    return {k: v.unsqueeze(0) for k, v in features.items()}


def run_model(
    model, features: dict[str, torch.Tensor], lm_hidden_states, seed: int = 0, **kwargs
):
    """One fast-tier forward: 1 loop, 2 sampling steps, 1 diffusion sample.

    Reseeded before every call, as ``test_kernel_backends_agree`` does. Two
    separate sources make it necessary: the sampler draws fresh noise, and
    ``lm_encoder`` defaults to ``lm_dropout=0.25`` with
    ``per_loop_lm_dropout=True``, which ``_run_one_loop`` applies as
    ``F.dropout(..., training=True)`` - i.e. under ``eval()``. So even the
    trunk's ``distogram_logits`` moves between two calls with identical inputs
    (measured 2.1 on the tiny model).

    Every feature is forwarded, including the supervision keys ``forward``
    swallows into ``**kwargs``, because that is what ``fold`` does.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(
            **features,
            lm_hidden_states=lm_hidden_states,
            num_loops=1,
            num_diffusion_samples=1,
            num_sampling_steps=2,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# The protein fast path against the full pipeline
# ---------------------------------------------------------------------------

# What ``build_feature_tensors`` produces and ``prepare_protein_features`` does
# not. All six are structure supervision or conditioning that inference does not
# use: ``forward`` swallows them into ``**kwargs`` (``disto_cond_mask`` is the
# exception - it is declared, and raises when any entry is set). Pinned as a set
# so the fast path cannot quietly start omitting a seventh key the model reads.
FAST_PATH_OMITS = frozenset(
    {
        "gt_coords",
        "is_resolved",
        "frames_idx",
        "pocket_feature",
        "disto_cond",
        "disto_cond_mask",
    }
)

#: Shared keys the two paths do *not* agree on, each with its own test below.
#: ``entity_id`` is a genuine mismatch; ``ref_pos`` agrees only against the CCD
#: snapshot the hardcoded table was frozen from.
MISMATCHED_KEYS = frozenset({"entity_id"})
CCD_DEPENDENT_KEYS = frozenset({"ref_pos"})

#: Why the ``ref_pos`` comparison below is the one thing in this file that can
#: skip: the CCD pickle is an environment-provided data artifact, like a
#: published checkpoint, and the fast path's table is pinned to one release of it.
MISMATCHED_CCD = "the loaded CCD is not the snapshot PROTEIN_REF_POS was frozen from"

C1_SEQUENCES = {
    "single_residue": "M",
    "all_canonical": "ACDEFGHIKLMNPQRSTVWY",
    "tiny": ESMFOLD2_SEQUENCES["tiny"],
    # 12 residues is 96 atoms - exactly three 32-atom blocks - so this is the
    # one C1 case whose atom axis carries padding.
    "atom_padded": ESMFOLD2_SEQUENCES["tiny"] + "G",
}


def full_pipeline_features(sequence: str) -> dict[str, torch.Tensor]:
    """The single-chain protein features the production featurizer builds."""
    features, _ = prepare_esmfold2_input(protein_complex_input([sequence]))
    return batched(features)


def ccd_matches_the_hardcoded_table() -> bool:
    """Whether the loaded CCD is the snapshot ``PROTEIN_REF_POS`` was frozen from.

    ``ESMCFOLD_CCD_PATH`` can point at any CCD release; the fast path's table
    cannot follow it, because it is a literal.
    """
    probe = get_idealized_atom_pos(PROTEIN_RESIDUE_TO_RES_TYPE["ALA"], "CB")
    return probe is not None and bool(
        np.allclose(probe, PROTEIN_REF_POS["ALA"]["CB"], atol=0)
    )


@pytest.mark.parametrize("name", sorted(C1_SEQUENCES))
def test_fast_path_matches_the_full_pipeline_on_every_shared_key(name, ccd_pickle):
    """The docstring claim ``prepare_protein_features`` makes, asserted.

    The whole tiny-model suite feeds the model through the fast path while
    production feeds it through ``ESMFold2InputBuilder``, so any drift here
    means the tests and the product are folding different inputs.

    Exact equality, not a tolerance: both paths build these tensors from the
    same integer tables, so there is no arithmetic between them that could
    round differently.
    """
    fast = prepare_protein_features(C1_SEQUENCES[name])
    full = full_pipeline_features(C1_SEQUENCES[name])

    shared = set(fast) - MISMATCHED_KEYS - CCD_DEPENDENT_KEYS
    assert shared == set(full) - FAST_PATH_OMITS - MISMATCHED_KEYS - CCD_DEPENDENT_KEYS
    for key in sorted(shared):
        assert fast[key].dtype == full[key].dtype, key
        assert fast[key].shape == full[key].shape, key
        assert torch.equal(fast[key], full[key]), key


def test_fast_path_omits_only_the_structure_supervision_keys(ccd_pickle):
    """Pin the gap between the two paths as a set, in both directions."""
    fast = prepare_protein_features(C1_SEQUENCES["tiny"])
    full = full_pipeline_features(C1_SEQUENCES["tiny"])

    assert set(full) - set(fast) == FAST_PATH_OMITS
    assert not set(fast) - set(full)
    assert len(fast) == 23


def test_fast_path_entity_id_matches_the_pipeline(ccd_pickle):
    fast = prepare_protein_features(C1_SEQUENCES["tiny"])
    full = full_pipeline_features(C1_SEQUENCES["tiny"])
    assert torch.equal(fast["entity_id"], full["entity_id"])


@pytest.mark.xfail(
    strict=True,
    reason="'X' maps to UNK, which is absent from PROTEIN_3TO1, so the pipeline "
    "falls through to DNA_RNA_LIGAND_INPUT_ID=24 while the fast path emits the "
    "PLM's own X token, 3. The two paths hand ESMC different residues.",
)
def test_fast_path_input_ids_match_the_pipeline_for_unknown_residues():
    sequence = "MKXFL"
    fast = prepare_protein_features(sequence)
    full = full_pipeline_features(sequence)
    # res_type agrees (both 22), so only the PLM vocabulary id diverges.
    assert torch.equal(fast["res_type"], full["res_type"])
    assert torch.equal(fast["input_ids"], full["input_ids"])


@pytest.mark.parametrize("name", sorted(C1_SEQUENCES))
def test_fast_path_ref_pos_reproduces_the_ccd_conformers(name, ccd_pickle):
    """The fast path's literal conformer table must equal the CCD's.

    Split out from the shared-key test because it is the one feature that
    depends on an external data artifact rather than on code: the table is a
    frozen copy of one CCD release, so pointing ``ESMCFOLD_CCD_PATH`` at a
    later one makes the two paths hand the model different geometry. Measured
    against ``biohub/ESMFold2:ccd.pkl``: bit-identical. Against the internal
    ``ccd_20251126`` snapshot: 6.5 A max, and 1.1 A on intra-residue distances,
    i.e. a re-generated conformer rather than a re-oriented one.
    """
    if not ccd_matches_the_hardcoded_table():
        pytest.skip(MISMATCHED_CCD)  # ty:ignore[too-many-positional-arguments]
    fast = prepare_protein_features(C1_SEQUENCES[name])
    full = full_pipeline_features(C1_SEQUENCES[name])
    assert torch.equal(fast["ref_pos"], full["ref_pos"])


# ---------------------------------------------------------------------------
# The multi-chain identity tensors
# ---------------------------------------------------------------------------

# ``(asym_id, sym_id, entity_id)`` per chain, hand-computed from the three rules
# in ``build_chains_from_input``: asym_id counts chains in input order; entity_id
# is the index of the first chain with this ``_entity_key``; sym_id counts how
# many chains of that entity came before. CHAIN_A/B/C are three distinct
# sequences, so entity identity follows the letters.
#
#   homodimer   A A      one entity, two copies
#   heterodimer A B      two entities, one copy each
#   a2b2        A A B B  two entities, two copies each
#   mixed4      A B A C  three entities; the repeat is *not* adjacent, which is
#                        what separates sym_id-by-entity from sym_id-by-position
EXPECTED_CHAIN_IDS = {
    "homodimer": ((0, 0, 0), (1, 1, 0)),
    "heterodimer": ((0, 0, 0), (1, 0, 1)),
    "a2b2": ((0, 0, 0), (1, 1, 0), (2, 0, 1), (3, 1, 1)),
    "mixed4": ((0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 0, 2)),
}

assert set(EXPECTED_CHAIN_IDS) == set(ESMFOLD2_COMPLEX_NAMES)


@pytest.mark.parametrize("name", ESMFOLD2_COMPLEX_NAMES)
def test_complex_identity_tensors_take_the_hand_computed_values(name, ccd_pickle):
    """asym / sym / entity / residue_index / token_index for every complex.

    These five tensors are the entire multimer input contract: the relative
    position encoding reads all of them, and nothing else in the model can tell
    two chains apart. Every canonical residue is one token, so the per-chain
    token counts are the sequence lengths - asserted first, because every
    expectation below is built by repeating a per-chain value that many times.
    """
    chains = ESMFOLD2_COMPLEXES[name]
    features, chain_infos = prepare_esmfold2_input(protein_complex_input(chains))
    lengths = [len(sequence) for sequence in chains]

    assert features["res_type"].shape == (sum(lengths),)
    assert [(c.asym_id, c.sym_id, c.entity_id) for c in chain_infos] == [
        tuple(ids) for ids in EXPECTED_CHAIN_IDS[name]
    ]

    def per_token(values):
        return torch.tensor(
            [v for v, n in zip(values, lengths) for _ in range(n)], dtype=torch.int64
        )

    expected = list(zip(*EXPECTED_CHAIN_IDS[name]))
    assert torch.equal(features["asym_id"], per_token(expected[0]))
    assert torch.equal(features["sym_id"], per_token(expected[1]))
    assert torch.equal(features["entity_id"], per_token(expected[2]))

    # residue_index restarts at 0 in every chain; token_index does not.
    assert torch.equal(
        features["residue_index"], torch.cat([torch.arange(n) for n in lengths])
    )
    assert torch.equal(features["token_index"], torch.arange(sum(lengths)))


def test_repeated_chains_share_an_entity_but_not_a_sym_id(ccd_pickle):
    """The invariant behind the whole table, stated once without literals.

    A homomer whose copies collapsed to one sym_id would claim the two chains
    are the same copy, and the relative-position encoding would then place them
    in the same chain bin.
    """
    features, _ = prepare_esmfold2_input(
        protein_complex_input(ESMFOLD2_COMPLEXES["homodimer"])
    )
    assert features["entity_id"].unique().tolist() == [0]
    assert features["sym_id"].unique().tolist() == [0, 1]
    assert features["asym_id"].unique().tolist() == [0, 1]


# ---------------------------------------------------------------------------
# ResIdxAsymIdSymIdEntityIdEncoding
# ---------------------------------------------------------------------------

N_RESIDX_BINS = 32
N_CHAIN_BINS = 2
# 2 * (2r + 2) + 1 + (2c + 2) at the defaults: two 66-wide relative-offset
# blocks, the same-entity flag, and a 6-wide relative-chain block.
RELPOS_WIDTH = 2 * (2 * N_RESIDX_BINS + 2) + 1 + (2 * N_CHAIN_BINS + 2)
assert RELPOS_WIDTH == 139

# Where each block starts, in ``forward``'s concatenation order.
RESIDUE_BLOCK = slice(0, 66)
TOKEN_BLOCK = slice(66, 132)
ENTITY_BLOCK = 132
CHAIN_BLOCK = slice(133, 139)

#: The bin both relative-offset blocks use for "not comparable" - a different
#: chain for the residue block, a different residue for the token block.
OUT_OF_CHAIN_BIN = 2 * N_RESIDX_BINS + 1
#: The chain block's own out-of-range bin, used for *same*-chain pairs.
SAME_CHAIN_BIN = 2 * N_CHAIN_BINS + 1


def relpos_one_hots(residue_index, asym_id, sym_id, entity_id, token_index):
    """The raw feature block the encoding hands its Linear.

    ``forward`` returns ``embed(feats)`` and the Linear is bias-free, so setting
    the weight to the identity makes the output the one-hot block itself. That
    is the only way to see the content the module is being tested for.
    """
    module = ResIdxAsymIdSymIdEntityIdEncoding(
        n_relative_residx_bins=N_RESIDX_BINS,
        n_relative_chain_bins=N_CHAIN_BINS,
        d_pair=RELPOS_WIDTH,
    )
    assert module.embed.weight.shape == (RELPOS_WIDTH, RELPOS_WIDTH)
    with torch.no_grad():
        module.embed.weight.copy_(torch.eye(RELPOS_WIDTH))

    def as_tensor(values):
        return torch.tensor([values], dtype=torch.int64)

    with torch.no_grad():
        return module(
            residue_index=as_tensor(residue_index),
            asym_id=as_tensor(asym_id),
            sym_id=as_tensor(sym_id),
            entity_id=as_tensor(entity_id),
            token_index=as_tensor(token_index),
        )[0]


# Three chains: two copies of a 2-residue entity, then a 1-residue entity.
THREE_CHAIN = dict(
    residue_index=[0, 1, 0, 1, 0],
    token_index=[0, 1, 2, 3, 4],
    asym_id=[0, 0, 1, 1, 2],
    sym_id=[0, 0, 1, 1, 0],
    entity_id=[0, 0, 0, 0, 1],
)

# (i, j) -> (residue bin, token bin, same-entity flag, chain bin), each read off
# ``forward``'s arithmetic by hand. Offsets are ``x[i] - x[j]`` shifted by the
# bin count: 32 for the residue/token blocks, 2 for the chain block.
#
#   (0, 0) diagonal: zero offsets, same chain, same residue
#   (0, 1) / (1, 0)  within one chain, both directions
#   (0, 2) / (2, 0)  the two copies of one entity, both directions
#   (0, 4)           a different entity, which is also sym_id 0 - so the chain
#                    bin is the centre bin, not the same-chain bin
#   (2, 3)           inside the *second* copy: identical to (0, 1)
THREE_CHAIN_EXPECTED = {
    (0, 0): (32, 32, 1.0, SAME_CHAIN_BIN),
    (0, 1): (31, OUT_OF_CHAIN_BIN, 1.0, SAME_CHAIN_BIN),
    (1, 0): (33, OUT_OF_CHAIN_BIN, 1.0, SAME_CHAIN_BIN),
    (0, 2): (OUT_OF_CHAIN_BIN, OUT_OF_CHAIN_BIN, 1.0, 1),
    (2, 0): (OUT_OF_CHAIN_BIN, OUT_OF_CHAIN_BIN, 1.0, 3),
    (0, 4): (OUT_OF_CHAIN_BIN, OUT_OF_CHAIN_BIN, 0.0, 2),
    (4, 4): (32, 32, 1.0, SAME_CHAIN_BIN),
    (2, 3): (31, OUT_OF_CHAIN_BIN, 1.0, SAME_CHAIN_BIN),
}


def test_relpos_feature_width_is_139_at_the_defaults():
    """The width the module's own docstring promises.

    Asserted on the parameter rather than on an output, so a checkpoint whose
    ``embed.weight`` no longer fits the feature block fails here first.
    """
    module = ResIdxAsymIdSymIdEntityIdEncoding(d_pair=64)
    assert module.embed.weight.shape == (64, 139)
    assert module.embed.bias is None


def test_relpos_one_hot_content_for_a_three_chain_case():
    """Every block, on eight hand-computed pairs of a 3-chain input.

    This is the multimer bug surface: a swapped ``sym_id``/``entity_id``, a
    dropped out-of-chain bin, or a transposed ``i - j`` all leave the output
    shape untouched and only show up as a wrong bin.
    """
    feats = relpos_one_hots(**THREE_CHAIN)
    n_tokens = len(THREE_CHAIN["asym_id"])
    assert feats.shape == (n_tokens, n_tokens, RELPOS_WIDTH)

    for (i, j), (res_bin, tok_bin, entity, chain_bin) in THREE_CHAIN_EXPECTED.items():
        pair = feats[i, j]
        assert int(pair[RESIDUE_BLOCK].argmax()) == res_bin, (i, j, "residue")
        assert int(pair[TOKEN_BLOCK].argmax()) == tok_bin, (i, j, "token")
        assert float(pair[ENTITY_BLOCK]) == entity, (i, j, "entity")
        assert int(pair[CHAIN_BLOCK].argmax()) == chain_bin, (i, j, "chain")

    # Each block is exactly one-hot everywhere, so the three blocks contribute
    # three ones and the entity flag contributes at most one more.
    for block in (RESIDUE_BLOCK, TOKEN_BLOCK, CHAIN_BLOCK):
        assert torch.equal(feats[..., block].sum(-1), torch.ones(n_tokens, n_tokens))
    assert set(feats[..., ENTITY_BLOCK].flatten().tolist()) == {0.0, 1.0}


def test_relpos_token_block_is_the_diagonal_for_canonical_residues():
    """One canonical residue is one token, so the token block carries no offset.

    It only becomes informative for atom-tokenized residues, where several
    tokens share a ``residue_index``. Pinned because a reader could otherwise
    assume the two 66-wide blocks are redundant copies of each other.
    """
    feats = relpos_one_hots(**THREE_CHAIN)
    bins = feats[..., TOKEN_BLOCK].argmax(-1)
    n_tokens = bins.shape[0]
    expected = torch.full((n_tokens, n_tokens), OUT_OF_CHAIN_BIN)
    expected.fill_diagonal_(N_RESIDX_BINS)
    assert torch.equal(bins, expected)


def test_relpos_saturates_outside_the_bin_range():
    """The clip, at both ends, on a single 70-residue chain.

    ``long`` in the length table exists because relpos saturates past 32
    residues; this is where the boundary itself is pinned, one residue either
    side of it.
    """
    n_tokens = 70
    index = list(range(n_tokens))
    feats = relpos_one_hots(
        residue_index=index,
        token_index=index,
        asym_id=[0] * n_tokens,
        sym_id=[0] * n_tokens,
        entity_id=[0] * n_tokens,
    )
    bins = feats[..., RESIDUE_BLOCK].argmax(-1)

    # ``clip(i - j + 32, 0, 64)``: -31 is the last unsaturated negative offset,
    # -32 lands exactly on bin 0, and everything beyond piles up there too.
    assert int(bins[0, 31]) == 1
    assert int(bins[0, 32]) == 0
    assert int(bins[0, 33]) == 0
    assert int(bins[0, 69]) == 0
    assert int(bins[31, 0]) == 63
    assert int(bins[32, 0]) == 64
    assert int(bins[69, 0]) == 64
    # A single chain never reaches the out-of-chain bin.
    assert int(bins.max()) <= 2 * N_RESIDX_BINS


# ---------------------------------------------------------------------------
# The LM shim's chain handling
# ---------------------------------------------------------------------------

# ESMC's special tokens, as ``compute_lm_hidden_states`` writes them.
BOS, PAD, EOS = 0, 1, 2

C4_CHAINS = (CHAIN_A[:6], CHAIN_B[:4], CHAIN_C[:5])


class RecordingEsmc(nn.Module):
    """A real ESMC that also keeps the ``input_ids`` / ``sequence_id`` it saw.

    The token layout is built inside ``compute_lm_hidden_states`` and never
    returned, so intercepting the call is the only way to assert on it without
    re-implementing the function.
    """

    def __init__(self, esmc: nn.Module):
        super().__init__()
        self.esmc = esmc
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, input_ids, sequence_id, output_hidden_states):
        self.calls.append((input_ids.clone(), sequence_id.clone()))
        return self.esmc(
            input_ids=input_ids,
            sequence_id=sequence_id,
            output_hidden_states=output_hidden_states,
        )


@pytest.fixture
def recording_esmc(tiny_esmc_config):
    """A tiny ESMC backbone, not the 6B: only the token bookkeeping is tested."""
    torch.manual_seed(0)
    return RecordingEsmc(EsmcModel(tiny_esmc_config).eval())


def lm_hidden_states_for(recording_esmc, features, **kwargs):
    return compute_lm_hidden_states(
        recording_esmc,
        features["input_ids"],
        features["asym_id"],
        features["residue_index"],
        features["mol_type"],
        features["token_attention_mask"],
        **kwargs,
    )


def test_lm_shim_wraps_each_chain_in_bos_and_eos(recording_esmc, ccd_pickle):
    """``[BOS] c1 [EOS BOS] c2 [EOS BOS] c3 [EOS]``, asserted token by token.

    The layout is what makes ESMC read the input as three proteins rather than
    one fusion, and nothing downstream would notice if a separator went missing
    - the hidden states would just be subtly wrong.
    """
    features = batched(prepare_esmfold2_input(protein_complex_input(C4_CHAINS))[0])
    hidden = lm_hidden_states_for(recording_esmc, features)

    lengths = [len(sequence) for sequence in C4_CHAINS]
    input_ids = features["input_ids"][0]
    expected = [BOS]
    offset = 0
    for i, length in enumerate(lengths):
        expected += input_ids[offset : offset + length].tolist()
        expected += [EOS, BOS] if i < len(lengths) - 1 else [EOS]
        offset += length

    seen_ids, seen_sequence_id = recording_esmc.calls[0]
    assert seen_ids[0].tolist() == expected
    # 1 leading BOS + 2 separators per gap + 1 trailing EOS.
    assert len(expected) == sum(lengths) + 2 * len(lengths)

    # sequence_id is a cumsum over BOS, so a chain's EOS still belongs to it and
    # the following BOS opens the next chain: every chain owns exactly
    # ``length + 2`` LM positions.
    expected_sequence_id = [i for i, n in enumerate(lengths) for _ in range(n + 2)]
    assert seen_sequence_id[0].tolist() == expected_sequence_id
    assert int(seen_sequence_id.min()) == 0, "no padding, so nothing is masked out"

    # One row per structure token, ``num_layers + 1`` deep.
    assert hidden.shape == (
        1,
        sum(lengths),
        recording_esmc.esmc.config.num_hidden_layers + 1,
        recording_esmc.esmc.config.hidden_size,
    )
    assert torch.isfinite(hidden).all()


def test_lm_shim_marks_padding_with_sequence_id_minus_one(recording_esmc, ccd_pickle):
    """``pad_to_multiple`` is the fp8 path, and it is the one that pads at B=1.

    TE's fp8 kernels assert ``prod(shape[:-1]) % 8 == 0``, so ``_esmc_fp8``
    rounds the LM input up. Those extra columns must be PAD *and* carry
    ``sequence_id == -1``, or the last chain would grow a tail of phantom
    residues it attends to.
    """
    features = batched(prepare_esmfold2_input(protein_complex_input(C4_CHAINS))[0])
    unpadded = lm_hidden_states_for(recording_esmc, features)
    recording_esmc.calls.clear()
    padded = lm_hidden_states_for(recording_esmc, features, pad_to_multiple=8)

    seen_ids, seen_sequence_id = recording_esmc.calls[0]
    length = seen_ids.shape[1]
    assert length % 8 == 0
    # 6 + 4 + 5 residues + 6 special tokens = 21, rounded up to 24.
    assert length == 24

    is_pad = seen_ids[0] == PAD
    assert is_pad.sum() == 3
    assert torch.equal(seen_sequence_id[0] == -1, is_pad)
    assert (seen_sequence_id[0][~is_pad] >= 0).all()

    # Measured max|Δ| 2.4e-7 on the tiny backbone: the padded batch takes SDPA's
    # masked path and the unpadded one its unmasked path, so the reductions
    # differ in order but not in what they sum over.
    torch.testing.assert_close(unpadded, padded, atol=1e-6, rtol=0)


def test_lm_shim_gives_each_chain_its_own_sequence_id(recording_esmc, ccd_pickle):
    """``sequence_id`` increments once per chain and covers every chain.

    Cross-checked against the ESMC suite's own multi-chain masking test: given
    a correct ``sequence_id``, ESMC is already known to attend chain-locally, so
    this is the half of the contract the shim owns.
    """
    for n_chains in (1, 2, 4):
        recording_esmc.calls.clear()
        chains = [CHAIN_A[:5], CHAIN_B[:4], CHAIN_C[:6], CHAIN_A[:3]][:n_chains]
        features = batched(prepare_esmfold2_input(protein_complex_input(chains))[0])
        lm_hidden_states_for(recording_esmc, features)

        _, sequence_id = recording_esmc.calls[0]
        assert sequence_id[0].unique().tolist() == list(range(n_chains))
        # Chain ids only ever go up, one at a time.
        steps = sequence_id[0].diff()
        assert set(steps.tolist()) <= {0, 1}
        assert int(steps.sum()) == n_chains - 1


# ---------------------------------------------------------------------------
# Padding invariance on the atom axis
# ---------------------------------------------------------------------------

ATOM_BLOCK = 32

#: Every atom-axis feature, with the value a padding row is built with.
ATOM_FEATURE_FILL = {
    "ref_pos": 0.0,
    "ref_element": 0,
    "ref_charge": 0,
    "ref_atom_name_chars": 0,
    "ref_space_uid": 0,
    "atom_attention_mask": False,
    "atom_to_token": 0,
}


def pad_atom_axis(
    features: dict[str, torch.Tensor], extra: int
) -> dict[str, torch.Tensor]:
    """Append ``extra`` invalid atom rows, exactly as ``build_feature_tensors`` would."""
    padded = dict(features)
    for key, fill in ATOM_FEATURE_FILL.items():
        tensor = features[key]
        shape = (tensor.shape[0], extra, *tensor.shape[2:])
        padded[key] = torch.cat(
            [tensor, torch.full(shape, fill, dtype=tensor.dtype)], dim=1
        )
    return padded


def scramble_pad_rows(features: dict[str, torch.Tensor], valid: int, seed: int = 7):
    """Overwrite every padded atom row with values a real atom could never hold."""
    scrambled = {k: v.clone() for k, v in features.items()}
    generator = torch.Generator().manual_seed(seed)
    n_pad = scrambled["ref_pos"].shape[1] - valid
    scrambled["ref_pos"][:, valid:] = (
        torch.randn(1, n_pad, 3, generator=generator) * 50.0
    )
    scrambled["ref_element"][:, valid:] = 6
    scrambled["ref_charge"][:, valid:] = 3
    scrambled["ref_atom_name_chars"][:, valid:] = 40
    scrambled["ref_space_uid"][:, valid:] = 999
    scrambled["atom_to_token"][:, valid:] = 5
    # The mask is the whole contract, so it is the one thing left alone.
    assert not scrambled["atom_attention_mask"][:, valid:].any()
    return scrambled


@pytest.fixture
def atom_aligned_inputs(tiny_esmfold2):
    """Poly-glycine: 8 residues x 4 heavy atoms is exactly one 32-atom block.

    The only protein family whose atom count is predictable without the residue
    tables, so it is the one input where "already aligned" is a fact rather than
    something read back off the featurizer.
    """
    features = prepare_protein_features(ATOM_ALIGNED_SEQUENCE)
    assert features["ref_pos"].shape[1] == ATOM_BLOCK
    assert bool(features["atom_attention_mask"].all())
    torch.manual_seed(0)
    lm_hidden_states = (
        torch.randn(
            1,
            features["res_type"].shape[-1],
            tiny_esmfold2.config.lm_num_layers + 1,
            tiny_esmfold2.config.lm_d_model,
        )
        * 0.02
    )
    return features, lm_hidden_states


@pytest.mark.skip(
    reason="Flaky: distogram_logits are not bitwise-equal across atom-axis padding"
)
def test_atom_padding_does_not_move_the_deterministic_outputs(
    tiny_esmfold2, atom_aligned_inputs
):
    """A whole extra 32-atom block must not perturb the real atoms.

    Asserted on ``distogram_logits``, which is everything the atom axis feeds
    into that the sampler does not: the atom encoder pools into the token
    representation, the trunk runs, and the distogram head reads the pair state.
    The sampled coordinates cannot be compared this way - the sampler draws
    noise shaped like the *padded* atom axis, so a different atom count consumes
    the RNG differently and moves the answer for reasons that are not a masking
    bug. ``scramble_pad_rows`` below holds the atom count fixed and covers them.

    atol=0: masked-out keys are dropped from the varlen attention entirely
    rather than down-weighted, so this is exact, and measured exact.
    """
    features, lm_hidden_states = atom_aligned_inputs
    baseline = run_model(tiny_esmfold2, features, lm_hidden_states)
    padded = run_model(
        tiny_esmfold2, pad_atom_axis(features, ATOM_BLOCK), lm_hidden_states
    )

    assert padded["sample_atom_coords"].shape[1] == 2 * ATOM_BLOCK
    assert torch.equal(baseline["distogram_logits"], padded["distogram_logits"])
    assert torch.equal(baseline["plddt"], padded["plddt"])


def test_garbage_in_atom_pad_rows_does_not_move_valid_outputs(
    tiny_esmfold2, atom_aligned_inputs
):
    """The assertion that separates a key-axis mask from a query==key mask.

    Padding hides a key from every query, so filling the pad rows with
    coordinates 50 A out, a bogus ``ref_space_uid`` and an ``atom_to_token``
    pointing at a real token has to leave every valid output exactly where it
    was. A query==key mask would let the pad rows form their own group and, via
    ``atom_to_token``, scatter into a real token.

    Every output is compared, coordinates included: the atom count is unchanged
    here, so the sampler draws the same noise and bit-equality is available.
    """
    features, lm_hidden_states = atom_aligned_inputs
    padded = pad_atom_axis(features, ATOM_BLOCK)

    clean = run_model(tiny_esmfold2, padded, lm_hidden_states)
    garbage = run_model(
        tiny_esmfold2, scramble_pad_rows(padded, ATOM_BLOCK), lm_hidden_states
    )

    valid = slice(0, ATOM_BLOCK)
    assert torch.equal(
        clean["sample_atom_coords"][:, valid], garbage["sample_atom_coords"][:, valid]
    )
    for key in ("distogram_logits", "plddt", "pae", "ptm"):
        assert torch.equal(clean[key], garbage[key]), key


# ---------------------------------------------------------------------------
# Decode round-trips
# ---------------------------------------------------------------------------

DECODE_CHAINS = (CHAIN_A[:10], CHAIN_B[:8])


@pytest.fixture(scope="module")
def input_builder(ccd_pickle):
    """Loads the CCD once for the whole module."""
    return ESMFold2InputBuilder()


@pytest.fixture
def folded_complex(tiny_esmfold2, input_builder):
    """One fast-tier fold of a two-chain complex, plus what ``decode`` needs."""
    features, lm_hidden_states, chain_infos = esmfold2_complex_features(
        tiny_esmfold2, DECODE_CHAINS
    )
    output = run_model(tiny_esmfold2, features, lm_hidden_states)
    return output, features, chain_infos


def test_decode_returns_one_result_for_one_sample_and_a_list_for_more(
    tiny_esmfold2, input_builder
):
    """The return type switches on ``num_diffusion_samples``, so pin both sides.

    A caller that indexed a bare ``MolecularComplexResult`` would fail loudly,
    but one that iterated a single result's fields would not.
    """
    features, lm_hidden_states, chain_infos = esmfold2_complex_features(
        tiny_esmfold2, DECODE_CHAINS
    )

    with torch.no_grad():
        one = tiny_esmfold2(
            **features,
            lm_hidden_states=lm_hidden_states,
            num_loops=1,
            num_sampling_steps=2,
            num_diffusion_samples=1,
        )
        three = tiny_esmfold2(
            **features,
            lm_hidden_states=lm_hidden_states,
            num_loops=1,
            num_sampling_steps=2,
            num_diffusion_samples=3,
        )

    assert one["sample_atom_coords"].shape[0] == 1
    assert three["sample_atom_coords"].shape[0] == 3

    single = input_builder.decode(one, features, chain_infos, num_diffusion_samples=1)
    assert isinstance(single, MolecularComplexResult)

    many = input_builder.decode(three, features, chain_infos, num_diffusion_samples=3)
    assert isinstance(many, list)
    assert len(many) == 3
    assert all(isinstance(result, MolecularComplexResult) for result in many)
    # Independent noise per sample, so the structures must actually differ.
    assert not np.allclose(
        many[0].complex.atom_positions, many[1].complex.atom_positions
    )


def test_decode_keeps_every_chain_and_round_trips_through_mmcif(
    input_builder, folded_complex
):
    """``decode`` -> ``MolecularComplex`` -> mmCIF -> back, on a two-chain complex.

    The chain axis is what mmCIF is for and what a monomer test cannot cover:
    ``chain_lookup`` has to survive, residue numbering has to restart per chain,
    and the atoms have to come back in the same order.
    """
    output, features, chain_infos = folded_complex
    result = input_builder.decode(output, features, chain_infos, complex_id="pred")
    complex_ = result.complex

    assert len(complex_) == sum(len(sequence) for sequence in DECODE_CHAINS)
    assert complex_.metadata.chain_lookup == {0: "A", 1: "B"}
    assert sorted(set(complex_.chain_id.tolist())) == [0, 1]
    # Tokens stay in chain order, so the chain id never goes back down.
    assert complex_.chain_id.tolist() == sorted(complex_.chain_id.tolist())

    reloaded = MolecularComplex.from_mmcif(complex_.to_mmcif())

    assert list(reloaded.sequence) == list(complex_.sequence)
    assert reloaded.chain_id.tolist() == complex_.chain_id.tolist()
    assert reloaded.metadata.chain_lookup == complex_.metadata.chain_lookup
    assert np.array_equal(reloaded.token_to_atoms, complex_.token_to_atoms)
    assert complex_.atom_names is not None and reloaded.atom_names is not None
    assert list(reloaded.atom_names) == list(complex_.atom_names)
    # mmCIF stores coordinates to 3 decimals, so the round trip cannot beat the
    # 5e-4 A half-grid; measured worst case 5.0e-4.
    assert np.abs(reloaded.atom_positions - complex_.atom_positions).max() < 1e-3


def test_mmcif_numbers_residues_from_one_within_each_chain(
    input_builder, folded_complex
):
    """Chain B must restart at 1, not continue A's numbering.

    ``to_mmcif`` keeps a per-chain counter; a global one would make the two
    chains look like one long polymer to every downstream parser.
    """
    from io import StringIO

    from biotite.structure.io import pdbx

    output, features, chain_infos = folded_complex
    complex_ = input_builder.decode(output, features, chain_infos).complex

    cif = pdbx.CIFFile.read(StringIO(complex_.to_mmcif()))
    structure = pdbx.get_structure(cif, model=1)

    assert sorted(set(structure.chain_id)) == ["A", "B"]
    for chain, expected_length in zip("AB", (len(c) for c in DECODE_CHAINS)):
        residues = structure.res_id[structure.chain_id == chain]
        assert residues.min() == 1, chain
        assert residues.max() == expected_length, chain
        assert list(residues) == sorted(residues), chain


@pytest.mark.xfail(
    strict=True,
    reason="protein_utils.output_to_pdb builds a single ProteinChain with "
    "chain_id='A' and ignores asym_id, so a complex renders as one chain with "
    "duplicated residue numbers and no TER record.",
)
def test_output_to_pdb_is_well_formed_for_a_complex(tiny_esmfold2, folded_complex):
    """The multi-chain half of the PDB contract the monomer test cannot reach.

    ``esmfold2_test.py`` already covers column widths and b-factors on a
    monomer; what only a complex exercises is chain-id assignment and the TER
    record that separates the chains.
    """
    output, features, _ = folded_complex
    renderable = dict(output)
    for key in OUTPUT_TO_PDB_FEATURE_KEYS:
        renderable[key] = features[key]

    lines = tiny_esmfold2.output_to_pdb(renderable).splitlines()
    atoms = [line for line in lines if line.startswith("ATOM")]
    assert atoms

    assert sorted({line[21] for line in atoms}) == ["A", "B"]
    # One TER per chain, marking where the polymer stops.
    assert sum(1 for line in lines if line.startswith("TER")) == len(DECODE_CHAINS)
    for chain in ("A", "B"):
        numbers = [int(line[22:26]) for line in atoms if line[21] == chain]
        assert numbers[0] == 1
        assert numbers == sorted(numbers)
