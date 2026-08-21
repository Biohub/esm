"""ESMFold2 sampler invariances.

These are the ESMFold2 analogue of ESMC's
``test_real_sequences_score_better_than_random_ones``: properties a stored
reference cannot express, because they relate two *different* forwards rather
than pinning one. Everything here runs on the tiny randomly-initialised model,
on CPU, with ``num_loops=1``, ``num_sampling_steps=2``,
``num_diffusion_samples=1`` unless a test needs otherwise.

Two facts about a randomly-initialised model shape every threshold below, and
both were measured rather than assumed:

1. The sampler is noise-dominated. Two seeds give Kabsch-aligned RMSDs of
   5.6-6.4 A on a 12-mer whose radius of gyration is 13.4 A, i.e. two samples
   are essentially unrelated structures. Any test phrased as "agrees within the
   sample spread" therefore carries a bound of ~6 A and can only catch gross
   breakage; where a sharper observable exists it is used instead, and where it
   does not the test asserts a deliberately-misaligned control blows past the
   bound so the bound is at least known not to pass anything.
2. ``distogram_logits`` is computed from the trunk *before* the sampler runs, so
   it is free of sampler noise and is 2-3 orders of magnitude more sensitive to
   the conditioning than the coordinates are. Perturbing ``entity_id`` moves the
   coordinates by 3e-4 A - invisible next to a 6 A spread - and the distogram by
   1.3e-1. It is the observable that gives the identity-tensor tests teeth.

``DiffusionStructureHead.sample`` draws fresh noise on every call, so
``torch.manual_seed`` is reset immediately before every forward that is meant to
be comparable, as ``test_kernel_backends_agree`` does. ``set_kernel_backend(None)``
and ``set_chunk_size(None)`` are pinned so a kernel or chunking difference cannot
be mistaken for a broken invariance.
"""

import pytest
import torch

from esm.models.esmfold2 import layers as _layers
from esm.models.esmfold2.layers import DiffusionStructureHead
from tests.conftest import (
    ESMFOLD2_COMPLEXES,
    ESMFOLD2_SEQUENCES,
    esmfold2_complex_features,
    esmfold2_inputs,
)

#: Fast-tier sampler settings, matching the rest of the ESMFold2 suite.
FAST_SAMPLER = dict(num_loops=1, num_diffusion_samples=1, num_sampling_steps=2)


@pytest.fixture
def sampler_model(tiny_esmfold2, monkeypatch):
    """The tiny model with every accelerator path pinned off.

    A kernel backend or a chunk-size difference would show up as a coordinate
    difference and be indistinguishable from a broken invariance, so both are
    fixed here rather than left at whatever the environment provides.
    """
    if not torch.cuda.is_available():
        monkeypatch.setattr(_layers, "FLASH_ATTN_AVAILABLE", False)
    tiny_esmfold2.set_kernel_backend(None)
    tiny_esmfold2.set_chunk_size(None)
    return tiny_esmfold2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _complex_inputs(model, chains: list[str] | tuple[str, ...]):
    """Batched multi-chain features plus synthetic LM states."""
    features, lm_hidden_states, _ = esmfold2_complex_features(model, list(chains))
    return features, lm_hidden_states


def _fold(model, features: dict, lm_hidden_states, seed: int, **overrides) -> dict:
    """One seeded forward.

    The seed is set immediately before the call because the diffusion sampler
    draws its initial noise and its per-step augmentation from the global RNG.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(
            **features,
            lm_hidden_states=lm_hidden_states,
            **{**FAST_SAMPLER, **overrides},
        )


def _rmsd(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """Un-aligned RMSD over the masked atoms of ``[1, A, 3]`` coordinates."""
    delta = (x - y) * mask[..., None]
    return float((delta.pow(2).sum() / mask.sum()).sqrt())


def _kabsch_rmsd(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """RMSD after superposing ``x`` onto ``y``.

    Uses the sampler's own Kabsch routine rather than a private copy, so the
    test measures with the same aligner the model uses internally to align
    consecutive denoising iterates.
    """
    weights = mask.float()
    aligned = DiffusionStructureHead._weighted_rigid_align(
        x.float(), y.float(), weights, weights
    )
    return _rmsd(aligned, y, mask)


def _atom_indices_by_chain(features: dict) -> list[torch.Tensor]:
    """Valid-atom indices for each chain, in ``asym_id`` order."""
    asym_id = features["asym_id"][0]
    valid = features["atom_attention_mask"][0].bool()
    atom_asym = asym_id[features["atom_to_token"][0]]
    return [
        torch.nonzero(valid & (atom_asym == chain), as_tuple=True)[0]
        for chain in asym_id.unique().tolist()
    ]


def _all_true(n: int) -> torch.Tensor:
    return torch.ones(1, n, dtype=torch.bool)


# ---------------------------------------------------------------------------
# The sample spread is a rigid motion plus a conformational difference
# ---------------------------------------------------------------------------


def test_kabsch_recovers_an_applied_rigid_motion(sampler_model):
    """The measurement itself, before it is used to judge the model.

    A rotated and translated copy of a prediction must come back with zero
    aligned RMSD and a large raw one. Without this, ``test_seeds_differ_...``
    below could pass because the aligner collapses everything rather than
    because the two structures really are related by a rigid motion.
    """
    features, lm_hidden_states = esmfold2_inputs(
        sampler_model, ESMFOLD2_SEQUENCES["tiny"]
    )
    mask = features["atom_attention_mask"].bool()
    coords = _fold(sampler_model, features, lm_hidden_states, seed=0)[
        "sample_atom_coords"
    ]

    # A fixed rotation (a unit quaternion) and a 13 A translation.
    q = torch.tensor([0.3, -0.5, 0.7, 0.41])
    r, i, j, k = q / q.norm()
    rotation = torch.tensor(
        [
            [1 - 2 * (j * j + k * k), 2 * (i * j - k * r), 2 * (i * k + j * r)],
            [2 * (i * j + k * r), 1 - 2 * (i * i + k * k), 2 * (j * k - i * r)],
            [2 * (i * k - j * r), 2 * (j * k + i * r), 1 - 2 * (i * i + j * j)],
        ]
    )
    moved = coords @ rotation.T + torch.tensor([[[7.0, -3.0, 11.0]]])

    # Measured: raw 24.32 A, aligned 2.9e-6 A. The bound is 1e-4, ~30x the
    # observed residual and ~5 orders of magnitude below the raw distance.
    assert _rmsd(coords, moved, mask) > 10.0
    assert _kabsch_rmsd(moved, coords, mask) < 1e-4


def test_seeds_differ_mostly_by_a_rigid_motion(sampler_model):
    """Changing only the sampler seed must move the structure mostly rigidly.

    The input carries no coordinates, so the only thing that fixes the global
    frame is the sampler's random augmentation, which is redrawn per seed. A
    model whose output frame were pinned by something else - a leaked reference
    orientation, an augmentation that is not applied to every atom - would show
    a raw RMSD no larger than the aligned one.
    """
    features, lm_hidden_states = esmfold2_inputs(
        sampler_model, ESMFOLD2_SEQUENCES["tiny"]
    )
    mask = features["atom_attention_mask"].bool()
    coords = [
        _fold(sampler_model, features, lm_hidden_states, seed=s)["sample_atom_coords"]
        for s in range(5)
    ]

    # Reseeding to the same value must reproduce the sample exactly, or none of
    # the comparisons below mean anything.
    repeat = _fold(sampler_model, features, lm_hidden_states, seed=0)
    assert torch.equal(repeat["sample_atom_coords"], coords[0])

    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    raw = [_rmsd(coords[i], coords[j], mask) for i, j in pairs]
    aligned = [_kabsch_rmsd(coords[i], coords[j], mask) for i, j in pairs]

    # Measured over the 10 pairs of seeds 0-4 on the 12-mer (94 valid atoms,
    # radius of gyration 13.35 A):
    #   raw      12.84 - 23.67 A   (mean 18.95)
    #   aligned   5.65 -  6.38 A   (mean  6.01)
    #   per-pair raw/aligned ratio >= 2.03
    # The bounds are 2.0 on the worst ratio (the observed minimum, rounded down
    # by 1.5%) and 7.0 A on the worst aligned RMSD (1.10x the observed maximum).
    # A rigid-invariance regression drives the ratio to 1; a sampler that stopped
    # converging at all drives the aligned RMSD past the radius of gyration.
    assert min(r / a for r, a in zip(raw, aligned)) >= 2.0
    assert max(aligned) < 7.0
    assert min(raw) > 10.0


# ---------------------------------------------------------------------------
# Homomer chain permutation: asym_id / sym_id / entity_id handling
# ---------------------------------------------------------------------------


def test_homodimer_chain_swap_reproduces_the_complex(sampler_model, ccd_pickle):
    """Swapping the two identical chains must reproduce the same complex.

    Two statements, because only the first is exact.

    The featurizer's part is exact: for a homodimer the two input orderings must
    produce bit-identical tensors, since ``entity_id`` is keyed on chemistry and
    ``asym_id``/``sym_id`` on position. An implementation that keyed entity on
    the chain *label* would break here. The heterodimer is the control - its
    tensors do differ under reversal, in 14 keys - so the equality is not the
    trivial consequence of a featurizer that ignores order.

    The model's part is only a spread-level statement. Exchange symmetry is not
    a property of the architecture: ``sym_id`` enters relpos through the signed
    difference ``sym_i - sym_j``, so the A->B and B->A pair blocks land in
    different bins and the predicted complex is *not* required to be symmetric
    under the swap. Measured on the trunk with no LM states, the distogram's
    exchange asymmetry is 2.9e-1 against a scale of 3.8, i.e. 7.6%. So the
    coordinate check asks only that the swap stays inside the sample spread.
    """
    chains = ESMFOLD2_COMPLEXES["homodimer"]
    features, lm_hidden_states = _complex_inputs(sampler_model, chains)
    reversed_features, _ = _complex_inputs(sampler_model, chains[::-1])
    assert all(
        torch.equal(features[k], reversed_features[k])
        for k in features
        if isinstance(features[k], torch.Tensor)
    )

    hetero = ESMFOLD2_COMPLEXES["heterodimer"]
    hetero_features, _ = _complex_inputs(sampler_model, hetero)
    hetero_reversed, _ = _complex_inputs(sampler_model, hetero[::-1])
    assert any(
        not torch.equal(hetero_features[k], hetero_reversed[k])
        for k in hetero_features
        if isinstance(hetero_features[k], torch.Tensor)
    )

    # The homodimer's identity tensors, spelled out: one entity, two asym ids,
    # two sym ids. Pinned here because everything below is a statement about
    # them.
    assert features["entity_id"][0].unique().tolist() == [0]
    assert features["asym_id"][0].unique().tolist() == [0, 1]
    assert features["sym_id"][0].unique().tolist() == [0, 1]

    chain_a, chain_b = _atom_indices_by_chain(features)
    assert len(chain_a) == len(chain_b) == 187
    n_tokens = features["asym_id"].shape[1]
    n_a = int((features["asym_id"][0] == 0).sum())
    token_swap = torch.cat([torch.arange(n_a, n_tokens), torch.arange(n_a)])

    # The swapped fold: identical features (asserted above), LM states exchanged
    # between the two chains, same seed. Its chain 0 is the original chain 1.
    swapped = _fold(sampler_model, features, lm_hidden_states[:, token_swap], seed=0)[
        "sample_atom_coords"
    ]
    original = _fold(sampler_model, features, lm_hidden_states, seed=0)[
        "sample_atom_coords"
    ]

    keep = torch.cat([chain_a, chain_b])
    relabelled = torch.cat([chain_b, chain_a])
    both_chains = _all_true(len(keep))
    swap_rmsd = _kabsch_rmsd(original[:, keep], swapped[:, relabelled], both_chains)

    spread = [
        _kabsch_rmsd(
            original[:, keep],
            _fold(sampler_model, features, lm_hidden_states, seed=s)[
                "sample_atom_coords"
            ][:, keep],
            both_chains,
        )
        for s in (1, 2, 3)
    ]

    # Control: the same comparison with each chain's atoms reversed, which
    # destroys the atom correspondence while leaving the point cloud intact.
    control = _kabsch_rmsd(
        original[:, keep],
        swapped[:, torch.cat([chain_b.flip(0), chain_a.flip(0)])],
        both_chains,
    )

    # Measured on the tiny model, 24+24 residues, 187+187 atoms:
    #   swap after relabelling          6.38 A
    #   seed-to-seed spread (seeds 1-3) 6.06 - 6.39 A
    #   reversed-atom control          15.9 - 21.5 A
    # The bound is 8.0 A: 1.25x the worst measured value. It cannot certify
    # exchange symmetry - see the docstring - but it does catch either chain
    # being treated differently from the other, which is what a per-chain
    # identity bug produces. It is blind to a *symmetric* atom-order shift,
    # since both folds get the same one; that case is
    # test_chains_fold_the_same_together_and_apart's.
    #
    # The control only has to show the metric can tell the orderings apart, so
    # its floor is 1.5x the bound, not 2x. The control's size depends on
    # per-residue atom ordering and therefore on the CCD release: 21.5 A on
    # ccd_20251126 but 15.9 A on ccd_20260516, so a 2x floor passed on one
    # mounted pickle and failed on another.
    assert swap_rmsd < 8.0
    assert max(spread) < 8.0
    assert control > 1.5 * 8.0


def test_chain_identity_tensors_reach_the_prediction(sampler_model, ccd_pickle):
    """``asym_id``/``sym_id``/``entity_id`` must not be dead inputs.

    Without this, ``test_homodimer_chain_swap_reproduces_the_complex`` would
    pass on a model that ignored all three: a tensor nobody reads is trivially
    handled correctly. Asserted on ``distogram_logits`` because it is the last
    output computed before the sampler adds noise; the same perturbations move
    the coordinates by only 3e-4 to 1.1e-3 A, which a 6 A sample spread hides.
    """
    features, lm_hidden_states = _complex_inputs(
        sampler_model, ESMFOLD2_COMPLEXES["homodimer"]
    )
    reference = _fold(sampler_model, features, lm_hidden_states, seed=0)[
        "distogram_logits"
    ]
    second_chain = features["asym_id"][0] == 1

    broken_entity = features["entity_id"].clone()
    broken_entity[0][second_chain] = 9
    broken_sym = features["sym_id"].clone()
    broken_sym[0] = 0
    broken_asym = features["asym_id"].clone()
    broken_asym[0] = 0

    # Measured max|delta distogram_logits| against a logit scale of 4.50:
    #   entity_id: the second chain made a distinct entity  1.32e-1
    #   sym_id:    both copies given sym_id 0               2.22e-1
    #   asym_id:   both chains merged into one              5.12e-1
    # The floor is 1e-2, ~13x below the smallest of the three, and ~450x above
    # the 0.0 that a tensor the model never reads would produce.
    for name, override in (
        ("entity_id", {"entity_id": broken_entity}),
        ("sym_id", {"sym_id": broken_sym}),
        ("asym_id", {"asym_id": broken_asym}),
    ):
        perturbed = _fold(
            sampler_model, {**features, **override}, lm_hidden_states, seed=0
        )["distogram_logits"]
        assert float((perturbed - reference).abs().max()) > 1e-2, name


# ---------------------------------------------------------------------------
# Residue_index translation invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tiny", "sub_window"])
@pytest.mark.parametrize("offset", [1, 37, 1000])
def test_residue_index_offset_changes_nothing(sampler_model, name, offset):
    """Relpos is differential, so a constant shift must be a no-op — exactly.

    ``ResIdxAsymIdSymIdEntityIdEncoding`` consumes ``residue_index`` only through
    ``residue_index[i] - residue_index[j]`` and ``residue_index[i] ==
    residue_index[j]``, and nothing else in the model reads it, so this is
    bit-equality rather than a tolerance. Measured max|delta| = 0.0 on
    coordinates, ``distogram_logits``, ``plddt`` and ``pae`` at every offset and
    both lengths; anything above 0 means a positional term crept in.
    """
    features, lm_hidden_states = esmfold2_inputs(
        sampler_model, ESMFOLD2_SEQUENCES[name]
    )
    reference = _fold(sampler_model, features, lm_hidden_states, seed=0)
    shifted = _fold(
        sampler_model,
        {**features, "residue_index": features["residue_index"] + offset},
        lm_hidden_states,
        seed=0,
    )
    for key in ("sample_atom_coords", "distogram_logits", "plddt", "pae"):
        assert torch.equal(shifted[key], reference[key]), key


@pytest.mark.parametrize("name", ["tiny", "sub_window"])
def test_residue_index_gap_changes_the_prediction(sampler_model, name):
    """The other direction, without which the test above is vacuous.

    A model that dropped ``residue_index`` entirely would pass
    ``test_residue_index_offset_changes_nothing`` perfectly. Inserting a gap
    inside the chain changes the *differences*, so it must change the output.
    """
    features, lm_hidden_states = esmfold2_inputs(
        sampler_model, ESMFOLD2_SEQUENCES[name]
    )
    reference = _fold(sampler_model, features, lm_hidden_states, seed=0)

    gapped = features["residue_index"].clone()
    n_tokens = gapped.shape[1]
    gapped[0, n_tokens // 2 :] += 5

    perturbed = _fold(
        sampler_model, {**features, "residue_index": gapped}, lm_hidden_states, seed=0
    )

    # Measured for a +5 gap at the midpoint, against 0.0 for the offset case:
    #   L=12   max|d distogram| 1.60e-1   max|d coords| 1.92e-3
    #   L=96   max|d distogram| 1.84e-1   max|d coords| 5.16e-4
    # Floors are 1e-2 on the distogram (~16x below the smaller measurement) and
    # 1e-4 on the coordinates (~5x below).
    assert (
        float(
            (perturbed["distogram_logits"] - reference["distogram_logits"]).abs().max()
        )
        > 1e-2
    )
    assert (
        float(
            (perturbed["sample_atom_coords"] - reference["sample_atom_coords"])
            .abs()
            .max()
        )
        > 1e-4
    )


# ---------------------------------------------------------------------------
# Chain independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chain", [0, 1])
def test_chains_fold_the_same_together_and_apart(sampler_model, chain, ccd_pickle):
    """Each chain's internal structure must survive being folded in a complex.

    A coarse bound by construction: on a randomly-initialised model the two
    chains do not interact in any meaningful sense, so what this really pins is
    that the per-chain slice of a multi-chain forward is the same *residues* in
    the same *order* as the single-chain forward. That is exactly what a wrong
    ``atom_to_token`` offset or a chain block written into the wrong atom range
    would break, and the reversed-atom control below shows the bound separates
    those cases by a factor of 2.7.

    Real chain independence - two non-binding chains predicted identically alone
    and together - is a property of trained weights and belongs with the
    published-weight tier.
    """
    chains = ESMFOLD2_COMPLEXES["heterodimer"]
    complex_features, complex_lm = _complex_inputs(sampler_model, chains)
    alone_features, alone_lm = _complex_inputs(sampler_model, [chains[chain]])

    in_complex = _atom_indices_by_chain(complex_features)[chain]
    alone = _atom_indices_by_chain(alone_features)[0]
    assert len(in_complex) == len(alone)
    mask = _all_true(len(alone))

    together = [
        _fold(sampler_model, complex_features, complex_lm, seed=s)["sample_atom_coords"]
        for s in range(4)
    ]
    apart = [
        _fold(sampler_model, alone_features, alone_lm, seed=s)["sample_atom_coords"]
        for s in range(4)
    ]

    spread = [
        _kabsch_rmsd(apart[i][:, alone], apart[j][:, alone], mask)
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    cross = [
        _kabsch_rmsd(t[:, in_complex], a[:, alone], mask)
        for t in together
        for a in apart
    ]
    control = [
        _kabsch_rmsd(t[:, in_complex], a[:, alone.flip(0)], mask)
        for t in together
        for a in apart
    ]

    # Measured on the heterodimer (24 + 18 residues, 187 + 138 atoms), seeds 0-3:
    #   chain 0  alone-vs-alone 5.77 - 6.24   together-vs-alone 5.38 - 6.39
    #   chain 1  alone-vs-alone 5.76 - 6.26   together-vs-alone 5.38 - 6.51
    #   reversed-atom control                 15.8 - 17.7
    # The bound is 8.0 A: 1.23x the worst together-vs-alone value. The control
    # floor is 1.5x the bound for the same reason as
    # test_homodimer_chain_swap_reproduces_the_complex: it scales with
    # per-residue atom ordering, so it moves with the mounted CCD release
    # (15.77 on ccd.pkl, 15.83 on ccd_20260516, 16.2+ on ccd_20251126).
    assert max(spread) < 8.0
    assert max(cross) < 8.0
    assert min(control) > 1.5 * 8.0


# ---------------------------------------------------------------------------
# Step-count convergence
# ---------------------------------------------------------------------------


def test_sampling_steps_converge(sampler_model):
    """Successive step counts must move the structure less, on average.

    The ODE is being integrated more finely, so ``RMSD(20, 68)`` should be
    smaller than ``RMSD(8, 20)``. A schedule bug - an off-by-one in the Karras
    grid, a ``max_inference_sigma`` truncation that changes the effective step
    count - leaves a single step count looking perfectly healthy and shows up
    only here.

    Asserted on the mean over eight seeds, not per seed: on a randomly-
    initialised model the sampler is noise-dominated and one seed in eight
    (seed 2: 3.52 then 3.80 A) goes the other way. Cheap enough to stay in the
    fast tier - all 24 forwards take ~2 s on CPU, so no ``nightly`` marker.
    """
    features, lm_hidden_states = esmfold2_inputs(
        sampler_model, ESMFOLD2_SEQUENCES["tiny"]
    )
    mask = features["atom_attention_mask"].bool()

    coarse, fine = [], []
    for seed in range(8):
        coords = {
            steps: _fold(
                sampler_model,
                features,
                lm_hidden_states,
                seed=seed,
                num_sampling_steps=steps,
            )["sample_atom_coords"]
            for steps in (8, 20, 68)
        }
        coarse.append(_kabsch_rmsd(coords[8], coords[20], mask))
        fine.append(_kabsch_rmsd(coords[20], coords[68], mask))

    # Measured over seeds 0-7 on the 12-mer:
    #   mean RMSD(8, 20)  6.513 A
    #   mean RMSD(20, 68) 4.145 A     ratio 0.636
    #   7 of 8 seeds decrease individually
    # Bounds: ratio < 0.85 (1.34x headroom over the measurement, and a schedule
    # that does not converge sits at or above 1.0) and at least 6 of 8 seeds
    # decreasing. Extending to 16 seeds gives ratio 0.787 and 11/16, so both
    # bounds hold with the seed set doubled.
    assert sum(fine) / sum(coarse) < 0.85
    assert sum(f < c for c, f in zip(coarse, fine)) >= 6


# ---------------------------------------------------------------------------
# Loop-count behaviour
# ---------------------------------------------------------------------------


def test_loop_count_changes_the_output_and_the_recurrence_stays_bounded(sampler_model):
    """``num_loops`` must reach the output, and the parcae recurrence must not drift.

    Deliberately *not* a pLDDT monotonicity test. On randomly-initialised
    weights the confidence head is uniform over its bins, so mean pLDDT is
    exactly 0.500 at 1, 4 and 20 loops and there is no signal to be monotone in.
    The published-weight tier owns "more loops predict better"; what is testable
    here is that the loop count is wired through at all and that iterating the
    linear recurrence twenty times does not blow up or collapse - the failure
    mode a mis-signed or mis-scaled ``_discretized_dynamics`` produces.
    """
    features, lm_hidden_states = esmfold2_inputs(
        sampler_model, ESMFOLD2_SEQUENCES["tiny"]
    )
    mask = features["atom_attention_mask"].bool()
    outputs = {
        loops: _fold(sampler_model, features, lm_hidden_states, seed=0, num_loops=loops)
        for loops in (1, 4, 20)
    }

    # Measured: consecutive loop counts differ by 6.10 and 6.06 A aligned, and
    # by 2.68 and 3.22 in max|distogram_logits|. The floors - 1.0 A and 0.5 -
    # are ~6x inside those, and a num_loops argument that never reached the
    # trunk would give exactly 0.0 for both.
    for first, second in ((1, 4), (4, 20)):
        assert (
            _kabsch_rmsd(
                outputs[first]["sample_atom_coords"],
                outputs[second]["sample_atom_coords"],
                mask,
            )
            > 1.0
        ), (first, second)
        assert (
            float(
                (
                    outputs[first]["distogram_logits"]
                    - outputs[second]["distogram_logits"]
                )
                .abs()
                .max()
            )
            > 0.5
        ), (first, second)

    reference = outputs[1]
    reference_coord_rms = float(
        reference["sample_atom_coords"][mask].pow(2).mean().sqrt()
    )
    reference_logit_max = float(reference["distogram_logits"].abs().max())

    # Stability. Measured across 1 / 4 / 20 loops (and 50, which is not asserted
    # but was checked): coordinate RMS 7.75 / 7.66 / 7.82 A, max|distogram|
    # 3.92 / 4.99 / 4.82, mean pLDDT 0.500 throughout. The bounds allow a 2x
    # growth in either quantity and a 2x collapse in the coordinate scale - wide
    # enough that no measured value is near them, narrow enough that a recurrence
    # with a spectral radius above 1 fails within twenty iterations.
    for loops, output in outputs.items():
        coords = output["sample_atom_coords"]
        assert torch.isfinite(coords).all(), loops
        assert torch.isfinite(output["distogram_logits"]).all(), loops
        coord_rms = float(coords[mask].pow(2).mean().sqrt())
        assert 0.5 * reference_coord_rms < coord_rms < 2.0 * reference_coord_rms, loops
        assert (
            float(output["distogram_logits"].abs().max()) < 2.0 * reference_logit_max
        ), loops
        assert abs(float(output["plddt"].mean()) - 0.5) < 0.05, loops
