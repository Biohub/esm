"""Tests for protein_chain.py"""

import numpy as np

from esm.utils.structure.protein_chain import ProteinChain


def _make_chain(sequence: str) -> ProteinChain:
    n = len(sequence)
    rng = np.random.default_rng(0)
    coords = rng.uniform(-10, 10, (n, 37, 3)).astype(np.float32)
    return ProteinChain(
        id="test",
        sequence=sequence,
        chain_id="A",
        entity_id=1,
        residue_index=np.arange(n),
        insertion_code=np.full(n, "", dtype="<U4"),
        atom37_positions=coords,
        atom37_mask=np.ones((n, 37), dtype=bool),
        confidence=np.ones(n),
    )


def test_infer_cbeta_does_not_mutate_cached_inferred_cbeta():
    # infer_cbeta() NaN-fills the glycine CB positions to build a *new* chain; it
    # must not mutate the original chain's cached `inferred_cbeta` array (nor
    # anything derived from it, such as pdist_CB).
    chain = _make_chain("AGA")  # glycine at index 1

    original = chain.inferred_cbeta.copy()
    original_pdist = chain.pdist_CB.copy()

    chain.infer_cbeta()

    np.testing.assert_array_equal(chain.inferred_cbeta, original)
    np.testing.assert_array_equal(chain.pdist_CB, original_pdist)
