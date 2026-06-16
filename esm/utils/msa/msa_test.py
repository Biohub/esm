"""Tests for MSA.from_a3m deletion handling (a3m lowercase insertions)."""

import gzip

import numpy as np

from esm.models.esmfold2.paired_msa import msa_to_res_type_and_deletions
from esm.utils.msa.msa import MSA, a3m_deletion_counts
from esm.utils.parsing import FastaEntry

# query has no insertions (5 match columns); row1 has "aa" inserted before col2,
# row2 has "c" inserted before col4.
_A3M = ">query\nMKLNT\n>s1 key=101\nMKaaLNT\n>s2 key=102\nM-LNcT\n"
_EXPECTED_DELETIONS = np.array(
    [[0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 1]], dtype=np.float32
)
_LETTER_TO_RES_TYPE = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def _write_a3m(path, gz: bool):
    if gz:
        with gzip.open(path, "wt") as f:
            f.write(_A3M)
    else:
        path.write_text(_A3M)


def _naive_deletion_counts(seq: str) -> list[int]:
    out, ins = [], 0
    for ch in seq:
        if ch == "." or ch.islower():
            ins += 1
        else:
            out.append(ins)
            ins = 0
    return out


def _a3m_msa(tmp_path) -> MSA:
    """Build the shared `_A3M` fixture as an MSA (insertion-stripped, deletions set)."""
    p = tmp_path / "m.a3m"
    _write_a3m(p, gz=False)
    return MSA.from_a3m(str(p))


def test_a3m_deletion_counts_vectorized():
    # leading "aa" before M, interior "xx" before L; trailing insertions are dropped
    np.testing.assert_array_equal(a3m_deletion_counts("aaMKxxLN"), [2, 0, 2, 0])
    for seq in ["MKLNT", "MKaaLNcT", ".aMK..LN", "MKLNdd"]:
        np.testing.assert_array_equal(
            a3m_deletion_counts(seq), _naive_deletion_counts(seq)
        )


def test_from_a3m_records_deletions(tmp_path):
    msa = _a3m_msa(tmp_path)  # remove_insertions=True (default)
    # stored sequences are insertion-stripped (equal length = query length)
    assert msa.sequences == ["MKLNT", "MKLNT", "M-LNT"]
    assert msa.deletions is not None
    np.testing.assert_array_equal(msa.deletions, _EXPECTED_DELETIONS)


def test_from_a3m_gz(tmp_path):
    p = tmp_path / "m.a3m.gz"
    _write_a3m(p, gz=True)
    msa = MSA.from_a3m(str(p))
    assert msa.deletions is not None
    np.testing.assert_array_equal(msa.deletions, _EXPECTED_DELETIONS)


def test_deletions_flow_through_featurization(tmp_path):
    """Stripped sequences would give zero deletions; stored ones must survive."""
    msa = _a3m_msa(tmp_path)
    _, deletions = msa_to_res_type_and_deletions(msa, _LETTER_TO_RES_TYPE)
    np.testing.assert_array_equal(deletions, _EXPECTED_DELETIONS)


def test_from_sequences_has_no_deletions_and_recomputes():
    """MSAs without stored deletions fall back to parsing the sequences."""
    no_del = MSA.from_sequences(["MKLNT", "MKLNT"])
    assert no_del.deletions is None
    _, deletions = msa_to_res_type_and_deletions(no_del, _LETTER_TO_RES_TYPE)
    np.testing.assert_array_equal(deletions, np.zeros((2, 5), dtype=np.float32))


def test_slicing_carries_deletions(tmp_path):
    """Row/column subselects carry the matching slice of deletions."""
    msa = _a3m_msa(tmp_path)
    by_row = msa.select_sequences([0, 2]).deletions
    by_col = msa.select_positions([2, 4]).deletions
    by_slice = msa[1:].deletions
    assert by_row is not None and by_col is not None and by_slice is not None
    np.testing.assert_array_equal(by_row, _EXPECTED_DELETIONS[[0, 2]])
    np.testing.assert_array_equal(by_col, _EXPECTED_DELETIONS[:, [2, 4]])
    np.testing.assert_array_equal(by_slice, _EXPECTED_DELETIONS[:, 1:])


def test_sliced_deletions_flow_through_featurization(tmp_path):
    """A per-chain column subselect (as chainbreak splitting does) keeps deletions,
    so featurizing the sliced MSA yields the sliced counts, not zeros."""
    sub = _a3m_msa(tmp_path).select_positions([2, 3, 4])
    _, deletions = msa_to_res_type_and_deletions(sub, _LETTER_TO_RES_TYPE)
    np.testing.assert_array_equal(deletions, _EXPECTED_DELETIONS[:, [2, 3, 4]])
    assert deletions.sum() > 0


def test_deletions_dropped_when_not_column_aligned():
    """If sequences keep insertions (length != match-column count), a column
    subselect drops deletions rather than misaligning them."""
    unstripped = MSA(
        entries=[FastaEntry("q", "MKaaLNT"), FastaEntry("s", "MKaaLNT")],
        deletions=np.zeros((2, 5), dtype=np.float32),
    )
    assert unstripped.select_positions([0, 1]).deletions is None


def test_pad_to_depth_carries_deletions(tmp_path):
    """Padded (all-gap) rows contribute zero deletions and keep the array aligned."""
    msa = _a3m_msa(tmp_path)
    padded = msa.pad_to_depth(5)
    assert padded.deletions is not None
    np.testing.assert_array_equal(padded.deletions[:3], _EXPECTED_DELETIONS)
    np.testing.assert_array_equal(padded.deletions[3:], np.zeros((2, 5), np.float32))


def test_stack_carries_deletions(tmp_path):
    """Row-stacking concatenates deletion rows, dropping the duplicated query row."""
    msa = _a3m_msa(tmp_path)
    stacked = MSA.stack([msa, msa], remove_query_from_later_msas=True)
    assert stacked.deletions is not None
    expected = np.concatenate([_EXPECTED_DELETIONS, _EXPECTED_DELETIONS[1:]], axis=0)
    np.testing.assert_array_equal(stacked.deletions, expected)


def test_stack_drops_deletions_if_any_missing(tmp_path):
    msa = _a3m_msa(tmp_path)
    no_del = MSA.from_sequences(["MKLNT", "MKLNT"])
    assert MSA.stack([msa, no_del]).deletions is None


def test_concat_carries_deletions_without_join_token(tmp_path):
    """Column-concatenation joins deletion columns when no join token is inserted."""
    msa = _a3m_msa(tmp_path)
    concatenated = MSA.concat([msa, msa], join_token=None)
    assert concatenated.deletions is not None
    np.testing.assert_array_equal(
        concatenated.deletions,
        np.concatenate([_EXPECTED_DELETIONS, _EXPECTED_DELETIONS], axis=1),
    )


def test_concat_drops_deletions_with_join_token(tmp_path):
    """A join token inserts columns with no deletion counterpart, so drop them."""
    msa = _a3m_msa(tmp_path)
    assert MSA.concat([msa, msa], join_token="|").deletions is None


def test_state_dict_round_trip(tmp_path):
    """state_dict/from_state_dict preserve sequences and deletions over the wire."""
    msa = _a3m_msa(tmp_path)
    payload = msa.state_dict(json_serializable=True)
    assert payload["sequences"] == msa.sequences
    assert isinstance(payload["deletions"], list)
    restored = MSA.from_state_dict(payload)
    assert restored.sequences == msa.sequences
    assert restored.deletions is not None
    np.testing.assert_array_equal(restored.deletions, _EXPECTED_DELETIONS)


def test_state_dict_omits_deletions_when_absent():
    """from_sequences MSAs carry no deletions, so none are serialized."""
    payload = MSA.from_sequences(["MKLNT", "MKLNT"]).state_dict(json_serializable=True)
    assert "deletions" not in payload
    assert MSA.from_state_dict(payload).deletions is None
