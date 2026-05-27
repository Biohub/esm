"""Taxonomy-paired MSA construction for ESMFold2 inference.

Taxonomy IDs are read from FASTA headers as ``key=N`` tokens. Rows
where any chain has ``key=-1`` (or no ``key=`` at all) are treated as
unpaired and assigned to that chain's block-diagonal section after
the paired rows.
"""

import re

import numpy as np

from esm.models.esmfold2.constants import (
    MSA_GAP_TOKEN_ID,
    PROTEIN_3TO1,
    PROTEIN_RESIDUE_TO_RES_TYPE,
    PROTEIN_UNK_RES_TYPE,
)
from esm.utils.msa.msa import MSA

_KEY_RE = re.compile(r"key=(-?\d+)")


def protein_letter_to_res_type() -> dict[str, int]:
    """Return the protein 1-letter → res_type mapping used by the MSA encoder."""
    mapping: dict[str, int] = {}
    for three, one in PROTEIN_3TO1.items():
        if three in PROTEIN_RESIDUE_TO_RES_TYPE:
            mapping[one] = PROTEIN_RESIDUE_TO_RES_TYPE[three]
    mapping["-"] = MSA_GAP_TOKEN_ID
    mapping["X"] = PROTEIN_UNK_RES_TYPE
    return mapping


def _taxonomy_from_header(header: str) -> int:
    if not header:
        return -1
    m = _KEY_RE.search(header)
    return int(m.group(1)) if m else -1


def msa_to_res_type_and_deletions(
    msa: MSA, letter_to_res_type: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an :class:`MSA` to ``(res_type[M, L], deletion_count[M, L])``.

    Handles a3m insertion convention: lowercase letters and ``.`` are
    insertions and are not emitted; their count is accumulated into the
    next non-insertion position's deletion value. ``L`` is the query
    length after stripping insertions from row 0.
    """
    query = msa.entries[0].sequence
    L = sum(1 for ch in query if not (ch.islower() or ch == "."))
    M = msa.depth

    res_type = np.full((M, L), MSA_GAP_TOKEN_ID, dtype=np.int64)
    deletions = np.zeros((M, L), dtype=np.float32)

    for r, entry in enumerate(msa.entries):
        col = 0
        ins = 0
        for ch in entry.sequence:
            if ch == "." or (ch.islower() and ch != "-"):
                ins += 1
                continue
            if col >= L:
                break
            if ch == "-":
                res_type[r, col] = MSA_GAP_TOKEN_ID
            else:
                res_type[r, col] = letter_to_res_type.get(
                    ch.upper(), PROTEIN_UNK_RES_TYPE
                )
            if ins > 0:
                deletions[r, col] = float(ins)
                ins = 0
            col += 1
    return res_type, deletions


def _dummy_msa_residues(query_res_types: np.ndarray) -> np.ndarray:
    """Single-row 'MSA' for chains without one — just the query."""
    return query_res_types[None, :]  # [1, L]


def construct_paired_msa(
    chain_msas: dict[int, MSA | None],
    chain_query_res_types: dict[int, np.ndarray],
    token_asym_ids: np.ndarray,
    token_res_ids: np.ndarray,
    letter_to_res_type: dict[str, int] | None = None,
    *,
    max_pairs: int = 8192,
    max_total: int = 16384,
    max_seqs: int = 16384,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build paired MSA features.

    Parameters
    ----------
    chain_msas
        ``asym_id -> MSA`` (or ``None`` for chains without an MSA).
    chain_query_res_types
        ``asym_id -> np.ndarray[L_c]`` of res-type ids for the chain's
        query. Used to build dummy MSAs when a chain has no MSA.
    token_asym_ids
        Per-token asym_id, length ``T``. Must be non-decreasing.
    token_res_ids
        Per-token residue index within chain, length ``T``.
    letter_to_res_type
        1-letter → res-type mapping. Defaults to
        :func:`protein_letter_to_res_type`.

    Returns
    -------
    msa_residues : ``np.ndarray[M, T]`` int64
    deletion_value : ``np.ndarray[M, T]`` float32 (raw deletion counts; the
        ``arctan(/3) * pi/2`` transform is applied by the caller)
    is_paired : ``np.ndarray[M, T]`` float32 broadcast of per-row,
        per-chain paired flags.
    """
    if letter_to_res_type is None:
        letter_to_res_type = protein_letter_to_res_type()

    chain_ids: list[int] = sorted(chain_msas.keys())

    # Build per-chain (res_type, deletions, taxonomy) tables.
    chain_res_type: dict[int, np.ndarray] = {}
    chain_deletions: dict[int, np.ndarray] = {}
    chain_taxonomies: dict[int, list[int]] = {}
    for c in chain_ids:
        m = chain_msas.get(c)
        if m is None or m.depth == 0:
            qres = chain_query_res_types[c]
            chain_res_type[c] = _dummy_msa_residues(qres)
            chain_deletions[c] = np.zeros((1, qres.shape[0]), dtype=np.float32)
            chain_taxonomies[c] = [-1]
            continue
        rt, dl = msa_to_res_type_and_deletions(m, letter_to_res_type)
        chain_res_type[c] = rt
        chain_deletions[c] = dl
        chain_taxonomies[c] = [_taxonomy_from_header(e.header) for e in m.entries]

    # Group by taxonomy, skip query row and unpaired (-1) entries.
    taxonomy_map: dict[int, list[tuple[int, int]]] = {}
    for c in chain_ids:
        for seq_idx, taxon in enumerate(chain_taxonomies[c]):
            if seq_idx == 0 or taxon == -1:
                continue
            taxonomy_map.setdefault(taxon, []).append((c, seq_idx))
    taxonomy_map = {k: v for k, v in taxonomy_map.items() if len(v) > 1}
    # Order taxonomies by number of distinct chains, descending.
    sorted_taxa = sorted(
        taxonomy_map.items(), key=lambda kv: len({c for c, _ in kv[1]}), reverse=True
    )

    visited = {s for _, items in taxonomy_map.items() for s in items}
    available: dict[int, list[int]] = {
        c: [i for i in range(1, len(chain_taxonomies[c])) if (c, i) not in visited]
        for c in chain_ids
    }

    pairing: list[dict[int, int]] = [{c: 0 for c in chain_ids}]
    is_paired: list[dict[int, int]] = [{c: 1 for c in chain_ids}]

    for _, pairs in sorted_taxa:
        per_chain: dict[int, list[int]] = {}
        for c, seq_idx in pairs:
            per_chain.setdefault(c, []).append(seq_idx)
        max_occ = max(len(v) for v in per_chain.values())
        for i in range(max_occ):
            row_pairing: dict[int, int] = {}
            row_is_paired: dict[int, int] = {}
            for c, seq_idxs in per_chain.items():
                row_pairing[c] = seq_idxs[i % len(seq_idxs)]
                row_is_paired[c] = 1
            for c in chain_ids:
                if c in row_pairing:
                    continue
                row_is_paired[c] = 0
                if available[c]:
                    row_pairing[c] = available[c].pop(0)
                else:
                    row_pairing[c] = -1
            pairing.append(row_pairing)
            is_paired.append(row_is_paired)
            if len(pairing) >= max_pairs:
                break
        if len(pairing) >= max_pairs:
            break

    max_left = max((len(v) for v in available.values()), default=0)
    for _ in range(min(max_total - len(pairing), max_left)):
        row_pairing = {}
        row_is_paired = {}
        for c in chain_ids:
            row_is_paired[c] = 0
            if available[c]:
                row_pairing[c] = available[c].pop(0)
            else:
                row_pairing[c] = -1
        pairing.append(row_pairing)
        is_paired.append(row_is_paired)
        if len(pairing) >= max_total:
            break

    pairing = pairing[:max_seqs]
    is_paired = is_paired[:max_seqs]
    M = len(pairing)
    T = len(token_asym_ids)

    msa_residues = np.full((M, T), MSA_GAP_TOKEN_ID, dtype=np.int64)
    deletion_value = np.zeros((M, T), dtype=np.float32)
    paired_mask = np.zeros((M, T), dtype=np.float32)

    # Vectorize per chain: gather chain rows according to pairing[c], then
    # index into them by the chain's token residue ids.
    for c in chain_ids:
        rt = chain_res_type[c]
        dl = chain_deletions[c]
        Lc = rt.shape[1]
        chain_pairing = np.array([row[c] for row in pairing], dtype=np.int64)
        chain_paired = np.array([row[c] for row in is_paired], dtype=np.float32)

        token_mask = token_asym_ids == c
        if not token_mask.any():
            continue
        token_res_in_chain = token_res_ids[token_mask]
        # Clamp residue indices to the MSA's column range. Modified-residue
        # tokens that exceed the query length fall back to the last column.
        cols = np.minimum(token_res_in_chain, Lc - 1)

        # Rows where pairing == -1 fall back to gap (already initialized).
        valid_rows = chain_pairing >= 0
        if valid_rows.any():
            gathered_rt = rt[chain_pairing[valid_rows]][:, cols]
            gathered_dl = dl[chain_pairing[valid_rows]][:, cols]
            valid_idx = np.where(valid_rows)[0]
            token_idx = np.where(token_mask)[0]
            msa_residues[np.ix_(valid_idx, token_idx)] = gathered_rt
            deletion_value[np.ix_(valid_idx, token_idx)] = gathered_dl

        paired_mask[:, token_mask] = chain_paired[:, None]

    return msa_residues, deletion_value, paired_mask
