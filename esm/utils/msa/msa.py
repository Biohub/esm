from __future__ import annotations

import dataclasses
import string
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from typing import Sequence

import numpy as np
from Bio import SeqIO
from scipy.spatial.distance import cdist

from esm.utils.misc import slice_any_object
from esm.utils.msa.filter_sequences import greedy_select_indices, hhfilter
from esm.utils.parsing import FastaEntry, read_sequences, write_sequences
from esm.utils.sequential_dataclass import SequentialDataclass
from esm.utils.system import PathOrBuffer

REMOVE_LOWERCASE_TRANSLATION = str.maketrans(dict.fromkeys(string.ascii_lowercase))


def remove_insertions_from_sequence(seq: str) -> str:
    return seq.translate(REMOVE_LOWERCASE_TRANSLATION)


@dataclass(frozen=True)
class MSA(SequentialDataclass):
    """Object-oriented interface to an MSA.

    Args:
        sequences (list[str]): List of protein sequences
        headers (list[str]): List of headers describing the sequences

    """

    entries: list[FastaEntry]

    @cached_property
    def sequences(self) -> list[str]:
        return [entry.sequence for entry in self.entries]

    @cached_property
    def headers(self) -> list[str]:
        return [entry.header for entry in self.entries]

    def __repr__(self):
        return (
            f"MSA({self.entries[0].header}: Depth={self.depth}, Length={self.seqlen})"
        )

    def to_fast_msa(self) -> FastMSA:
        return FastMSA(self.array, self.headers)

    @classmethod
    def from_a3m(
        cls,
        path: PathOrBuffer,
        remove_insertions: bool = True,
        max_sequences: int | None = None,
    ) -> MSA:
        entries = []
        for header, seq in islice(read_sequences(path), max_sequences):
            if remove_insertions:
                seq = remove_insertions_from_sequence(seq)
            if entries:
                assert (
                    len(seq) == len(entries[0].sequence)
                ), f"Sequence length mismatch. Expected: {len(entries[0].sequence)}, Received: {len(seq)}"
            entries.append(FastaEntry(header, seq))
        return cls(entries)

    def to_a3m(self, path: PathOrBuffer) -> None:
        write_sequences(self.entries, path)

    @classmethod
    def from_stockholm(
        cls,
        path: PathOrBuffer,
        remove_insertions: bool = True,
        max_sequences: int | None = None,
    ) -> MSA:
        entries = []
        for record in islice(SeqIO.parse(path, "stockholm"), max_sequences):
            header = f"{record.id} {record.description}"
            seq = str(record.seq)
            if entries:
                assert (
                    len(seq) == len(entries[0].sequence)
                ), f"Sequence length mismatch. Expected: {len(entries[0].sequence)}, Received: {len(seq)}"
            entries.append(FastaEntry(header, seq))
        msa = cls(entries)
        if remove_insertions:
            keep_inds = [i for i, aa in enumerate(msa.query) if aa != "-"]
            msa = msa.select_positions(keep_inds)
        return msa

    def to_bytes(self) -> bytes:
        version = 1
        version_bytes = version.to_bytes(1, "little")
        seqlen_bytes = self.seqlen.to_bytes(4, "little")
        depth_bytes = self.depth.to_bytes(4, "little")
        array_bytes = self.array.tobytes()
        header_bytes = "\n".join(entry.header for entry in self.entries).encode()
        all_bytes = (
            version_bytes + seqlen_bytes + depth_bytes + array_bytes + header_bytes
        )
        return all_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> MSA:
        version_bytes, seqlen_bytes, depth_bytes, data = (
            data[:1],
            data[1:5],
            data[5:9],
            data[9:],
        )
        version = int.from_bytes(version_bytes, "little")
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        seqlen = int.from_bytes(seqlen_bytes, "little")
        depth = int.from_bytes(depth_bytes, "little")
        array_bytes, header_bytes = data[: seqlen * depth], data[seqlen * depth :]
        array = np.frombuffer(array_bytes, dtype="|S1")
        array = array.reshape(depth, seqlen)
        headers = header_bytes.decode().split("\n")
        # Sometimes the separation is two newlines, which results in an empty header.
        headers = [header for header in headers if header]
        # If all headers were empty (e.g., saved from from_sequences), use empty headers
        if len(headers) == 0 and depth > 0:
            headers = [""] * depth
        entries = [
            FastaEntry(header, b"".join(row).decode())
            for header, row in zip(headers, array)
        ]
        return cls(entries)

    # TODO(jmaccarl): set remove_insertions to True by default here to match other utils
    @classmethod
    def from_sequences(
        cls, sequences: list[str], remove_insertions: bool = False
    ) -> MSA:
        if remove_insertions:
            entries = [
                FastaEntry("", remove_insertions_from_sequence(seq))
                for seq in sequences
            ]
        else:
            entries = [FastaEntry("", seq) for seq in sequences]
        return cls(entries)

    def to_sequence_bytes(self) -> bytes:
        """Stores ONLY SEQUENCES in array format as bytes. Header information will be lost."""
        seqlen_bytes = self.seqlen.to_bytes(4, "little")
        array_bytes = self.array.tobytes()
        all_bytes = seqlen_bytes + array_bytes
        return all_bytes

    @classmethod
    def from_sequence_bytes(cls, data: bytes) -> MSA:
        seqlen_bytes, array_bytes = data[:4], data[4:]
        seqlen = int.from_bytes(seqlen_bytes, "little")
        array = np.frombuffer(array_bytes, dtype="|S1")
        array = array.reshape(-1, seqlen)
        entries = [FastaEntry("", b"".join(row).decode()) for row in array]
        return cls(entries)

    @property
    def depth(self) -> int:
        return len(self.entries)

    @property
    def seqlen(self) -> int:
        return len(self.entries[0].sequence)

    @cached_property
    def array(self) -> np.ndarray:
        return np.array([list(seq) for seq in self.sequences], dtype="|S1")

    @property
    def query(self) -> str:
        return self.entries[0].sequence

    def select_sequences(self, indices: Sequence[int] | np.ndarray) -> MSA:
        """Subselect rows of the MSA."""
        entries = [self.entries[idx] for idx in indices]
        return dataclasses.replace(self, entries=entries)

    def select_positions(self, indices: Sequence[int] | np.ndarray) -> MSA:
        """Subselect columns of the MSA."""
        entries = [
            FastaEntry(header, "".join(seq[idx] for idx in indices))
            for header, seq in self.entries
        ]
        return dataclasses.replace(self, entries=entries)

    def __getitem__(self, indices: int | list[int] | slice | np.ndarray):
        if isinstance(indices, int):
            indices = [indices]

        entries = [
            FastaEntry(header, slice_any_object(seq, indices))
            for header, seq in self.entries
        ]
        return dataclasses.replace(self, entries=entries)

    def __len__(self):
        return self.seqlen

    def greedy_select(self, num_seqs: int, mode: str = "max") -> MSA:
        """Greedily select sequences that either maximize or minimize hamming distance.

        Algorithm proposed in the MSA Transformer paper. Starting from the query sequence,
        iteratively add sequences to the list with the maximum (minimum) average Hamming
        distance to the existing set of sequences.

        Args:
            num_seqs (int): Number of sequences to select.
            mode (str): Whether to maximize or minimize diversity. DO NOT pick 'min' unless
                you're doing it to prove a point for a paper.

        Returns:
            MSA object w/ subselected sequences.
        """
        assert mode in ("max", "min")
        if self.depth <= num_seqs:
            return self

        indices = greedy_select_indices(self.array, num_seqs, mode)
        return self.select_sequences(indices)

    def hhfilter(
        self,
        seqid: int = 90,
        diff: int = 0,
        cov: int = 0,
        qid: int = 0,
        qsc: float = -20.0,
        binary: str = "hhfilter",
    ) -> MSA:
        """Apply hhfilter to the sequences in the MSA and return a filtered MSA."""

        indices = hhfilter(
            self.sequences,
            seqid=seqid,
            diff=diff,
            cov=cov,
            qid=qid,
            qsc=qsc,
            binary=binary,
        )
        return self.select_sequences(indices)

    def select_random_sequences(self, num_seqs: int) -> MSA:
        """Uses random sampling to subselect sequences from the MSA. Always
        keeps the query sequence.
        """
        if num_seqs >= self.depth:
            return self

        # Subselect random, always keeping the query sequence.
        indices = np.sort(
            np.append(
                0, np.random.choice(self.depth - 1, num_seqs - 1, replace=False) + 1
            )
        )
        msa = self.select_sequences(indices)  # type: ignore
        return msa

    def select_diverse_sequences(self, num_seqs: int) -> MSA:
        """Applies hhfilter to select ~num_seqs sequences, then uses random sampling
        to subselect if necessary.
        """
        if num_seqs >= self.depth:
            return self

        msa = self.hhfilter(diff=num_seqs)
        if num_seqs < msa.depth:
            msa = msa.select_random_sequences(num_seqs)
        return msa

    def pad_to_depth(self, depth: int) -> MSA:
        if depth < self.depth:
            raise ValueError(f"Cannot pad to depth {depth} when depth is {self.depth}")
        elif depth == self.depth:
            return self

        num_to_add = depth - self.depth
        extra_entries = [FastaEntry("", "-" * self.seqlen) for _ in range(num_to_add)]
        return dataclasses.replace(self, entries=self.entries + extra_entries)

    @classmethod
    def stack(
        cls, msas: Sequence[MSA], remove_query_from_later_msas: bool = True
    ) -> MSA:
        """Stack a series of MSAs. Optionally remove the query from msas after the first."""
        all_entries = []
        for i, msa in enumerate(msas):
            entries = msa.entries
            if i > 0 and remove_query_from_later_msas:
                entries = entries[1:]
            all_entries.extend(entries)
        return cls(entries=all_entries)

    @cached_property
    def seqid(self) -> np.ndarray:
        array = self.array.view(np.uint8)
        seqid = 1 - cdist(array[0][None], array, "hamming")
        return seqid[0]

    @classmethod
    def concat(
        cls,
        msas: Sequence[MSA],
        join_token: str | None = "|",
        allow_depth_mismatch: bool = False,
    ) -> MSA:
        """Concatenate a series of MSAs horizontally, along the sequence dimension."""
        if not msas:
            raise ValueError("Cannot concatenate an empty list of MSAs")
        msa_depths = [msa.depth for msa in msas]
        if len(set(msa_depths)) != 1:
            if not allow_depth_mismatch:
                raise ValueError("Depth mismatch in concatenating MSAs")
            else:
                max_depth = max(msa_depths)
                msas = [msa.pad_to_depth(max_depth) for msa in msas]
        headers = [
            "|".join([str(h) for h in headers])
            for headers in zip(*(msa.headers for msa in msas))
        ]

        if join_token is None:
            join_token = ""

        seqs = [join_token.join(vals) for vals in zip(*(msa.sequences for msa in msas))]
        entries = [FastaEntry(header, seq) for header, seq in zip(headers, seqs)]
        return cls(entries)


@dataclass(frozen=True)
class FastMSA(SequentialDataclass):
    """Object-oriented interface to an MSA stored as a numpy uint8 array."""

    array: np.ndarray
    headers: list[str] | None = None

    def __post_init__(self):
        if self.headers is not None:
            assert (
                len(self.headers) == self.depth
            ), "Number of headers must match depth."

    @classmethod
    def from_bytes(cls, data: bytes) -> FastMSA:
        version_bytes, seqlen_bytes, depth_bytes, data = (
            data[:1],
            data[1:5],
            data[5:9],
            data[9:],
        )
        version = int.from_bytes(version_bytes, "little")
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        seqlen = int.from_bytes(seqlen_bytes, "little")
        depth = int.from_bytes(depth_bytes, "little")
        array_bytes, header_bytes = data[: seqlen * depth], data[seqlen * depth :]
        array = np.frombuffer(array_bytes, dtype="|S1")
        array = array.reshape(depth, seqlen)
        headers = header_bytes.decode().split("\n")
        # Sometimes the separation is two newlines, which results in an empty header.
        headers = [header for header in headers if header]
        # If all headers were empty (e.g., saved from from_sequences), use empty headers
        if len(headers) == 0 and depth > 0:
            headers = [""] * depth
        return cls(array, headers)

    @classmethod
    def from_sequence_bytes(cls, data: bytes) -> FastMSA:
        seqlen_bytes, array_bytes = data[:4], data[4:]
        seqlen = int.from_bytes(seqlen_bytes, "little")
        array = np.frombuffer(array_bytes, dtype="|S1")
        array = array.reshape(-1, seqlen)
        return cls(array)

    @property
    def depth(self) -> int:
        return self.array.shape[0]

    @property
    def seqlen(self) -> int:
        return self.array.shape[1]

    def __len__(self):
        return self.seqlen

    def __getitem__(self, indices: int | list[int] | slice | np.ndarray):
        if isinstance(indices, int):
            indices = [indices]

        return dataclasses.replace(self, array=self.array[:, indices])

    def select_sequences(self, indices: Sequence[int] | np.ndarray) -> FastMSA:
        """Subselect rows of the MSA."""
        array = self.array[indices]
        headers = (
            [self.headers[idx] for idx in indices] if self.headers is not None else None
        )
        return dataclasses.replace(self, array=array, headers=headers)

    def select_random_sequences(self, num_seqs: int) -> FastMSA:
        """Uses random sampling to subselect sequences from the MSA. Always
        keeps the query sequence.
        """
        if num_seqs >= self.depth:
            return self

        # Subselect random, always keeping the query sequence.
        indices = np.sort(
            np.append(
                0, np.random.choice(self.depth - 1, num_seqs - 1, replace=False) + 1
            )
        )
        msa = self.select_sequences(indices)  # type: ignore
        return msa

    def pad_to_depth(self, depth: int) -> FastMSA:
        if depth < self.depth:
            raise ValueError(f"Cannot pad to depth {depth} when depth is {self.depth}")
        elif depth == self.depth:
            return self

        num_to_add = depth - self.depth
        array = np.pad(
            self.array,
            [(0, num_to_add), (0, 0)],
            constant_values=ord("-") if self.array.dtype == np.uint8 else b"-",
        )
        headers = self.headers
        if headers is not None:
            headers = headers + [""] * num_to_add
        return dataclasses.replace(self, array=array, headers=headers)

    @classmethod
    def concat(
        cls,
        msas: Sequence[FastMSA],
        join_token: str | None = None,
        allow_depth_mismatch: bool = False,
    ) -> FastMSA:
        """Concatenate a series of MSAs horizontally, along the sequence dimension."""
        if not msas:
            raise ValueError("Cannot concatenate an empty list of MSAs")
        if join_token is not None and join_token != "":
            raise NotImplementedError("join_token is not supported for FastMSA")

        msa_depths = [msa.depth for msa in msas]
        if len(set(msa_depths)) != 1:
            if not allow_depth_mismatch:
                raise ValueError("Depth mismatch in concatenating MSAs")
            else:
                max_depth = max(msa_depths)
                msas = [msa.pad_to_depth(max_depth) for msa in msas]
        headers = [
            "|".join([str(h) for h in headers])
            for headers in zip(
                *(
                    msa.headers if msa.headers is not None else [""] * msa.depth
                    for msa in msas
                )
            )
        ]

        array = np.concatenate([msa.array for msa in msas], axis=1)
        return cls(array, headers)

    def to_msa(self) -> MSA:
        headers = (
            self.headers
            if self.headers is not None
            else [f"seq{i}" for i in range(self.depth)]
        )
        entries = [
            FastaEntry(header, b"".join(row).decode())
            for header, row in zip(headers, self.array)
        ]
        return MSA(entries)

    @classmethod
    def stack(
        cls, msas: Sequence[FastMSA], remove_query_from_later_msas: bool = True
    ) -> FastMSA:
        """Stack a series of MSAs. Optionally remove the query from msas after the first."""
        arrays = []
        all_headers = []
        for i, msa in enumerate(msas):
            array = msa.array
            headers = msa.headers
            if i > 0 and remove_query_from_later_msas:
                array = array[1:]
                if headers is not None:
                    headers = headers[1:]
            arrays.append(array)
            if headers is not None:
                all_headers.extend(headers)
        return cls(np.concatenate(arrays, axis=0), all_headers)
