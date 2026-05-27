import io
from pathlib import Path
from typing import Generator, Iterable, NamedTuple

PathOrBuffer = str | Path | io.TextIOBase
FastaEntry = NamedTuple("FastaEntry", [("header", str), ("sequence", str)])


def parse_fasta(fasta_string: str) -> Generator[FastaEntry, None, None]:
    """
    Parses a fasta file and yields FastaEntry objects

    Args:
        fasta_string: The fasta file as a string
    Returns:
        A generator of FastaEntry objects
    """
    header = None
    seq = []
    num_sequences = 0
    for line in fasta_string.splitlines():
        if not line or line[0] == "#":
            continue
        if line.startswith(">"):
            if header is not None:
                yield FastaEntry(header, "".join(seq))
                seq = []
            header = line[1:].strip()
        else:
            seq.append(line)
    if header is not None:
        num_sequences += 1
        yield FastaEntry(header, "".join(seq))

    if num_sequences == 0:
        raise ValueError("Found no sequences in input")


def read_sequences(path: PathOrBuffer) -> Generator[FastaEntry, None, None]:
    # Uses duck typing to try and call the right method
    # Doesn't use explicit isinstance check to support
    # inputs that are not explicitly str/Path/TextIOBase but
    # may support similar functionality
    data = None  # type: ignore
    try:
        if str(path).endswith(".gz"):
            import gzip

            data = gzip.open(path, "rt")  # type: ignore
        else:
            try:
                data = open(path)  # type: ignore
            except TypeError:
                data: io.TextIOBase = path  # type: ignore

        yield from parse_fasta(data.read())
    finally:
        if data is not None:
            data.close()


def read_first_sequence(path: PathOrBuffer) -> FastaEntry:
    return next(iter(read_sequences(path)))


def count_fasta_sequences(path: str | Path) -> int:
    """Count sequences in a FASTA file by counting header lines.

    Faster than parsing the full file — only scans for '>' prefixes.
    Returns 0 if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.startswith(">"))


def append_fasta_sequence(header: str, sequence: str, path: str | Path) -> None:
    """Append a single sequence to a FASTA file (creating it if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The existing file may not end with a newline (e.g., write_sequences()
    # explicitly avoids writing a newline at the end), so we insert one before
    # appending to avoid merging with the last line.
    needs_newline = (
        path.exists() and path.stat().st_size > 0 and path.read_bytes()[-1:] != b"\n"
    )
    with open(path, "a") as f:
        if needs_newline:
            f.write("\n")
        f.write(f">{header}\n{sequence}\n")


def write_sequences(sequences: Iterable[tuple[str, str]], path: PathOrBuffer) -> None:
    needs_closing = False
    handle = None
    try:
        try:
            handle = open(path, "w")  # type: ignore
            needs_closing = True
        except TypeError:
            handle = path
        has_prev = False
        for header, seq in sequences:
            if has_prev:
                handle.write("\n")  # type: ignore
            handle.write(f">{header}\n{seq}")  # type: ignore
            has_prev = True
    finally:
        if needs_closing:
            handle.close()  # type: ignore
