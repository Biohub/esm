import numpy as np
from cloudpathlib import AnyPath

from esm.utils.types import PathLike


class LSHTable:
    def __init__(self, n_bits: int, dim: int, hyperplanes: np.ndarray | None = None):
        if hyperplanes is None:
            hyperplanes = np.random.randn(n_bits, dim)
            hyperplanes = hyperplanes / np.linalg.norm(
                hyperplanes, axis=-1, keepdims=True
            )
        else:
            assert hyperplanes.shape == (n_bits, dim), (
                hyperplanes.shape,
                (n_bits, dim),
            )
        assert hyperplanes is not None
        self.hyperplanes: np.ndarray = hyperplanes
        self.values = 1 << np.arange(n_bits)

    def __call__(self, array, tokenize: bool = True):
        similarity = self.hyperplanes @ array.T
        bits = np.where(similarity >= 0, 1, 0)
        if tokenize:
            tokens = bits.T @ self.values
            return tokens
        else:
            return bits.T


class LSHTokenized:
    def __init__(
        self,
        n_bits: int,
        dim: int,
        num_tables: int = 1,
        filepath: PathLike | None = None,
        allow_create_hyperplanes: bool = False,  # set this if you want the lsh to allow creation of hyperplanes
    ):
        table_hyperplanes = None
        if filepath is not None:
            filepath = AnyPath(filepath)
            if not filepath.exists():
                raise FileNotFoundError(filepath)
            table_hyperplanes = np.load(filepath)
            for i in range(num_tables):
                assert str(i) in table_hyperplanes, f"Missing hyperplane for table {i}"
        elif not allow_create_hyperplanes:
            raise RuntimeError(
                "Not allowed to create hyperplanes but no filepath provided"
            )

        self.tables = [
            LSHTable(
                n_bits,
                dim,
                table_hyperplanes[str(i)] if table_hyperplanes is not None else None,
            )
            for i in range(num_tables)
        ]

    def write_hyperplanes(self, filepath: PathLike):
        hyperplanes: dict[str, np.ndarray] = {
            str(i): table.hyperplanes for i, table in enumerate(self.tables)
        }
        np.savez(filepath, **hyperplanes)

    def __call__(self, array):
        tokens = np.stack([table(array) for table in self.tables], 1)
        return tokens


class LSHBitstream:
    def __init__(
        self,
        n_bits: int,
        dim: int,
        filepath: PathLike | None = None,
        allow_create_hyperplanes: bool = False,  # set this if you want the lsh to allow creation of hyperplanes
    ):
        table_hyperplanes = None
        if filepath is not None:
            filepath = AnyPath(filepath)
            if not filepath.exists():
                raise FileNotFoundError(filepath)
            table_hyperplanes = np.load(filepath)
        elif not allow_create_hyperplanes:
            raise RuntimeError(
                "Not allowed to create hyperplanes but no filepath provided"
            )

        self.table = LSHTable(
            n_bits, dim, table_hyperplanes if table_hyperplanes is not None else None
        )

    def write_hyperplanes(self, filepath: PathLike):
        np.save(filepath, self.table.hyperplanes)

    def __call__(self, array):
        return self.table(array, tokenize=False)
