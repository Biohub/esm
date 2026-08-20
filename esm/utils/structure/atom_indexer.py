import numpy as np

from esm.utils.structure.protein_structure import index_by_atom_name


class AtomIndexer:
    def __init__(self, structure, property: str, dim: int):
        self.structure = structure
        self.property = property
        self.dim = dim

    def __getitem__(self, atom_names: str | list[str]) -> np.ndarray:
        return index_by_atom_name(
            getattr(self.structure, self.property), atom_names, self.dim
        )
