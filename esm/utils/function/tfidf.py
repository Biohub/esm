"""Term-Frequency / Inverse Document Frequency (TF-IDF) model."""

from collections import Counter
from functools import cached_property

import numpy as np
from cloudpathlib import AnyPath
from scipy import sparse

from esm.utils.types import PathLike


class TFIDFModel:
    """Term-Frequency / Inverse Document Frequency (TF-IDF) model.
    Mimics sklearn.feature_extraction.text.TfidfVectorizer with sublinear_tf=True
    """

    def __init__(self, vocabulary_path: PathLike, idf_path: PathLike):
        with AnyPath(vocabulary_path).open("r") as f:
            self.vocabulary = f.read().strip().split("\n")

        with AnyPath(idf_path).open("rb") as f:
            self.idf_ = np.load(f)

        assert self.idf_.ndim == 1
        assert len(self.idf_) == len(self.vocabulary), (
            f"IDF size must match vocabulary size, got {len(self.idf_)} and {len(self.vocabulary)}"
        )

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:
        return {term: index for index, term in enumerate(self.vocabulary)}

    def encode(self, terms: list[str]) -> sparse.csr_matrix:
        """Encodes terms as TF-IDF vectors.

        Args:
            terms: list of terms to encode.

        Returns:
            TF-IDF vector encoded as sparse matrix of shape (1, num_terms)
        """
        counter = Counter(filter(self.vocabulary.__contains__, terms))
        indices = [self.vocab_to_index[term] for term in counter]

        tf = np.array([count for term, count in counter.items()])
        idf = np.take(self.idf_, indices)

        values = (1 + np.log(tf)) * idf
        values /= np.linalg.norm(values)

        return sparse.csr_matrix(
            (values, (np.zeros_like(indices), indices)), shape=(1, len(self.vocabulary))
        )

    def decode(self, vec: sparse.csr_matrix) -> list[str]:
        """Extract terms from TF-IDF."""
        return [self.vocabulary[i] for i in vec.indices]
