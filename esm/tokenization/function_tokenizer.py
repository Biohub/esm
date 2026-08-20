"""Tokenizes annotations of protein function."""

import re
import string
from functools import cache, cached_property, partial
from typing import Collection

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C
from esm.utils.function import interpro, lsh, tfidf
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.types import FunctionAnnotation, PathLike


def _default_data_path(x: PathLike | None, d: PathLike) -> PathLike:
    return x if x is not None else C.data_root("esm3") / d


def _default_local_data_path(x: PathLike | None, d: PathLike) -> PathLike:
    return x if x is not None else d


class InterProQuantizedTokenizer(EsmTokenizerBase):
    """Tokenizer for functional annotations.

    This tokenizer converts InterPro and/or function keywords into a multi-token
    representation by hashing TF-IDF vector representations of the text associated with
    the fuction and then applying a locality sensitive hash (LSH).
    """

    def __init__(
        self,
        depth: int = 8,
        lsh_bits_per_token: int = 8,
        lsh_path: PathLike | None = None,
        keyword_vocabulary_path: PathLike | None = None,
        keyword_idf_path: PathLike | None = None,
        interpro_entry_path: PathLike | None = None,
        interpro2keywords_path: PathLike | None = None,
    ):
        """Constructs function tokenizer.

        Args:
            depth: number of tokens emitted in each position.
            lsh_bits_per_token: Number of LSH bits per token. Determines the vocabulary
                  size.
            lsh_path: path to locality sensitive hash (LSH) hyperplanes.
            keyword_vocabulary_path: path to csv containing function keyword vocabulary.
            keyword_idf_path: path to IDF values for each keyword.
            interpro_entry_csv_path: path to list of InterPro entries in CSV format.
            interpro2keywords_path: path to CSV mapping InterPro IDs to function keywords.
        """
        self.depth = depth

        self.keyword_vocabulary_path = _default_local_data_path(
            keyword_vocabulary_path, C.KEYWORDS_VOCABULARY
        )
        self.keyword_idf_path = _default_local_data_path(
            keyword_idf_path, C.KEYWORDS_IDF
        )

        self._interpro2keywords_path = _default_local_data_path(
            interpro2keywords_path, C.INTERPRO2KEYWORDS
        )
        self.interpro_ = interpro.InterPro(
            entries_path=_default_local_data_path(interpro_entry_path, C.INTERPRO_ENTRY)
        )

        self.lsh_path = lsh_path
        self.lsh_bits_per_token = lsh_bits_per_token
        self.lsh_vocab_size = 1 << lsh_bits_per_token

        # This is the offset into the vocabulary where LSH tokens start.
        self._lsh_token_vocab_offset = len(self.special_tokens) + 1  # +1 for <none>

    @cached_property
    def _lsh(self) -> lsh.LSHTokenized:
        """Locality sensitive hash for function annotations."""
        return lsh.LSHTokenized(
            self.lsh_bits_per_token,
            len(self.keyword_vocabulary),
            self.depth,
            _default_data_path(self.lsh_path, C.LSH_TABLE_PATHS["8bit"]),
        )

    @cached_property
    def interpro2keywords(self) -> dict[str, list[str]]:
        """Mapping from InterPro ID to function keywords."""
        df = pd.read_csv(self._interpro2keywords_path)
        assert "interpro_id" in df.columns and "keywords" in df.columns, df.columns
        return dict(zip(df.interpro_id, df.keywords.str.split(",")))

    @cached_property
    def interpro_labels(self) -> list[str]:
        """The set of supported InterPro labels."""
        return sorted(self.interpro2keywords.keys())

    @cached_property
    def interpro_to_index(self) -> dict[str, int]:
        """Mapping from InterPro id to index."""
        return {id: i for i, id in enumerate(self.interpro_labels)}

    @property
    def keyword_vocabulary(self) -> list[str]:
        """Set of supported keywords."""
        return self._tfidf.vocabulary

    @property
    def keyword_to_index(self) -> dict[str, int]:
        """Mapping from keywords to index."""
        return self._tfidf.vocab_to_index

    @cached_property
    def _tfidf(self) -> tfidf.TFIDFModel:
        """Creates TF-IDF model for encoding function keywords."""
        return tfidf.TFIDFModel(
            vocabulary_path=self.keyword_vocabulary_path, idf_path=self.keyword_idf_path
        )

    @cached_property
    def special_tokens(self) -> list[str]:
        """List of special tokens which come before cluster tokens in vocab."""
        return ["<pad>", "<motif>", "<unk>"]

    @cached_property
    def vocab(self) -> list[str]:
        """Vocabulary of function tokens."""
        lsh_tokens = [f"<lsh:{i}>" for i in range(self.lsh_vocab_size)]
        return self.special_tokens + ["<none>"] + lsh_tokens

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:
        return {token: token_id for token_id, token in enumerate(self.vocab)}

    def get_special_tokens_mask(self, encoded: torch.Tensor) -> torch.Tensor:
        """Determines where in the sequence are special tokens."""
        where = encoded < len(self.special_tokens)
        assert torch.all(torch.all(where, dim=1) | torch.all(~where, dim=1))
        return where[:, 0]

    def tokenize(
        self,
        annotations: list[FunctionAnnotation],
        seqlen: int,
        p_keyword_dropout: float = 0.0,
    ) -> list[str]:
        """Encodes range-annotations of protein function as tokens.

        Args:
            features: Annotated function ranges, either as InterPro ids or keywords.
            seqlen: length of sequence.
            p_keyword_dropout: Optional probability of dropping out keywords from the
              input annotations.
        Returns:
            Tokenized representation of function annotations as a list of string tokens
            of size seqlen.
        """
        assert seqlen >= 0

        if not annotations:
            return ["<pad>"] * seqlen

        # Expand the range annotations into positional annotaiton sets.
        positional_labels: list[set[str]] = [set() for _ in range(seqlen)]
        for annotation in annotations:
            assert 1 <= annotation.start <= annotation.end <= seqlen, (
                f"Invalid annotation range [{annotation.start}, {annotation.end}] for "
                f"sequence length {seqlen}."
            )
            for i in range(annotation.start - 1, annotation.end):
                positional_labels[i].add(annotation.label)

        if p_keyword_dropout > 0:
            keyword_mask = (
                np.random.random(len(self._tfidf.vocabulary)) < p_keyword_dropout
            )
        else:
            keyword_mask = None

        # Annotations tend to be repetitive over the length of the sequence - cache their
        # hashes to speed up tokenization.
        hash_fn = cache(partial(self._function_text_hash, keyword_mask=keyword_mask))

        tokens: list[str] = []
        for labels in positional_labels:
            if not labels:
                token = "<none>"
            else:
                lsh_hash = hash_fn(frozenset(labels))
                if lsh_hash is not None:
                    assert len(lsh_hash) == self.depth
                    token = "<lsh:" + ",".join(map(str, lsh_hash)) + ">"
                else:
                    token = "<unk>"

            tokens.append(token)

        return tokens

    def _function_text_hash(
        self, labels: Collection[str], keyword_mask: np.ndarray | None = None
    ) -> np.ndarray | None:
        """Applies a locality sensitive hash (LSH) to function text.

        Args:
            labels: InterPro ids and/or keywords.
            keyword_mask: optional boolean array shaped (keyword_vocab_size,) indicating
                which keywords to drop before hashing.
        Returns:
            LSH shaped (depth,) or None if there is no text or keywords to hash.
        """
        # Split labels into either InterPro ids or keywords.
        interpro_ids = []
        keywords = []
        for label in labels:
            match = re.search(r"IPR\d+", label)
            if match and match.group() in self.interpro_to_index:
                interpro_ids.append(match.group())
            elif label in self._tfidf.vocab_to_index:
                keywords.append(label)
            else:
                raise ValueError(f"Unsupported: {label}")

        vec: sp.csr_matrix = self._tfidf.encode(keywords)

        # Perform an element-wise maximum over TF-IDF vectors from distinct tags to
        # avoid tags getting "washed out" by eg. 4 very similar tags. Keywords are
        # incorporated as another TF-IDF vector
        vec: sp.csr_matrix = self._tfidf.encode(keywords)
        for interpro_id in interpro_ids:
            interpro_keywords = self.interpro2keywords.get(interpro_id, [])
            vec_ = self._tfidf.encode(interpro_keywords)
            vec = vec.maximum(vec_)

        if keyword_mask is not None:
            vec.data *= 1 - np.take(keyword_mask, vec.indices)

        if vec.sum() == 0:
            return None

        return self._lsh(vec)[0, :]

    def encode(
        self, tokens: list[str], add_special_tokens: bool = True
    ) -> torch.Tensor:
        """Encodes string tokens as token-id tensor.

        Args:
            tokens: list of individual tokens. e.g. ["<none>", "<pq:1,2,3,4>"]
            add_special_tokens: whether to add a single pad token at the start and end
              of the sequence to act as <cls> and <eos> tokens.
        Returns:
            <int>[length, depth] function tokens. Length will be +2 of input tokens
            length when add_special_tokens is True.
        """
        token_ids = torch.zeros(size=(len(tokens), self.depth), dtype=torch.int64)
        for i, token in enumerate(tokens):
            token_ids[i, :] = torch.tensor(self._token2ids(token))
        if add_special_tokens:
            token_ids = F.pad(
                token_ids, (0, 0, 1, 1), value=self.vocab_to_index["<pad>"]
            )
        return token_ids

    def lookup_annotation_name(self, annotation: FunctionAnnotation) -> str | None:
        return self.interpro_.lookup_name(annotation.label)

    def format_annotation(self, annotation: FunctionAnnotation) -> str:
        annotation_name = self.lookup_annotation_name(annotation)
        if annotation_name is not None:
            return f"{annotation_name} ({annotation.label})"
        else:
            return annotation.label

    def _token2ids(self, token: str) -> list[int]:
        """Converts token into token_id set of length depth."""
        if re.match(r"<lsh:[\d+,]+>", token):
            lsh_ids = [int(lsh_id) for lsh_id in re.findall(r"\d+", token)]
            assert len(lsh_ids) == self.depth, (
                f"Expected token to have {self.depth} ids found {lsh_ids}"
            )
            return [self._lsh_token_vocab_offset + lsh_id for lsh_id in lsh_ids]
        elif token == "<none>" or token in self.special_tokens:
            return [self.vocab_to_index[token]] * self.depth
        else:
            raise ValueError(f"Unknown token: {token}")

    def batch_encode(
        self, token_batch: list[list[str]], add_special_tokens: bool = True
    ) -> torch.Tensor:
        """Encodes batch of function tokens.

        Args:
            token_batch: batch of function tokens.
            add_special_tokens: whether to add special tokens.
        Returns:
            <int>[batch_size, max_length, depth] batch of encoded tokens.
        """
        encoded = [
            self.encode(tokens, add_special_tokens=add_special_tokens)
            for tokens in token_batch
        ]
        return stack_variable_length_tensors(
            encoded, constant_value=self.vocab_to_index["<pad>"]
        )

    def decode(self, encoded: torch.Tensor):
        raise NotImplementedError(
            "Function token decoding should be handled with "
            "util.decoding.decode_function_annotations"
        )

    @property
    def mask_token(self) -> str:
        return "<pad>"

    @property
    def mask_token_id(self) -> int:
        return self.vocab_to_index[self.mask_token]

    @property
    def bos_token(self) -> str:
        return "<pad>"

    @property
    def bos_token_id(self) -> int:
        return self.vocab_to_index[self.bos_token]

    @property
    def eos_token(self) -> str:
        return "<pad>"

    @property
    def eos_token_id(self) -> int:
        return self.vocab_to_index[self.eos_token]

    @property
    def pad_token(self) -> str:
        return "<pad>"

    @property
    def pad_token_id(self) -> int:
        return self.vocab_to_index[self.pad_token]

    @property
    def chain_break_token(self) -> str:
        return "<pad>"

    @property
    def chain_break_token_id(self) -> int:
        return self.vocab_to_index[self.chain_break_token]

    @property
    def all_token_ids(self):
        return list(range(len(self.vocab)))

    @property
    def special_token_ids(self):
        return [self.vocab_to_index[token] for token in self.special_tokens]


def _texts_to_keywords(texts: list[str]) -> list[str]:
    """Breaks InterPro/GO free-text description set into bag-of-n-grams for n={1,2}.

    Args:
        texts: collection of text descriptions, i.e. InterPro/GO names.
    Returns:
        Collection of terms/n-grams
    """
    keywords = []
    for text in texts:
        keywords.extend(_keywords_from_text(text))
    return keywords


def _keywords_from_text(text: str) -> list[str]:
    """Splits text into unigrams and bigrams."""
    elements = text.split(", ")

    terms = []
    for element in elements:
        element = _sanitize(element)
        words = element.split()

        # Add 1-mers
        terms.extend(words)

        # Add 2-mers
        for i in range(len(words) - 1):
            bigram = words[i] + " " + words[i + 1]
            terms.append(bigram)

    return [term for term in terms if len(term) > 1 and term not in _EXCLUDED_TERMS]


def _sanitize(text: str) -> str:
    text = text.replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return text


# These terms are omitted from textual representations since they are pervasive and
# unspecific to particular protein function.
_EXCLUDED_TERMS = {
    "binding domain",
    "biological_process",
    "biological process",
    "biologicalprocess",
    "c",
    "cellular_component",
    "cellular component",
    "cellularcomponent",
    "cellular_process",
    "cellularprocess",
    "cellular process",
    "cellularprocess",
    "like domain",
    "molecular function",
    "molecular_function",
    "molecularfunction",
    "n",
}
