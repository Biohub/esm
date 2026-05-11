"""Function Token Decoder."""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from cloudpathlib import AnyPath

from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
from esm.utils.constants import esm3 as C
from esm.utils.misc import merge_annotations, merge_ranges
from esm.utils.types import FunctionAnnotation


@dataclass(frozen=True)
class FunctionTokenDecoderConfig:
    """Configures function token decoder."""

    # Embedding dimension of decoder.
    d_model: int = 1024
    # Number of attention heads of decoder.
    n_heads: int = 8
    # Number of layers of decoder.
    n_layers: int = 3
    # Number of integer values that function tokens may assume.
    function_token_vocab_size: int = 260
    # Number of function tokens at each position.
    function_token_depth: int = 8
    # Number of InterPro labels that can be decoded.
    num_interpro_classes: int = 29026
    # Number of function keywords that can be decoded.
    keyword_vocabulary_size: int = 58641
    # List of supported InterPro ids.
    interpro_entry_list: str = field(default_factory=lambda: str(C.INTERPRO_ENTRY))
    # Path to keywords vocabulary.
    keyword_vocabulary_path: str = field(
        default_factory=lambda: str(C.data_root("esm3") / C.KEYWORDS_VOCABULARY)
    )
    # Whether to unpack LSH bits into single-bit tokens.
    unpack_lsh_bits: bool = True
    # The number of special tokens in the function tokenizer vocabulary which come
    # before the LSH tokens.
    num_special_tokens: int = 4
    # The number of bits per LSH token in the function tokenizer.
    bits_per_token: int = 8


class FunctionTokenDecoder(nn.Module):
    def __init__(self, config: FunctionTokenDecoderConfig | None = None):
        """Constructs function token decoder."""
        super().__init__()
        if config is None:
            config = FunctionTokenDecoderConfig()
        self.config = config

        # Get the supported set of InterPro ids.
        with AnyPath(config.interpro_entry_list).open("r") as f:
            df = pd.read_csv(f, sep="\t")
        self.interpro_ids = sorted(df.ENTRY_AC)
        self.interpro2index = {
            interpro_id: i for i, interpro_id in enumerate(self.interpro_ids)
        }
        assert len(self.interpro_ids) == config.num_interpro_classes

        with AnyPath(config.keyword_vocabulary_path).open("r") as f:
            self.keywords_vocabulary: list[str] = list(f.read().strip().split("\n"))
            assert len(self.keywords_vocabulary) == config.keyword_vocabulary_size

        if config.unpack_lsh_bits:
            vocab_size = 2 * config.function_token_depth * config.bits_per_token
        else:
            # Function-token id's re-use the same token ids at each position along the depth
            # dimension, despite distinct meanings. The decoder should take this into
            # account so create distinct embeddings for tokens at each position.
            vocab_size = (
                self.config.function_token_depth * self.config.function_token_vocab_size
            )

        self.embedding = nn.Embedding(
            # Function-token id's re-use the same token ids at each position along the
            # depth dimension, despite distinct meanings. The decoder should take this
            # into account so create distinct embeddings for tokens at each position.
            num_embeddings=(vocab_size),
            embedding_dim=config.d_model,
        )
        self.decoder = TransformerStack(
            d_model=config.d_model,
            n_heads=config.n_heads,
            v_heads=None,
            n_layers=config.n_layers,
            n_layers_geom=0,
            scale_residue=False,
            bias=True,
            qk_layernorm=False,
            ffn_type="gelu",
            expansion_ratio=4,
        )
        self.heads = nn.ModuleDict(
            {
                # Binary classification head predicting which keywords are present.
                "keyword_logits": RegressionHead(
                    d_model=config.d_model,
                    output_dim=config.keyword_vocabulary_size,
                    hidden_dim=4 * config.d_model,
                ),
                # Regresses the TF-IDF value of each present keyword.
                "keyword_tfidf": RegressionHead(
                    d_model=config.d_model,
                    output_dim=config.keyword_vocabulary_size,
                    hidden_dim=4 * config.d_model,
                ),
                # Predicts which InterPro annotations are present.
                "interpro_logits": RegressionHead(
                    d_model=config.d_model,
                    output_dim=config.num_interpro_classes,
                    hidden_dim=4 * config.d_model,
                ),
            }
        )

    def forward(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through function token decoder.

        Args:
            token_ids: <int>[batch_size, function_token_depth] batch of function tokens
                ids to decode.
        Returns:
            interpro_logits: binary classification logits tensor of shape
                <float>[batch_size, num_interpro_classes]
        """
        assert token_ids.ndim == 2
        assert token_ids.shape[1] == self.config.function_token_depth
        batch_size, depth = token_ids.shape

        if self.config.unpack_lsh_bits:
            # Shift values into [0, 2^bits/token)
            lsh_bits = token_ids - self.config.num_special_tokens
            # extract each bit. (hob stands for highest-order bit)
            bits = torch.concat(
                [
                    torch.bitwise_and(lsh_bits, 1 << hob).gt(0).to(torch.int32)
                    for hob in range(self.config.bits_per_token)
                ],
                dim=1,
            )
            assert bits.shape == (batch_size, depth * self.config.bits_per_token)

            # Shift each bit into individual vocabulary ranges, so they get distinct
            # embeddings.
            vocab_offsets = 2 * torch.arange(
                depth * self.config.bits_per_token, device=token_ids.device
            )
            inputs = vocab_offsets[None, :] + bits

            # zero-out special tokens, i.e. non LSH tokens.
            where_special = token_ids < self.config.num_special_tokens
            inputs = torch.where(where_special.any(dim=1, keepdim=True), 0, inputs)
        else:
            # Apply depth-position offset to use distinct vocabs. See __init__ for
            # explaination.
            vocab_offsets = self.config.function_token_vocab_size * torch.arange(
                self.config.function_token_depth, device=token_ids.device
            )
            inputs = token_ids + vocab_offsets[None, :]

        embed = self.embedding(inputs)
        encoding, _, _, _ = self.decoder(embed)
        pooled = torch.mean(encoding, dim=1)

        return {name: head(pooled) for name, head in self.heads.items()}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def decode(
        self,
        function_token_ids: torch.Tensor,
        tokenizer: InterProQuantizedTokenizer,
        decode_annotations: bool = True,
        annotation_threshold: float = 0.1,
        decode_keywords=True,
        keywords_threshold: float = 0.5,
        annotation_min_length: int | None = 5,
        annotation_gap_merge_max: int | None = 3,
    ):
        """Decodes function tokens into predicted annotations and keywords.

        Args:
            function_token_ids: <int>[length, depth] function token ids. NOTE:
                without <bos>/<eos> prefix
            tokenizer: function tokenizer.
            decode_annotations: whether to decode InterPro annotations.
            annotation_threshold: threshold for emitting a function annotation.
            decode_keywords: whether to decode function keywords.
            keywords_threshold: threshold for emitting a keyword.
            annotation_min_length: optional minimum length of predicted annotations for
                size filtering.
            annotation_gap_merge_max: optional merge adjacent annotation of the same type
        Returns:
            Decoder outputs:
            - "interpro_logits": <float>[length, num_interpro] predicted interpro logits.
            - "interpro_preds": <bool>[length, num_interpro] predicted intepro labels.
            - "interpro_annotations": list[FunctionAnnotation] predicted InterPro
                annotations
            - "keyword_logits": <float>[length, keyword_vocabulary] binary prediciton
              logits for keywrods.
            - "function_keywords": list[FunctionAnnotation] predicted function keyword
                ranges.
        """
        assert function_token_ids.ndim == 2
        assert function_token_ids.shape[1] == tokenizer.depth
        assert self.config.function_token_depth == tokenizer.depth

        outputs = {}

        outputs = self(function_token_ids.to(self.device))

        # Only decode in positions that have function tokens.
        where_decode = torch.all(
            (function_token_ids != tokenizer.vocab_to_index["<pad>"])
            & (function_token_ids != tokenizer.vocab_to_index["<none>"])
            & (function_token_ids != tokenizer.vocab_to_index["<unk>"]),
            dim=1,
        )

        # Decode InterPro annotations ranges.
        interpro_preds = F.sigmoid(outputs["interpro_logits"])
        interpro_preds = interpro_preds >= annotation_threshold
        interpro_preds[~where_decode, :] = False
        outputs["interpro_preds"] = interpro_preds
        if decode_annotations:
            annotations: list[FunctionAnnotation] = []
            preds: np.ndarray = interpro_preds.detach().cpu().numpy()
            for position_index, class_index in zip(*preds.nonzero()):
                interpro_id = self.interpro_ids[class_index]
                annotation = FunctionAnnotation(
                    label=interpro_id,
                    start=position_index,  # one-index inclusive (BOS shifts indexes +1)
                    end=position_index,  # one-index inclusive
                )
                annotations.append(annotation)

            annotations = merge_annotations(
                annotations, merge_gap_max=annotation_gap_merge_max
            )

            # Drop very small annotations.
            if annotation_min_length is not None:
                annotations = [
                    annotation
                    for annotation in annotations
                    if annotation.end - annotation.start + 1 >= annotation_min_length
                ]

            outputs["interpro_annotations"] = annotations

        # Decode function keyword ranges.
        keyword_logits = outputs["keyword_logits"]
        keyword_logits[~where_decode, :] = -torch.inf
        if decode_keywords:
            keyword_preds = F.sigmoid(keyword_logits) >= keywords_threshold
            keywords = self._preds_to_keywords(keyword_preds.detach().cpu().numpy())
            keywords = merge_annotations(
                keywords, merge_gap_max=annotation_gap_merge_max
            )
            if annotation_min_length is not None:
                keywords = [
                    annotation
                    for annotation in keywords
                    if annotation.end - annotation.start + 1 >= annotation_min_length
                ]

            outputs["function_keywords"] = keywords

        return outputs

    def _preds_to_keywords(self, keyword_preds: np.ndarray) -> list[FunctionAnnotation]:
        """Converts output log-TFDF to predicted keywords over the sequence.

        Args:
            keyword_precs: <bool>[length, keyword_vocab] positional predictions of
              function keywords from the keyword prediction head.
        Returns:
            Non-overlapping keyword annotated ranges along the sequence. Note that indices
            will index into the *sequence*, not the function token array which has a
            <pad> prefix.
        """
        assert keyword_preds.ndim == 2
        assert keyword_preds.shape[1] == self.config.keyword_vocabulary_size

        keyword_positions: dict[str, list[range]] = defaultdict(list)
        for position, keyword_id in zip(*np.nonzero(keyword_preds)):
            keyword = self.keywords_vocabulary[keyword_id]
            keyword_positions[keyword].append(range(position, position + 1))

        annotations: list[FunctionAnnotation] = []
        for keyword, ranges in keyword_positions.items():
            for range_ in merge_ranges(ranges):
                annotation = FunctionAnnotation(
                    label=keyword,
                    start=range_.start,  # one-index inclusive  (BOS shifts indexes +1)
                    end=range_.stop - 1,  # one-index exclusive -> one-index inclusive
                )
                annotations.append(annotation)

        return annotations
