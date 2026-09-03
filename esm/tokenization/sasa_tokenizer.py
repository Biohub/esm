import math
from functools import cached_property

import torch

from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C


class SASADiscretizingTokenizer(EsmTokenizerBase):
    """Tokenizer for Solvent Accessible Surface Area (SASA)."""

    def __init__(self, boundaries: list[float] = C.SASA_DISCRETIZATION_BOUNDARIES):
        self._boundaries = sorted(boundaries)

    @cached_property
    def special_tokens(self) -> list[str]:
        return ["<pad>", "<motif>", "<unk>"]

    @cached_property
    def vocab(self) -> list[str]:
        """Discrete token vocabulary.

        Returns:
            token vocabulary with ranges represented as "<low-high>".
        """
        boundary_strs = ["0"] + [str(b) for b in self._boundaries] + ["inf"]
        range_tokens = [
            f"<{low}-{high}>"
            for low, high in zip(boundary_strs[:-1], boundary_strs[1:])
        ]
        return self.special_tokens + range_tokens

    @cached_property
    def midpoints_tensor(self) -> torch.Tensor:
        """Midpoints of the SASA token ranges."""
        boundaries = [0] + self._boundaries + [self._boundaries[-1] * 2]
        midpoint_tokens = [
            (float(high) + float(low)) / 2
            for low, high in zip(boundaries[:-1], boundaries[1:])
        ]
        midpoint_tokens = [float("nan"), float("nan"), float("nan")] + midpoint_tokens
        return torch.Tensor(midpoint_tokens)

    def midpoints(self) -> list[float]:
        """Midpoints of the SASA token ranges."""
        return self.midpoints_tensor.tolist()

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:
        """Constructs token -> token id mapping."""
        return {word: i for i, word in enumerate(self.vocab)}

    def get_special_tokens_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Determines which positions are special tokens.

        Args:
            tokens: <int>[length]
        Returns:
            <bool>[length] tensor, true where special tokens are located in the input.
        """
        return tokens < len(self.special_tokens)

    def encode(
        self, values: list[float | str], add_special_tokens: bool = True
    ) -> torch.Tensor:
        """Encodes SASA values as discrete tokens.

        Args:
            values: list of either SASA values or individual tokens. For example
                [1.2, "<pad>", 10.3, <pad>, 0.]
        Returns:
            Token ids as tensor. Adds BOS and EOS special tokens.
        """
        ids = []
        if add_special_tokens:
            ids.append(self.vocab_to_index["<pad>"])  # BOS
        for value in values:
            if isinstance(value, (float, int)):
                if math.isnan(value):
                    # torch.bucketize would silently bin NaN into the top SASA range.
                    raise ValueError(
                        "Cannot tokenize NaN SASA value. Use None or "
                        f"'{self.mask_token}' to mask a position."
                    )
                bucket = torch.bucketize(value, torch.tensor(self._boundaries))
                token_id = len(self.special_tokens) + bucket
            elif isinstance(value, str):
                token_id = self.vocab_to_index[value]
            else:
                raise TypeError(value)
            ids.append(token_id)
        if add_special_tokens:
            ids.append(self.vocab_to_index["<pad>"])  # EOS

        return torch.tensor(ids, dtype=torch.int64)

    def decode_float(self, encoded: torch.Tensor) -> list[float | str | None]:
        """Decodes SASA token ids into range midpoints.

        Special tokens have no midpoint: `<pad>` decodes to None and the others to
        their vocab string, so that the result re-encodes to the tokens it came from.
        """
        decoded: list[float | str | None] = []
        for token_id in encoded.cpu().tolist():
            if token_id == self.mask_token_id:
                decoded.append(None)
            elif token_id < len(self.special_tokens):
                decoded.append(self.vocab[token_id])
            else:
                decoded.append(self.midpoints_tensor[token_id].item())
        return decoded

    def decode(self, encoded: torch.Tensor) -> str:
        """Decodes SASA token ids."""
        return ",".join(self.vocab[i] for i in encoded)

    def decode_list(self, encoded: torch.Tensor) -> list[str]:
        """Decodes SASA token ids."""
        return [self.vocab[i] for i in encoded]

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
