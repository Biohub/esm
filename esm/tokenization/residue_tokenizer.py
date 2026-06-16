from functools import cached_property
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from cloudpathlib import AnyPath

from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C

Sample = dict[str, Any]


class ResidueAnnotationsTokenizer(EsmTokenizerBase):
    def __init__(self, csv_path: str | None = None, max_annotations: int = 16):
        if csv_path is None:
            csv_path = str(C.data_root("esm3") / C.RESID_CSV)
        self.csv_path = csv_path
        self.max_annotations = max_annotations

    @cached_property
    def _description2label(self) -> dict[str, str]:
        with AnyPath(self.csv_path).open() as f:
            df = pd.read_csv(f)
        return dict(zip(df.label, df.label_clean))

    @cached_property
    def _labels(self) -> list[str]:
        with AnyPath(self.csv_path).open() as f:
            df = pd.read_csv(f)
        labels = (
            df.groupby("label_clean")["count"]
            .sum()
            .sort_values(ascending=False, kind="stable")
            .index.tolist()
        )
        assert isinstance(labels, list)
        return labels

    def _description2id(self, description: str) -> int | None:
        label = self._description2label.get(description)
        return self._label2id.get(label)

    @cached_property
    def _label2id(self) -> dict[str, int]:
        offset = len(self.special_tokens) + 1  # +1 for "<none>"
        return {label: offset + i for i, label in enumerate(self._labels)}

    @cached_property
    def special_tokens(self) -> list[str]:
        """List of special tokens which come before cluster toknes in vocab."""
        return ["<pad>", "<motif>", "<unk>"]

    @cached_property
    def vocab(self):
        annotation_tokens = [f"<ra:{id}>" for _, id in self._label2id.items()]
        return self.special_tokens + ["<none>"] + annotation_tokens

    @cached_property
    def vocab_to_index(self) -> dict[str, int]:
        return {token: token_id for token_id, token in enumerate(self.vocab)}

    @cached_property
    def vocabulary(self) -> list[str]:
        """Full vocabulary."""
        return [*self.special_tokens, "<none>", *self._labels]

    def get_special_tokens_mask(self, encoded: torch.Tensor) -> torch.Tensor:
        """Determines where in the sequence are special tokens."""
        return encoded[:, 0] < len(self.special_tokens)

    def tokenize(
        self, sample: Sample | None, sequence: str, fail_on_mismatch: bool = False
    ) -> list[str]:
        """
        # interpro_site_starts
        # interpro_site_ends  # should always == interpro_site_starts.  but I haven't checked overall.
        # interpro_site_residues  # the residue identity of the specfic residue that is annotated.  good for a sanity check that parsing occurred correctly.
        # interpro_site_descriptions
        # ASSERT (i.e. drop if bad)
        # interpro_site_residues matches the residue at that position
        # all these lists ^ above are the same length
        """
        seqlen = len(sequence)
        assert seqlen >= 0
        # None mean sequence is *not annotated* - so use full <pad>
        if sample is None:
            return ["<pad>"] * seqlen

        if any(
            sample.get(field) is None
            for field in [
                "interpro_site_descriptions",
                "interpro_site_starts",
                "interpro_site_ends",
                "interpro_site_residues",
            ]
        ):
            return ["<pad>"] * seqlen

        num_annotations = len(sample["interpro_site_descriptions"])
        if any(
            len(sample[field]) != num_annotations
            for field in [
                "interpro_site_starts",
                "interpro_site_ends",
                "interpro_site_residues",
            ]
        ):
            # mismatched length.
            return ["<pad>"] * seqlen

        positional_ids = [set() for _ in range(seqlen)]
        for description, start, end, residues in zip(
            sample["interpro_site_descriptions"],
            sample["interpro_site_starts"],
            sample["interpro_site_ends"],
            sample["interpro_site_residues"],
        ):
            try:
                start = int(start)
                end = int(end)
            except (TypeError, ValueError):
                continue

            # Start / End are 1-indexed [inclusive, inclusive].
            if start <= 0 or end > seqlen or start > end:
                print(f"invalid start/end: ({start}, {end}), len: {seqlen}")
                continue

            if len(residues) != (end - start) + 1:
                print(f"bad reference residue: {residues}")
                continue

            token_id = self._description2id(description)
            if token_id is None:
                token_id = self.vocab_to_index["<unk>"]

            for i, residue in zip(range(start - 1, end), residues):
                # If there are any mismatching residues, skip the entire sample.
                if sequence[i] != residue:
                    if fail_on_mismatch:
                        raise ValueError(
                            f"Residue mismatch at position {i} (1-indexed): {sequence[i]} != {residue}"
                        )
                    return ["<pad>"] * seqlen

                positional_ids[i].add(token_id)

        tokens = []
        for token_ids in positional_ids:
            if token_ids:
                token = "<ra:" + ",".join(str(token_id) for token_id in token_ids) + ">"
            else:
                token = "<none>"
            tokens.append(token)
        return tokens

    def _token2ids(self, token: str) -> list[int]:
        if token.startswith("<ra:") and token.endswith(">"):
            return [int(token_id) for token_id in token[4:-1].split(",")]
        else:
            token_id = self.vocab_to_index[token]
            return [token_id]

    def encode(
        self, tokens: list[str], add_special_tokens: bool = True
    ) -> torch.Tensor:
        token_ids = torch.full(
            size=(len(tokens), self.max_annotations),
            dtype=torch.int64,
            fill_value=self.vocab_to_index["<pad>"],
        )
        for i, token in enumerate(tokens):
            ids = self._token2ids(token)[: self.max_annotations]
            token_ids[i, : len(ids)] = torch.tensor(ids)

        if add_special_tokens:
            token_ids = F.pad(
                token_ids, (0, 0, 1, 1), value=self.vocab_to_index["<pad>"]
            )
        return token_ids

    def decode(self, encoded: torch.Tensor) -> list[str]:
        raise NotImplementedError(
            "Residue annotation decoding should be handled with util.decoding.decode_residue_annotations"
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
