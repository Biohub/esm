"""Tests for misc.py"""

import json

import numpy as np
import pytest
import torch

from esm.utils.misc import maybe_list, merge_annotations
from esm.utils.types import FunctionAnnotation


def test_merge_annotations():
    merged = merge_annotations(
        [
            FunctionAnnotation("a", start=1, end=10),
            FunctionAnnotation("b", start=5, end=15),
            FunctionAnnotation("a", start=10, end=20),
            FunctionAnnotation("b", start=2, end=6),
            FunctionAnnotation("c", start=4, end=10),
        ]
    )
    assert len(merged) == 3
    assert FunctionAnnotation("a", start=1, end=20) in merged
    assert FunctionAnnotation("b", start=2, end=15) in merged
    assert FunctionAnnotation("c", start=4, end=10) in merged


def test_merge_annotations_gap():
    merged = merge_annotations(
        [
            FunctionAnnotation("a", start=1, end=10),
            FunctionAnnotation("a", start=13, end=20),  # gap is 2
            FunctionAnnotation("a", start=24, end=30),
        ],
        merge_gap_max=2,
    )

    assert len(merged) == 2
    assert FunctionAnnotation("a", 1, 20) in merged
    assert FunctionAnnotation("a", 24, 30) in merged


class TestRoundDecimals:
    def test_rounding_happens_in_float64(self):
        x = np.array([4.937121868133545], dtype=np.float32)
        assert json.dumps(maybe_list(x, round_decimals=2)) == "[4.94]"

    def test_a_torch_input_rounds_like_a_numpy_one(self):
        x = np.array([[4.937121868133545, 9.927359580993652]], dtype=np.float32)
        assert maybe_list(torch.from_numpy(x), round_decimals=2) == maybe_list(
            x, round_decimals=2
        )

    def test_rounding_composes_with_convert_nan_to_none(self):
        x = np.array([4.937121868133545, np.nan], dtype=np.float32)
        assert maybe_list(x, convert_nan_to_none=True, round_decimals=2) == [4.94, None]

    def test_none_is_still_none(self):
        assert maybe_list(None, round_decimals=2) is None

    def test_values_are_untouched_by_default(self):
        # Every existing caller passes no round_decimals and must keep full precision.
        x = np.array([4.937121868133545], dtype=np.float32)
        assert maybe_list(x) == x.tolist()

    @pytest.mark.parametrize("decimals", [0, 1, 3])
    def test_the_decimal_count_is_honoured(self, decimals):
        x = np.array([4.937121868133545], dtype=np.float32)
        assert maybe_list(x, round_decimals=decimals) == [
            round(4.937121868133545, decimals)
        ]
