"""Tests for misc.py"""

from esm.utils.misc import merge_annotations
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
