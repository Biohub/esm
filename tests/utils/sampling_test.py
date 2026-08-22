import pytest
import torch

from esm.utils.sampling import sample_logits


def test_sample_logits():
    # batched input. temperature != 0.0.
    sampled = sample_logits(
        logits=torch.randn((64, 8, 4096)), temperature=0.8, valid_ids=list(range(4096))
    )
    assert sampled.shape == (64, 8)

    # batched input. temperature == 0.0.
    sampled = sample_logits(
        logits=torch.randn((64, 8, 4096)), temperature=0.0, valid_ids=list(range(4096))
    )
    assert sampled.shape == (64, 8)

    # non-batched input. temperature != 0.0.
    sampled = sample_logits(
        logits=torch.randn((8, 4096)), temperature=0.8, valid_ids=list(range(4096))
    )
    assert sampled.shape == (8,)

    # non-batched input. temperature == 0.0.
    sampled = sample_logits(
        logits=torch.randn((8, 4096)), temperature=0.0, valid_ids=list(range(4096))
    )
    assert sampled.shape == (8,)

    with pytest.raises(ValueError):
        sampled = sample_logits(
            logits=torch.randn((8, 4096)), temperature=0.0, valid_ids=[]
        )


test_sample_logits()
