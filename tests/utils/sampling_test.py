import pytest
import torch

from esm.utils.sampling import sample_logits, top_p_logits


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


def test_top_p_logits_reaches_threshold():
    # top-p keeps the smallest set of tokens whose cumulative probability reaches
    # top_p, including the token that crosses the threshold; the kept mass must be
    # >= top_p, not < top_p.
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    logits = torch.log(probs).unsqueeze(0)
    for top_p, expected_kept in [(0.8, [0, 1, 2]), (0.7, [0, 1]), (0.35, [0])]:
        out = top_p_logits(logits.clone(), top_p=top_p)
        kept = (out.squeeze(0) > torch.finfo(out.dtype).min).nonzero().squeeze(-1)
        assert kept.tolist() == expected_kept
        assert probs[kept].sum().item() >= top_p - 1e-6


test_sample_logits()
test_top_p_logits_reaches_threshold()
