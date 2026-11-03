"""The per-fold pair-bias cache in ``AttentionPairBias``.

The pair bias is a projection of ``z`` into ``[B, H, Q, K]``. Within one fold
``z``, the mask and the projection weights are all fixed, so every diffusion
step of every DiT block used to recompute the same tensor. The cache stores it
in the caller's per-fold ``inference_cache``.

This is a caching change only, so every test here is a ``torch.equal`` test, not
an ``allclose`` one. The cache must also stay out of the way of autograd: the
binder-design path takes input gradients through these blocks with frozen
weights, and a detached cache hit would silently break it.
"""

import warnings

import pytest
import torch

from esm.models.esmfold2.layers import AttentionPairBias

_D_MODEL = 32
_D_PAIR = 16
_N_HEADS = 4
_N_TOKENS = 7


def _block(seed: int = 0) -> AttentionPairBias:
    torch.manual_seed(seed)
    block = AttentionPairBias(
        d_model=_D_MODEL, d_pair=_D_PAIR, num_heads=_N_HEADS, use_conditioning=False
    ).eval()
    # Random weights: the default LayerNorm/Linear init would make several
    # tensors identical and hide a keying mistake.
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return block


def _inputs(batch: int = 1, seed: int = 1):
    torch.manual_seed(seed)
    a = torch.randn(batch, _N_TOKENS, _D_MODEL)
    z = torch.randn(batch, _N_TOKENS, _N_TOKENS, _D_PAIR)
    mask = torch.ones(batch, _N_TOKENS, dtype=torch.bool)
    return a, z, mask


class _ProjCounter:
    """Counts pair-bias projections, i.e. cache misses."""

    def __init__(self, block: AttentionPairBias) -> None:
        self.calls = 0
        self._handle = block.pair_bias_proj.register_forward_hook(self._hook)

    def _hook(self, *_args) -> None:
        self.calls += 1

    def close(self) -> None:
        self._handle.remove()


def test_cache_is_bitwise_and_projects_once():
    """Three cached calls give the uncached answer and one projection."""
    block = _block()
    a, z, mask = _inputs()

    with torch.no_grad():
        counter = _ProjCounter(block)
        reference = block(a, None, z, attention_mask=mask)
        assert counter.calls == 1

        cache: dict = {}
        outs = [
            block(a, None, z, attention_mask=mask, inference_cache=cache)
            for _ in range(3)
        ]
        counter.close()

    assert counter.calls == 2, "the cached calls projected more than once"
    for i, out in enumerate(outs):
        assert torch.equal(out, reference), f"cached call {i} is not bitwise equal"


def test_cache_holds_the_expanded_bias_for_several_diffusion_samples():
    """``z`` is expanded per sample inside the block; the cache holds the result."""
    block = _block()
    samples = 3
    a_base, z, mask = _inputs(batch=1)
    a = a_base.repeat_interleave(samples, dim=0)

    with torch.no_grad():
        counter = _ProjCounter(block)
        reference = block(
            a, None, z, attention_mask=mask, num_diffusion_samples=samples
        )
        cache: dict = {}
        first = block(
            a,
            None,
            z,
            attention_mask=mask,
            num_diffusion_samples=samples,
            inference_cache=cache,
        )
        second = block(
            a,
            None,
            z,
            attention_mask=mask,
            num_diffusion_samples=samples,
            inference_cache=cache,
        )
        counter.close()

    assert counter.calls == 2
    assert torch.equal(first, reference)
    assert torch.equal(second, reference)
    cached = cache["pair_bias"][block]["eager"]
    assert cached.shape[0] == samples


def test_two_blocks_sharing_one_cache_do_not_collide():
    """The slot is keyed by module instance, so different weights stay apart."""
    first, second = _block(seed=0), _block(seed=1)
    a, z, mask = _inputs()

    with torch.no_grad():
        ref_first = first(a, None, z, attention_mask=mask)
        ref_second = second(a, None, z, attention_mask=mask)
        cache: dict = {}
        got_first = first(a, None, z, attention_mask=mask, inference_cache=cache)
        got_second = second(a, None, z, attention_mask=mask, inference_cache=cache)

    assert not torch.equal(ref_first, ref_second), "the two blocks are not distinct"
    assert torch.equal(got_first, ref_first)
    assert torch.equal(got_second, ref_second)


def test_grad_enabled_alone_refuses_the_cache_and_warns():
    """First guard, in isolation: frozen weights, no grad-tracking input."""
    block = _block().requires_grad_(False)
    a, z, mask = _inputs()
    cache: dict = {}

    assert torch.is_grad_enabled()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        block(a, None, z, attention_mask=mask, inference_cache=cache)

    assert "pair_bias" not in cache, "the cache engaged under grad"
    assert any("pair-bias cache not engaged" in str(w.message) for w in caught), (
        "refusing to cache was silent"
    )


def test_a_grad_tracking_input_alone_refuses_the_cache():
    """Second guard, in isolation: no grad context, grad-tracking ``z``."""
    block = _block().requires_grad_(False)
    a, z_base, mask = _inputs()
    z = z_base.clone().requires_grad_(True)

    with torch.no_grad():
        assert block._pair_bias_cache_slot({}, z_base, mask, a.shape[0], 1) is not None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            slot = block._pair_bias_cache_slot({}, z, mask, a.shape[0], 1)

    assert slot is None, "the cache engaged on a grad-tracking z"
    assert any("pair-bias cache not engaged" in str(w.message) for w in caught)


def test_the_input_gradient_is_unchanged_when_a_cache_is_offered():
    """The binder-design path: frozen weights, gradient w.r.t. the pair input."""
    block = _block().requires_grad_(False)
    a, z_base, mask = _inputs()

    def input_grad(cache: dict | None) -> torch.Tensor:
        z = z_base.clone().requires_grad_(True)
        out = block(a, None, z, attention_mask=mask, inference_cache=cache)
        return torch.autograd.grad(out.square().sum(), z)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = input_grad(None)
        cache: dict = {}
        with_cache = input_grad(cache)
        again = input_grad(cache)

    assert "pair_bias" not in cache
    assert torch.equal(with_cache, reference), "the input gradient changed"
    assert torch.equal(again, reference), "a second grad call diverged"


def test_nothing_survives_the_caller_dropping_the_cache():
    """All state lives in the caller's dict; the module keeps none of it."""
    block = _block()
    a, z, mask = _inputs()
    before = set(block.__dict__)

    with torch.no_grad():
        cache: dict = {}
        block(a, None, z, attention_mask=mask, inference_cache=cache)

    assert cache["pair_bias"][block]["eager"] is not None
    assert set(block.__dict__) == before, "the block grew per-fold state"

    with torch.no_grad():
        fresh: dict = {}
        counter = _ProjCounter(block)
        block(a, None, z, attention_mask=mask, inference_cache=fresh)
        counter.close()
    assert counter.calls == 1, "a new cache dict reused an old bias"


@pytest.mark.gpu
def test_fused_path_cache_is_bitwise():
    """Same check on the fused Triton kernel, which is the hot path."""
    block = _block().cuda()
    a, z, mask = _inputs()
    a, z, mask = a.cuda(), z.cuda(), mask.cuda()
    block.set_kernel_backend("fused")

    with torch.no_grad():
        assert block._can_use_fused_pair_bias(z, _N_TOKENS, 0.0), (
            "the fused pair-bias path did not engage"
        )
        reference = block(a, None, z, attention_mask=mask)
        cache: dict = {}
        first = block(a, None, z, attention_mask=mask, inference_cache=cache)
        second = block(a, None, z, attention_mask=mask, inference_cache=cache)

    assert "fused" in cache["pair_bias"][block]
    assert torch.equal(first, reference)
    assert torch.equal(second, reference)
    assert cache["pair_bias"][block]["fused"].shape == (
        1,
        _N_HEADS,
        _N_TOKENS,
        _N_TOKENS,
    )
