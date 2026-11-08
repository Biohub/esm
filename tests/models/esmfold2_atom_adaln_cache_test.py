"""The atom blocks' adaLN modulation is computed once per fold, not per step.

``c_l`` is the diffusion conditioning and is fixed for a whole fold, so every
diffusion step recomputed the identical SiLU + Linear in all six atom blocks.
Profiled at 82 ms of a 6687 ms production fold on H100.

The cache is keyed on the identity of ``c_l``, so these tests pin both halves of
that contract: it must hit when the tensor is the same object, and it must
recompute when it is not. It is bit-exact either way -- a cache that changed a
value would be a bug, not a trade.

Identity keying only pays if the producer stops rebuilding ``c_l``. It did:
``repeat_interleave`` allocated a fresh tensor of identical values on every
step, so the slot missed 84 times out of 84 and bought nothing. The expanded
conditioning is now cached too, and ``test_expanded_conditioning_is_reused``
pins that, because without it this whole lever is dead code.
"""

import pytest
import torch

from esm.models.esmfold2.layers import SWAAtomBlock

D_ATOM = 32


def _block() -> SWAAtomBlock:
    torch.manual_seed(0)
    blk = SWAAtomBlock(d_atom=D_ATOM, n_heads=2, half_window=4).eval()
    # adaln_modulation is zero-init, which would make any cache look correct.
    with torch.no_grad():
        blk.adaln_modulation[1].weight.normal_(0.0, 0.05)  # ty:ignore[call-non-callable]
    return blk


def test_cache_is_bitwise_and_hits() -> None:
    blk = _block()
    c_l = torch.randn(1, 8, D_ATOM)
    cache: dict = {}
    with torch.inference_mode():
        first = blk._adaln_modulation_cached(c_l, cache)
        second = blk._adaln_modulation_cached(c_l, cache)
        assert second is first, "the second call recomputed"
        assert torch.equal(first, blk.adaln_modulation(c_l))


def test_a_different_conditioning_tensor_recomputes() -> None:
    """Identity-keyed: a new tensor object must not be served the stale slot.

    Two cases, because one assertion cannot catch both failures. An
    equal-valued clone is caught only by object identity -- a stale hit would
    return numerically correct values. A different-valued tensor is caught by
    the values, which is what a reader expects the test to check.
    """
    blk = _block()
    c_l = torch.randn(1, 8, D_ATOM)
    cache: dict = {}
    with torch.inference_mode():
        first = blk._adaln_modulation_cached(c_l, cache)

        # Equal values, distinct object: only identity can tell these apart.
        clone = c_l.clone()
        got_clone = blk._adaln_modulation_cached(clone, cache)
        assert got_clone is not first, "an equal-valued clone served the stale slot"
        assert torch.equal(got_clone, blk.adaln_modulation(clone))

        # Different values: a stale hit is now numerically visible too.
        shifted = c_l + 1.0
        got_shifted = blk._adaln_modulation_cached(shifted, cache)
        assert torch.equal(got_shifted, blk.adaln_modulation(shifted))
        assert not torch.equal(got_shifted, first)


def test_two_blocks_do_not_share_a_slot() -> None:
    """The slot is keyed by module instance; blocks have different weights."""
    a, b = _block(), _block()
    with torch.no_grad():
        b.adaln_modulation[1].weight.mul_(-1.0)  # ty:ignore[call-non-callable]
    c_l = torch.randn(1, 8, D_ATOM)
    cache: dict = {}
    with torch.inference_mode():
        out_a = a._adaln_modulation_cached(c_l, cache)
        out_b = b._adaln_modulation_cached(c_l, cache)
    assert not torch.equal(out_a, out_b)
    assert torch.equal(out_b, b.adaln_modulation(c_l))


def test_no_cache_and_grad_both_bypass() -> None:
    blk = _block()
    c_l = torch.randn(1, 8, D_ATOM)
    with torch.inference_mode():
        assert torch.equal(
            blk._adaln_modulation_cached(c_l, None), blk.adaln_modulation(c_l)
        )
    grad_c = c_l.clone().requires_grad_(True)
    cache: dict = {}
    out = blk._adaln_modulation_cached(grad_c, cache)
    assert out.requires_grad
    assert cache == {}, "the cache captured an autograd graph"


@pytest.mark.gpu
def test_block_forward_is_bitwise_with_and_without_the_cache() -> None:
    """The whole point: the cache must not move a single bit of the block."""
    import torch.nn.functional as F

    from esm.models.esmfold2.layers import SWAAtomTransformer

    # Production atom shape: head_dim must be >= the 32-wide 3D RoPE
    # (3 axes x 2 spatial pairs + 10 uid pairs). d_atom=128 / 4 heads gives 32.
    d_atom, n_heads = 128, 4
    torch.manual_seed(0)
    tf = SWAAtomTransformer(d_atom=d_atom, n_blocks=1, n_heads=n_heads).cuda().eval()
    blk = tf.blocks[0]
    with torch.no_grad():
        blk.adaln_modulation[1].weight.normal_(0.0, 0.05)  # ty:ignore[not-subscriptable, unresolved-attribute]

    torch.manual_seed(1)
    b, a = 1, 32
    x = torch.randn(b, a, d_atom, device="cuda")
    c_l = torch.randn(b, a, d_atom, device="cuda")
    mask = torch.ones(b, a, dtype=torch.bool, device="cuda")
    cos, sin = tf._build_3d_rope(
        torch.randn(b, a, 3, device="cuda"),
        torch.zeros(b, a, dtype=torch.long, device="cuda"),
    )
    seqlens = mask.sum(dim=-1, dtype=torch.int32)
    params = (
        cos,
        sin,
        torch.nonzero(mask.flatten(), as_tuple=False).flatten(),
        F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)),
        int(seqlens.max().item()),
    )

    with torch.inference_mode():
        ref = blk(x, c_l, params)
        got = blk(x, c_l, params, inference_cache={})
        # a second step off a warm slot: the case that actually pays
        cache: dict = {}
        blk(x, c_l, params, inference_cache=cache)
        warm = blk(x, c_l, params, inference_cache=cache)

    assert torch.equal(got, ref)
    assert torch.equal(warm, ref)


@pytest.mark.gpu
def test_expanded_conditioning_is_reused() -> None:
    """The producer must hand back one object per fold, not one per step.

    repeat_interleave allocates. If this regresses, the adaLN slot above misses
    on every diffusion step and the lever silently stops paying.
    """
    from esm.models.esmfold2.layers import EsmFold2AtomEncoder

    torch.manual_seed(0)
    enc = EsmFold2AtomEncoder(d_atom=128, d_token=256, n_blocks=1, n_heads=4)
    enc = enc.cuda().eval()
    b, n = 1, 24
    kw = dict(
        ref_pos=torch.randn(b, n, 3, device="cuda"),
        atom_attention_mask=torch.ones(b, n, dtype=torch.bool, device="cuda"),
        ref_space_uid=torch.zeros(b, n, dtype=torch.long, device="cuda"),
        ref_charge=torch.zeros(b, n, device="cuda"),
        ref_element=torch.zeros(b, n, 128, device="cuda"),
        ref_atom_name_chars=torch.zeros(b, n, 4, 64, device="cuda"),
        atom_to_token=torch.arange(n, device="cuda").unsqueeze(0) // 4,
    )
    cache: dict = {}
    with torch.inference_mode():
        enc(**kw, inference_cache=cache)
        first = cache["atomencoder"]["c_expanded"]
        enc(**kw, inference_cache=cache)
        assert cache["atomencoder"]["c_expanded"] is first
