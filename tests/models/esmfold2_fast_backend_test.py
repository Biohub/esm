"""End-to-end contracts of the approximate kernels behind ``set_kernel_backend("fast")``."""

import pytest
import torch
import torch.nn.functional as F

from esm.models.esmfold2 import layers as L
from esm.models.esmfold2.kernels import fused_dit
from esm.models.esmfold2.layers import (
    DiffusionTransformer,
    PairUpdateBlock,
    SWAAtomTransformer,
    Transition,
)

D_ATOM, ATOM_HEADS, ATOM_BLOCKS = 128, 4, 3
D_TOKEN, D_PAIR, DIT_HEADS, DIT_BLOCKS = 768, 256, 16, 2
SEQ_LEN, N_SAMPLES = 65, 3
ATOM_KERNELS = (
    "_fused_rms_adaln",
    "_fused_gated_residual_rms_adaln",
    "_fused_swa_attention",
    "_fused_swiglu_ffn",
    "_fused_qk_norm_rope",
)
DIT_KERNELS = ("adaln", "gate_residual_adaln", "swiglu")


def _count(monkeypatch) -> dict[str, int]:
    """Approximate-kernel calls that RETURNED, keyed by name; ``name/bf16`` counts bf16 operands."""
    done: dict[str, int] = {}
    targets = [(L, n) for n in ATOM_KERNELS] + [(fused_dit, n) for n in DIT_KERNELS]
    if L._t16 is not None:
        targets.append((L._t16, "transition_cute"))
    if L._tx is not None:
        targets.append((L._tx, "forward"))
    for mod, name in targets:

        def counted(*args, _orig=getattr(mod, name), _name=name, **kwargs):
            out = _orig(*args, **kwargs)
            done[_name] = done.get(_name, 0) + 1
            if args and getattr(args[0], "dtype", None) is torch.bfloat16:
                done[f"{_name}/bf16"] = done.get(f"{_name}/bf16", 0) + 1
            return out

        monkeypatch.setattr(mod, name, counted)
    return done


def _live(module: torch.nn.Module) -> None:
    with (
        torch.no_grad()
    ):  # zero-init adaLN and out gates would make these paths identities
        for m in module.modules():
            if isinstance(m, torch.nn.Linear) and m.weight.abs().max() == 0:
                m.weight.normal_(0.0, 0.02)


def _atom(seed: int = 0) -> SWAAtomTransformer:
    torch.manual_seed(seed)
    tf = SWAAtomTransformer(d_atom=D_ATOM, n_blocks=ATOM_BLOCKS, n_heads=ATOM_HEADS)
    _live(tf)
    return tf.cuda().eval()


def _dit(seed: int = 0) -> DiffusionTransformer:
    torch.manual_seed(seed)
    tf = DiffusionTransformer(
        d_model=D_TOKEN,
        d_pair=D_PAIR,
        num_heads=DIT_HEADS,
        num_blocks=DIT_BLOCKS,
        d_cond=D_TOKEN,
        transition_multiplier=2,
    )
    _live(tf)
    return tf.cuda().eval()


def _atom_mask(hole: bool = False) -> torch.Tensor:
    mask = torch.ones(2, 256, dtype=torch.bool, device="cuda")
    mask[1, 216:] = False
    if hole:
        mask[0, 40:50] = False
    return mask


def _atom_inputs(tf: SWAAtomTransformer, mask: torch.Tensor) -> dict:
    torch.manual_seed(1)
    b, a = mask.shape
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
    return {
        "q_l": torch.randn(b, a, D_ATOM, device="cuda"),
        "c_l": torch.randn(b, a, D_ATOM, device="cuda"),
        "attention_params": params,
    }


def _dit_inputs():
    """``(a, s_per_input, z, mask)``; "fast" takes s once per input, "fused" per sample."""
    torch.manual_seed(1)
    a = torch.randn(N_SAMPLES, SEQ_LEN, D_TOKEN, device="cuda")
    s1 = torch.randn(1, SEQ_LEN, D_TOKEN, device="cuda")
    z = torch.randn(1, SEQ_LEN, SEQ_LEN, D_PAIR, device="cuda") * 0.1
    return a, s1, z, torch.ones(1, SEQ_LEN, dtype=torch.bool, device="cuda")


def _atom_run(tf: SWAAtomTransformer, backend, inputs: dict) -> torch.Tensor:
    tf.set_kernel_backend(backend)
    with torch.inference_mode():
        return tf(**inputs, inference_cache={})


def _dit_run(tf: DiffusionTransformer, backend, a, s, z, mask) -> torch.Tensor:
    tf.set_kernel_backend(backend)
    with torch.inference_mode():
        return tf(
            a,
            s,
            z,
            beta=0.0,
            attention_mask=mask,
            num_diffusion_samples=a.shape[0] // s.shape[0],
            inference_cache={},
        )[0]


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    scale = ref.double().abs().max().clamp_min(1e-6)
    return ((got.double() - ref.double()).abs().max() / scale).item()


@pytest.mark.gpu
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16_switch"])
def test_fast_tracks_fused_on_the_diffusion_path(monkeypatch, bf16) -> None:
    """Every "fast" kernel completes and stays close to "fused"; none fires under "fused"."""
    monkeypatch.setattr(L, "BF16_DIT_GEMM", False)
    done = _count(monkeypatch)
    atom, dit = _atom(), _dit()
    atom_in = _atom_inputs(atom, _atom_mask())
    a, s1, z, mask = _dit_inputs()
    ref = {
        "atom": _atom_run(atom, "fused", atom_in),
        "dit": _dit_run(dit, "fused", a, s1.repeat_interleave(N_SAMPLES, 0), z, mask),
    }
    assert not done, f"approximate kernels fired under 'fused': {done}"

    monkeypatch.setattr(L, "BF16_DIT_GEMM", bf16)
    got = {
        "atom": _atom_run(atom, "fast", atom_in),
        "dit": _dit_run(dit, "fast", a, s1, z, mask),
    }
    want = {
        "_fused_swa_attention": ATOM_BLOCKS,
        "_fused_swiglu_ffn": ATOM_BLOCKS,
        "adaln": 1,
        "gate_residual_adaln": 2 * DIT_BLOCKS,
        "swiglu": DIT_BLOCKS,
    }
    for name, n in want.items():
        assert done.get(name) == n, f"{name}: {done.get(name, 0)} of {n} completed"
    for name in ("_fused_swiglu_ffn", "swiglu"):
        assert done.get(f"{name}/bf16", 0) == (want[name] if bf16 else 0), done
    tol = 4e-3 if bf16 else 1e-3
    for name, out in got.items():
        assert torch.isfinite(out).all(), f"{name}: non-finite output"
        rel = _rel(out, ref[name])
        assert rel < tol, f"{name} drifted from 'fused': relative max delta {rel:.3e}"


@pytest.mark.gpu
def test_fast_serves_new_weights_after_a_reload(monkeypatch) -> None:
    """Per-fold bf16 weight copies must follow a ``load_state_dict``."""
    monkeypatch.setattr(L, "BF16_DIT_GEMM", True)
    done = _count(monkeypatch)
    atom_in = _atom_inputs(_atom(), _atom_mask())
    a, s1, z, mask = _dit_inputs()
    cases = {
        "atom": (_atom, lambda m: _atom_run(m, "fast", atom_in)),
        "dit": (_dit, lambda m: _dit_run(m, "fast", a, s1, z, mask)),
    }
    for name, (make, run) in cases.items():
        model, other = make(0), make(1)
        before = run(model)
        model.load_state_dict(other.state_dict())
        after, want = run(model), run(other)
        assert not torch.equal(before, want), f"{name}: new weights do not change it"
        assert torch.equal(after, want), f"{name}: served stale weights after a reload"
    assert done.get("_fused_swiglu_ffn/bf16") and done.get("swiglu/bf16"), done


@pytest.mark.gpu
def test_training_never_runs_an_approximate_kernel(monkeypatch) -> None:
    """With autograd recording, "fast" must run what "fused" runs and give its gradients."""
    monkeypatch.setattr(L, "BF16_DIT_GEMM", False)
    done = _count(monkeypatch)
    n_pair = L._TX_MIN_TOKENS + 63

    def grads(backend) -> list:
        atom, dit = _atom(), _dit()
        torch.manual_seed(2)
        blk = PairUpdateBlock(d_pair=D_PAIR).cuda()
        for m in (atom, dit, blk):
            m.set_kernel_backend(backend)
        inputs = _atom_inputs(atom, _atom_mask())
        q = inputs.pop("q_l").requires_grad_(True)
        a, s1, z, mask = _dit_inputs()
        a.requires_grad_(True)
        pair = torch.randn(
            1, n_pair, n_pair, D_PAIR, device="cuda", dtype=torch.bfloat16
        ).requires_grad_(True)
        s_n = s1.repeat_interleave(N_SAMPLES, 0)
        loss = (
            atom(q_l=q, **inputs).pow(2).mean()
            + dit(
                a,
                s_n,
                z,
                beta=0.0,
                attention_mask=mask,
                num_diffusion_samples=N_SAMPLES,
            )[0]
            .pow(2)
            .mean()  # fmt: skip
            + blk(pair).float().pow(2).mean()
        )
        loss.backward()
        params = [p.grad for m in (atom, dit, blk) for p in m.parameters()]
        return [q.grad, a.grad, pair.grad, *params]

    ref = grads("fused")
    got = grads("fast")
    assert not done, f"approximate kernels ran with autograd recording: {done}"
    for i, (g, r) in enumerate(zip(got, ref, strict=True)):
        assert (g is None) == (r is None), f"gradient {i}: presence differs"
        if r is not None:
            assert _rel(g, r) < 1e-3, f"gradient {i}: relative delta {_rel(g, r):.3e}"


@pytest.mark.gpu
def test_opt_core_kernels_track_fp32_and_follow_new_weights(monkeypatch) -> None:
    """``t16`` and ``tx`` stay within 2x the Triton error and serve reloaded weights."""
    if not (L.T16_AVAILABLE and L.TX_AVAILABLE):
        pytest.skip("t16/tx need opt_core")  # ty:ignore[too-many-positional-arguments]
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("t16/tx need an sm_90 GPU")  # ty:ignore[too-many-positional-arguments]
    done = _count(monkeypatch)
    n = L._TX_MIN_TOKENS + 63
    torch.manual_seed(1)
    x = torch.randn(1, n, n, D_PAIR, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, n, n, device="cuda")
    mask[:, -5:, :] = 0
    mask[:, :, -5:] = 0

    def transition(seed: int) -> Transition:
        torch.manual_seed(seed)
        t = Transition(D_PAIR).cuda().to(torch.bfloat16).eval()
        t.set_chunk_size(None)
        return t

    def transition_ref(t: Transition) -> torch.Tensor:
        fp32 = Transition(D_PAIR).cuda().eval()
        fp32.load_state_dict(t.state_dict())
        fp32.set_chunk_size(None)
        return fp32(x.float())

    def pair_block(seed: int) -> PairUpdateBlock:
        torch.manual_seed(seed)
        blk = PairUpdateBlock(d_pair=D_PAIR).cuda().eval()
        with torch.no_grad():
            for p in blk.parameters():
                if p.dim() == 2:
                    p.normal_(0.0, 0.05)
        return blk

    def run(model, backend, direction):
        model.set_kernel_backend(backend)
        if direction is None:
            return model(x)
        return model._fused_trimul_with_residual(x, direction, mask)

    def reference(model, direction):
        if direction is None:
            return transition_ref(model)
        tri = model.tri_mul_out if direction == "outgoing" else model.tri_mul_in
        return x.float() + tri(x.float(), mask=mask)

    cases = [("t16", transition, None), ("tx", pair_block, "outgoing")]
    cases.append(("tx", pair_block, "incoming"))
    for name, make, direction in cases:
        model, other = make(0), make(1)
        with torch.inference_mode():
            ref = reference(model, direction)
            err_triton = (
                (run(model, "fused", direction).float() - ref).abs().max().item()
            )
            err = (run(model, "fast", direction).float() - ref).abs().max().item()
            before = run(model, "fast", direction)
        assert err <= 2 * err_triton + 1e-6, (
            f"{name}: {err:.3e} vs Triton {err_triton:.3e}"
        )
        model.load_state_dict(other.state_dict())
        with torch.inference_mode():
            after, want = run(model, "fast", direction), run(other, "fast", direction)
        assert not torch.equal(before, want), f"{name}: new weights do not change it"
        assert torch.equal(after, want), f"{name}: served stale weights after a reload"
    assert done.get("transition_cute", 0) >= 3 and done.get("forward", 0) >= 6, done


@pytest.mark.gpu
def test_refused_inputs_fall_back_exactly(monkeypatch) -> None:
    """Autocast, a holed atom layout and a refusing kernel must not cost correctness."""
    monkeypatch.setattr(L, "BF16_DIT_GEMM", False)
    atom, dit = _atom(), _dit()
    a, s1, z, mask = _dit_inputs()
    s_n = s1.repeat_interleave(N_SAMPLES, 0)

    atom_in = _atom_inputs(atom, _atom_mask())
    bf = {k: v.bfloat16() if k != "attention_params" else v for k, v in atom_in.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = _atom_run(atom, "fast", bf)
        assert torch.equal(got, _atom_run(atom, None, bf)), "atom: autocast fallback"
        dit_bf = (a.bfloat16(), s_n.bfloat16(), z.bfloat16(), mask)
        got = _dit_run(dit, "fast", *dit_bf)
        assert torch.equal(got, _dit_run(dit, "fused", *dit_bf)), (
            "dit: autocast fallback"
        )

    done = _count(monkeypatch)
    holed = _atom_inputs(atom, _atom_mask(hole=True))
    rel = _rel(_atom_run(atom, "fast", holed), _atom_run(atom, None, holed))
    assert rel < 1e-3, f"atom: holed layout drifted, relative max delta {rel:.3e}"
    assert not done.get("_fused_swa_attention"), "the prefix-only kernel served a hole"
    assert done.get("_fused_qk_norm_rope") == ATOM_BLOCKS, done

    def refuse(*_args, **_kwargs):
        raise L._AtomBlockKernelUnavailable("refused")

    def refuse_dit(*_args, **_kwargs):
        raise fused_dit.DitKernelUnavailable("refused")

    for name in ATOM_KERNELS:
        monkeypatch.setattr(L, name, refuse)
    monkeypatch.setattr(fused_dit, "gate_residual_adaln", refuse_dit)
    got = _atom_run(atom, "fast", atom_in)
    assert torch.equal(got, _atom_run(atom, None, atom_in)), "atom: refusal fallback"
    got = _dit_run(dit, "fast", a, s1, z, mask)
    assert torch.equal(got, _dit_run(dit, "fused", a, s_n, z, mask)), "dit: refusal"
