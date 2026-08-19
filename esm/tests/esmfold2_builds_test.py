"""ESMFold2 build configurations.

The axes a user can turn without changing the weights: kernel backend, chunk
size, flash-attention, device, precision and ``torch.compile``. Every case here
either asserts a *structural* fact about the module tree (which is what makes
``set_kernel_backend`` testable without a GPU) or holds a build against the
CPU / fp32 result - never against itself, because a build compared to its own
output passes when the whole configuration drifts together.

Two axes are deliberately absent. Context parallelism needs a ``distributed/``
tree this package does not ship. Backbone precision needs a bundled backbone, and
the tiny configs have none (``model.esmc is None``), so bf16 / fp32 / fp8
selection has no module to act on; it belongs with the published-weight tests.

Defects this file pins:

- ``set_kernel_backend`` never reaches ``msa_encoder``, so an MSA-enabled
  checkpoint runs four triangle multiplications on whatever backend they were
  constructed with. ``set_chunk_size`` on the release model does reach them.
- ``EsmFold2ExperimentalModel.set_chunk_size`` misses ``msa_encoder`` too.
- ``apply_torch_compile``'s "does NOT stack with our Triton kernels" docstring is
  a comment and nothing else: the method inspects nothing and warns nothing when
  a fused build is compiled.

Each is a strict ``xfail`` or an explicit expectation, so a fix flips the test
rather than going unnoticed.

The ``torch.compile`` cases are ``nightly``, not ``gpu``, because
``apply_torch_compile`` does not behave the same way on every GPU. On the CI
runner it compiles in both modes and reproduces eager. On our L40 and H100 dev
boxes the first forward dies in Inductor with ``PendingUnbackedSymbolNotFound``,
raised by the ``capture_scalar_outputs = True`` the method sets to avoid a graph
break at the ``.item()`` calls in the atom-attention path - measured at L=12 and
L=197, in both modes, with ``set_kernel_backend(None)``, so it is not a length,
a noise patch or a kernel-stacking effect.

An earlier revision pinned the dev-box failure as a strict ``xfail``. That
XPASSed on the CI GPU, which is a hard failure, so the required GPU job went red
on a claim about our workstations. Pinning the opposite outcome would just move
the redness. Until compilation behaves the same on both, this equivalence is not
a property a required gate can assert, so it runs on demand instead:

    pytest -m nightly esm/tests/esmfold2_builds_test.py

That keeps the assertion honest - compiled output must match eager - and keeps
an Inductor crash a real failure wherever it happens, without making either
platform's behaviour everyone's expectation.
"""

import contextlib
import warnings

import pytest
import torch
import torch._dynamo

from esm.models.esmfold2 import (
    EsmFold2Config,
    EsmFold2ExperimentalModel,
    EsmFold2Model,
)
from esm.models.esmfold2 import layers as _layers
from esm.tests.conftest import (
    ESMFOLD2_LENGTHS,
    ESMFOLD2_SEQUENCES,
    esmfold2_inputs,
)

#: Chunk sizes the matrix sweeps. ``None`` disables chunking entirely.
CHUNK_SIZES = (None, 16, 32, 64, 128)

#: The length the chunk matrix and the GPU builds run at. 197 residues is longer
#: than every chunk size above, so each one really splits the pair axis, and it
#: is prime, so every one of them also leaves a short remainder chunk - which is
#: where off-by-one bugs live.
MATRIX_LENGTH = "over_window"

# The claim the chunk-matrix docstring makes, asserted rather than asserted-by-
# comment: the 12-mer every other ESMFold2 test uses is shorter than the
# smallest chunk size, so on it ``set_chunk_size`` cannot change a single op.
assert ESMFOLD2_LENGTHS["tiny"] < min(c for c in CHUNK_SIZES if c is not None)
assert ESMFOLD2_LENGTHS[MATRIX_LENGTH] > max(c for c in CHUNK_SIZES if c is not None)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def report(label: str, value: float) -> float:
    """Print a measured delta so tolerances can be re-derived, not widened.

    Every tolerance in this file is a number somebody read off a run. Run with
    ``-s`` to see them again after a change.
    """
    print(f"[measured] {label}: {value:.3e}")
    return value


def max_abs_delta(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float().cpu() - expected.float().cpu()).abs().max())


@contextlib.contextmanager
def flash_attention(enabled: bool):
    """Force the SWA atom attention onto (or off) the flash-attn path.

    ``flash_attn`` imports fine on a CPU-only box but its kernels need a CUDA
    runtime, so the CPU reference has to run with the flag off even on a GPU
    machine. Read at forward time, so patching the module global is enough.
    """
    previous = _layers.FLASH_ATTN_AVAILABLE
    _layers.FLASH_ATTN_AVAILABLE = enabled
    try:
        yield
    finally:
        _layers.FLASH_ATTN_AVAILABLE = previous


@contextlib.contextmanager
def counted_flash_attention():
    """Count real calls into the flash-attn kernels.

    The dispatch assertion for the flash axis: setting the flag proves nothing
    on its own, because ``SWA3DRoPEAttention`` only reaches the kernel when the
    unpadded branch is taken.
    """
    calls: list[str] = []
    originals = {
        name: getattr(_layers, name)
        for name in ("flash_attn_func", "flash_attn_varlen_func")
    }

    def wrap(name):
        original = originals[name]

        def counted(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        return counted

    for name in originals:
        if originals[name] is not None:
            setattr(_layers, name, wrap(name))
    try:
        yield calls
    finally:
        for name, original in originals.items():
            setattr(_layers, name, original)


@contextlib.contextmanager
def device_independent_noise(seed: int = 0):
    """Draw every sampler noise tensor on the CPU, then move it.

    ``DiffusionStructureHead.sample`` calls ``torch.randn(..., device=device)``
    and ``torch.randn_like``. CUDA and CPU have different generators, so seeding
    both and running gives two different noise realisations and the coordinates
    are then incomparable - which would make a CPU-vs-GPU test meaningless
    rather than merely loose. Routing the draws through one CPU generator makes
    the *noise* device-independent so the remaining difference is arithmetic.
    """
    real_randn, real_randn_like = torch.randn, torch.randn_like
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*size, device=None, dtype=None, **kwargs):
        drawn = real_randn(*size, generator=generator, dtype=torch.float32, **kwargs)
        return drawn.to(device=device, dtype=dtype or torch.float32)

    def randn_like(tensor, **kwargs):
        drawn = real_randn(
            tuple(tensor.shape), generator=generator, dtype=torch.float32
        )
        return drawn.to(device=tensor.device, dtype=tensor.dtype)

    torch.randn, torch.randn_like = randn, randn_like  # ty:ignore[invalid-assignment]
    try:
        yield
    finally:
        torch.randn, torch.randn_like = real_randn, real_randn_like


def build_model(config, device: str = "cpu", model_class=EsmFold2Model):
    """A seeded model on ``device`` with chunking and kernels reset.

    Always built on the CPU and then moved, so every device gets bit-identical
    weights. ``set_kernel_backend`` has to come *after* the move: the fused
    Transition reads ``self.ffn.w12.weight.device`` when it installs its kernel.
    """
    torch.manual_seed(0)
    model = model_class(config).eval().to(device)
    model.set_kernel_backend(None)
    model.set_chunk_size(None)
    return model


def run_forward(model, sequence: str, *, flash: bool, seed: int = 0) -> dict:
    """One seeded forward at the cheap sampler settings."""
    features, lm_hidden_states = esmfold2_inputs(model, sequence)
    device = next(model.parameters()).device
    features = {k: v.to(device) for k, v in features.items()}
    lm_hidden_states = lm_hidden_states.to(device)

    with flash_attention(flash), device_independent_noise(seed), torch.no_grad():
        return model(
            **features,
            lm_hidden_states=lm_hidden_states,
            num_loops=1,
            num_diffusion_samples=1,
            num_sampling_steps=2,
        )


def config_with(config: EsmFold2Config, **overrides) -> EsmFold2Config:
    base = {
        k: v
        for k, v in config.to_dict().items()
        if k != "model_type" and k not in overrides
    }
    return EsmFold2Config(**base, **overrides)


#: Enough MSA encoder to build two blocks, not enough to cost anything.
TINY_MSA_ENCODER = {
    "enabled": True,
    "hidden_size": 32,
    "outer_hidden_size": 8,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "head_width": 8,
}


def model_variant(name: str, config: EsmFold2Config):
    """The four architectures whose fan-out is written by hand, so can be wrong."""
    if name == "release":
        return EsmFold2Model(config)
    if name == "release_msa":
        return EsmFold2Model(config_with(config, msa_encoder=TINY_MSA_ENCODER))
    if name == "experimental":
        return EsmFold2ExperimentalModel(config_with(config, type="experimental"))
    if name == "experimental_msa":
        return EsmFold2ExperimentalModel(
            config_with(config, type="experimental", msa_encoder=TINY_MSA_ENCODER)
        )
    raise AssertionError(name)


def instrument_fanout(model, method_name: str):
    """Wrap ``method_name`` on every module that defines it, recording calls.

    Instrumentation rather than a hardcoded list of module paths: the bug being
    guarded against is a *new* module that the hand-written fan-out never learns
    about, and a list written today cannot see one added tomorrow. The wrapper
    lands in the instance ``__dict__``, so the parent's
    ``self.folding_trunk.set_kernel_backend(...)`` resolves to it and the
    original still runs, keeping the fan-out intact.
    """
    calls: list[tuple[str, object]] = []
    expected = {
        name for name, module in model.named_modules() if hasattr(module, method_name)
    }

    for name, module in model.named_modules():
        if not hasattr(module, method_name):
            continue

        def recorder(value, _name=name, _original=getattr(module, method_name)):
            calls.append((_name, value))
            return _original(value)

        setattr(module, method_name, recorder)

    return calls, expected


# ---------------------------------------------------------------------------
# ``set_kernel_backend`` / ``set_chunk_size`` are total
# ---------------------------------------------------------------------------


# The msa_encoder variants are here because this test found that neither model's
# set_kernel_backend named msa_encoder, so on an MSA-enabled ("Pro") checkpoint
# the msa_encoder.blocks.*.tri_mul_{in,out} modules kept whichever backend they
# were constructed with. MSAEncoder had no set_kernel_backend to call at all.
# Fixed; these cases are what keep it fixed.
KERNEL_BACKEND_VARIANTS = ["release", "release_msa", "experimental", "experimental_msa"]


@pytest.mark.parametrize("variant", KERNEL_BACKEND_VARIANTS)
@pytest.mark.parametrize("backend", [None, "fused", "cuequivariance"])
def test_set_kernel_backend_reaches_every_module(
    tiny_esmfold2_config, monkeypatch, variant, backend
):
    """Every module that *can* take a backend must be *given* one.

    This is an assertion about the module tree, not about outputs, so it needs
    no GPU and no kernels installed - and it is the only kind of test that
    catches the failure it is aimed at. ``EsmFold2Model.set_kernel_backend``
    names its five children by hand; a sixth added later is silently left on
    the reference path, and every output comparison still passes because the
    reference path is correct, merely slow. That is exactly the defect behind
    ``fix(esm): no-op set_kernel_backend on the CP diffusion wrapper``.

    ``CUE_AVAILABLE`` is forced on because ``TriangleMultiplicativeUpdate``
    refuses ``cuequivariance`` outright when the package is missing, which would
    reduce this case to a skip on every CPU runner. Nothing here executes a
    kernel, so the flag only unblocks the traversal.
    """
    monkeypatch.setattr(_layers, "CUE_AVAILABLE", True)
    model = model_variant(variant, tiny_esmfold2_config).eval()
    calls, expected = instrument_fanout(model, "set_kernel_backend")

    model.set_kernel_backend(backend)

    reached = {name for name, _ in calls}
    assert reached == expected, f"never called: {sorted(expected - reached)}"
    # A module reached with the wrong value is as broken as one not reached.
    assert {value for _, value in calls} == {backend}
    # Some backends swap submodules in (``PairUpdateBlock`` rebuilds
    # ``row_drop``); anything new must not itself be configurable, or it just
    # arrived unconfigured.
    after = {n for n, m in model.named_modules() if hasattr(m, "set_kernel_backend")}
    assert after == expected, f"appeared unconfigured: {sorted(after - expected)}"


#: Also found by this test: ``EsmFold2ExperimentalModel.set_chunk_size`` fans
#: out to ``folding_trunk`` and ``confidence_head`` only, so an enabled
#: ``msa_encoder`` keeps its construction-time chunk size. The release model
#: does call ``msa_encoder.set_chunk_size``; the experimental one does not.
CHUNK_SIZE_VARIANTS = ["release", "release_msa", "experimental", "experimental_msa"]


@pytest.mark.parametrize("variant", CHUNK_SIZE_VARIANTS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_set_chunk_size_reaches_every_module(tiny_esmfold2_config, variant, chunk_size):
    """The same totality property for chunking, which has more leaves.

    Chunking is where a missed module is *invisible*: the unchunked path is
    correct, so the only symptom is an OOM on a long input months later.
    """
    model = model_variant(variant, tiny_esmfold2_config).eval()
    calls, expected = instrument_fanout(model, "set_chunk_size")

    model.set_chunk_size(chunk_size)

    reached = {name for name, _ in calls}
    assert reached == expected, f"never called: {sorted(expected - reached)}"
    assert {value for _, value in calls} == {chunk_size}


def test_the_fanout_instrumentation_can_actually_fail(tiny_esmfold2_config):
    """The test above is worthless if the recorder cannot observe a gap.

    Detach one child from the fan-out and the totality assertion must break.
    Without this, a bug in ``instrument_fanout`` would make every case above
    pass vacuously.
    """
    model = EsmFold2Model(tiny_esmfold2_config).eval()
    calls, expected = instrument_fanout(model, "set_chunk_size")

    # Exactly what the CP-wrapper regression looked like: a child that quietly
    # stops forwarding the call.
    def swallow(chunk_size: int | None) -> None:
        return None

    model.confidence_head.set_chunk_size = swallow  # ty:ignore[invalid-assignment]

    model.set_chunk_size(64)

    missed = expected - {name for name, _ in calls}
    assert "confidence_head.folding_trunk" in missed


# ---------------------------------------------------------------------------
# Chunk-size matrix (CPU)
# ---------------------------------------------------------------------------

# Measured on CPU / fp32 at L=197 on the tiny model, worst over chunk sizes
# 16 / 32 / 64 / 128 (``None`` reproduces the reference exactly, 0.0):
#   sample_atom_coords  9.73e-05 A at chunk=64   (coordinates run to |x| ~ 20 A)
#   plddt               0.0       at every chunk size
#   pae                 7.63e-06  at chunk=16
# Chunking only re-associates a sum, so this is fp32 rounding rather than a
# different computation - hence rtol=0 and ~10x headroom, not a round number.
CHUNK_TOLERANCE = {"sample_atom_coords": 1e-3, "plddt": 1e-4, "pae": 1e-4}


def test_every_chunk_size_agrees_on_a_length_that_actually_chunks(tiny_esmfold2):
    """The full ``{None, 16, 32, 64, 128}`` matrix, at a length that splits.

    Both ``Transition`` and ``TriangleMultiplicativeBlock`` return the unchunked
    result whenever ``L <= chunk_size``, so a chunk test on the 12-mer the rest of
    the suite uses would compare a path against itself. 197 residues is longer
    than every chunk size above and prime, so each one splits the pair axis and
    leaves a short remainder chunk.
    """
    sequence = ESMFOLD2_SEQUENCES[MATRIX_LENGTH]

    tiny_esmfold2.set_chunk_size(None)
    reference = run_forward(tiny_esmfold2, sequence, flash=False)

    for chunk_size in CHUNK_SIZES:
        tiny_esmfold2.set_chunk_size(chunk_size)
        candidate = run_forward(tiny_esmfold2, sequence, flash=False)
        for key, tolerance in CHUNK_TOLERANCE.items():
            report(
                f"chunk={chunk_size} {key}",
                max_abs_delta(candidate[key], reference[key]),
            )
            torch.testing.assert_close(
                candidate[key], reference[key], atol=tolerance, rtol=0, msg=key
            )


# ---------------------------------------------------------------------------
# Device / dtype, CPU tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_whole_model_low_precision_cast_is_unsupported(tiny_esmfold2, dtype):
    """Pin the documented constraint instead of pretending the cast works.

    The trunk feeds fp32 features to its weights under
    ``torch.autocast("cuda")``, so casting the *weights* leaves the two sides of
    the first Linear disagreeing. This is asserted, not skipped, because the
    supported-precision story is a contract: if a future change makes
    ``model.bfloat16()`` run, this test fails and somebody has to decide whether
    that is the new documented behaviour.

    The message is ``expected m1 and m2 to have the same dtype, but got: float
    != c10::BFloat16`` on this torch build, from ``F.linear`` in the inputs
    embedder - not the ``mat1 and mat2 must have the same dtype`` wording the
    plan and ``esmfold2_test``'s TODO quote, which came from an addmm on an
    older build. Matched on the stable half of it.
    """
    tiny_esmfold2.to(dtype)
    with pytest.raises(RuntimeError, match="to have the same dtype"):
        run_forward(tiny_esmfold2, ESMFOLD2_SEQUENCES["tiny"], flash=False)


def test_cuda_autocast_is_a_no_op_off_cuda(tiny_esmfold2):
    """Wrapping a CPU forward in CUDA autocast must change nothing at all.

    Bit-identical, not merely close: ``torch.autocast("cuda")`` only rewrites
    CUDA ops, so a CPU run under it executes the same fp32 kernels. This is what
    makes ``test_cpu_fp32``'s claim true, and it is the reason bf16 is
    unreachable on CPU rather than simply untested.
    """
    sequence = ESMFOLD2_SEQUENCES["tiny"]
    plain = run_forward(tiny_esmfold2, sequence, flash=False)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        wrapped = run_forward(tiny_esmfold2, sequence, flash=False)

    for key in ("sample_atom_coords", "plddt", "pae"):
        assert torch.equal(wrapped[key], plain[key]), key


def test_use_amp_is_off_for_cpu_inputs(tiny_esmfold2):
    """``use_amp = ref_pos.device.type == "cuda"``, observed rather than read.

    The single device-conditional switch in ``forward``. Asserted where it takes
    effect - inside the autocast block, at the first module the block wraps -
    so a change to the condition fails here rather than silently halving the
    precision of a CPU run.
    """
    seen: list[bool] = []
    handle = tiny_esmfold2.inputs_embedder.register_forward_pre_hook(
        lambda module, args: seen.append(torch.is_autocast_enabled("cuda"))
    )
    try:
        run_forward(tiny_esmfold2, ESMFOLD2_SEQUENCES["tiny"], flash=False)
    finally:
        handle.remove()

    assert seen, "inputs_embedder never ran; the hook is on the wrong module"
    assert not any(seen)


# ---------------------------------------------------------------------------
# GPU builds against the CPU / fp32 result
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cpu_fp32_result(kernel_esmfold2_config):
    """The one configuration that reproduces anywhere, computed once.

    Every GPU build below is held against *this*, never against another GPU
    build: a build compared to its own output cannot detect a fault the whole
    configuration shares.
    """
    model = build_model(kernel_esmfold2_config, device="cpu")
    result = run_forward(model, ESMFOLD2_SEQUENCES[MATRIX_LENGTH], flash=False)
    # A comparison against a constant tensor passes for every build, so refuse
    # to be that test.
    for key in GPU_TOLERANCE:
        assert result[key].float().std() > 1e-3, f"{key} is degenerate"
    return result


# What a CUDA build can be held to against the CPU / fp32 result, and on which
# output. The gap is *not* mostly a kernel difference: ``use_amp`` is true on
# CUDA, so the whole trunk runs under bf16 autocast there while the CPU runs
# fp32, and that dominates.
#
# Observed at L=197, first the floor (two eager CUDA runs, so only CUDA's own
# reduction-order nondeterminism separates them), then the worst case across all
# seven GPU_BUILDS:
#                       gpu-vs-gpu floor   cpu-vs-gpu, all seven builds
#   sample_atom_coords  1.19e-03 A         9.96e-03 to 1.014e-02 A
#   distogram_logits    2.72e-02           2.44
#   pae                 2.41e-02           2.28 to 2.30
#   plddt               0.0                0.0
# The coordinate spread across builds is under 2%: backend, chunk size and the
# flash flag barely move it, the device does. So only the coordinates are worth
# asserting, and the other three are each excluded for a recorded reason.
#
# ``distogram_logits``: 2.44 is half its own range (|x| <= 5.07), so an untrained
# one-loop trunk in bf16 does not reproduce its fp32 self and a bound wide enough
# to pass would assert nothing.
#
# ``pae``: discontinuous in the coordinates - ``ConfidenceHead`` bins pairwise
# distances and looks the bin up in ``dist_bin_pairwise_embed``, so a shift
# across a bin edge moves a logit by a whole embedding row. It tracks the
# coordinate difference rather than the build.
#
# ``plddt``: ``ConfidenceHead.plddt_weight`` is zero-initialised, so on a
# random-weight model pLDDT is 0.5 everywhere on every device. Exactly 0.0 is a
# vacuous pass.
#
# 3e-2 is 3x the worst coordinate delta and 25x the floor.
GPU_TOLERANCE = {"sample_atom_coords": 3e-2}

# Every build a GPU can reach that the CPU cannot, as (backend, flash, chunk).
# ESMC precision is absent: these configs have no bundled backbone. dtype is
# absent as a *weight* axis because a whole-model cast is unsupported (see
# ``test_a_whole_model_low_precision_cast_is_unsupported``); the bf16 that does
# run comes from autocast and is on in every row.
GPU_BUILDS = [
    (None, False, None),
    (None, False, 64),
    (None, True, None),
    ("fused", False, None),
    ("fused", True, 64),
    ("cuequivariance", False, None),
    ("cuequivariance", True, 64),
]


def assert_backend_dispatched(model, backend):
    """The module tree really took the backend, before any output is compared.

    Without this a missing package turns the whole matrix into a comparison of
    the reference path against itself.
    """
    transitions = [m for m in model.modules() if isinstance(m, _layers.Transition)]
    pair_blocks = [m for m in model.modules() if isinstance(m, _layers.PairUpdateBlock)]
    attentions = [
        m for m in model.modules() if isinstance(m, _layers.AttentionPairBias)
    ]
    tri_muls = [
        m
        for m in model.modules()
        if isinstance(m, _layers.TriangleMultiplicativeUpdate)
    ]
    assert transitions and pair_blocks and attentions and tri_muls

    for module in transitions + pair_blocks + attentions:
        assert module._kernel_backend == backend, type(module).__name__

    if backend == "fused":
        assert _layers.TRITON_KERNELS_AVAILABLE
        for module in transitions:
            assert module._fused_swiglu is not None
        for block in pair_blocks:
            assert block.row_drop._use_fused_kernels
    else:
        for module in transitions:
            assert module._fused_swiglu is None

    for tri_mul in tri_muls:
        assert tri_mul._engine._use_kernels == (backend == "cuequivariance")


def assert_chunking_dispatched(model, chunk_size):
    engines = [
        m for m in model.modules() if isinstance(m, _layers.TriangleMultiplicativeBlock)
    ]
    transitions = [m for m in model.modules() if isinstance(m, _layers.Transition)]
    assert engines and transitions
    for module in engines + transitions:
        assert module._chunk_size == chunk_size, type(module).__name__


def assert_close_to_reference(result, reference, tolerance: dict, label: str):
    """Report every delta, then hold each one to its own measured bound."""
    for key, bound in tolerance.items():
        delta = report(f"{label} {key}", max_abs_delta(result[key], reference[key]))
        assert delta < bound, f"{label}: {key} moved {delta:.3e} (bound {bound})"


def skip_if_backend_missing(backend):
    if backend == "fused" and not _layers.TRITON_KERNELS_AVAILABLE:
        pytest.skip("triton kernels unavailable")  # ty:ignore[too-many-positional-arguments]
    if backend == "cuequivariance" and not _layers.CUE_AVAILABLE:
        pytest.skip("cuequivariance unavailable")  # ty:ignore[too-many-positional-arguments]


@pytest.mark.gpu
def test_cpu_and_gpu_agree(kernel_esmfold2_config, cpu_fp32_result):
    """E1: same weights, same seeded noise, both devices.

    ESMFold2 has far more device-conditional code than ESMC - ``use_amp``,
    ``_fused_active``'s ``tensor.is_cuda``, the cuSOLVER SVD path in
    ``_weighted_rigid_align`` - and none of it had a test. The sampler's noise
    is drawn through one CPU generator on both sides (see
    ``device_independent_noise``); without that the two runs get different
    noise realisations and the coordinates are not comparable at all.
    """
    model = build_model(kernel_esmfold2_config, device="cuda")
    assert next(model.parameters()).is_cuda
    result = run_forward(model, ESMFOLD2_SEQUENCES[MATRIX_LENGTH], flash=False)

    assert_close_to_reference(result, cpu_fp32_result, GPU_TOLERANCE, "cpu-vs-gpu")


@pytest.mark.gpu
def test_use_amp_is_on_for_cuda_inputs(tiny_esmfold2_config):
    """The other half of ``use_amp = ref_pos.device.type == "cuda"``.

    Pinned as an expectation because it is *why* the CPU/GPU tolerances below
    are bf16-sized rather than fp32-sized: there is no supported way to run the
    CUDA trunk in full fp32.
    """
    model = build_model(tiny_esmfold2_config, device="cuda")
    seen: list[bool] = []
    handle = model.inputs_embedder.register_forward_pre_hook(
        lambda module, args: seen.append(torch.is_autocast_enabled("cuda"))
    )
    try:
        run_forward(model, ESMFOLD2_SEQUENCES["tiny"], flash=False)
    finally:
        handle.remove()

    assert seen and all(seen)


@pytest.mark.gpu
@pytest.mark.parametrize("backend,flash,chunk_size", GPU_BUILDS)
def test_every_gpu_build_reaches_the_cpu_reference(
    kernel_esmfold2_config, cpu_fp32_result, backend, flash, chunk_size
):
    """E2: the build matrix, each case held against the CPU / fp32 result.

    Dispatch first, comparison second. A backend that silently fell back, a
    chunk size that never reached a leaf, or a flash flag that never routed a
    call would all produce a passing comparison otherwise - the reference path
    is correct, so falling back to it is invisible in the numbers.
    """
    skip_if_backend_missing(backend)
    try:
        model = build_model(kernel_esmfold2_config, device="cuda")
        model.set_kernel_backend(backend)
        model.set_chunk_size(chunk_size)
    # Only a genuinely absent package is a reason to skip. A bare ``except``
    # here would turn a real regression - a shape assertion inside a kernel, a
    # renamed attribute - into a green skip across the whole matrix.
    except ImportError as exc:
        pytest.skip(f"build {backend}/{flash}/{chunk_size} unavailable: {exc}")  # ty:ignore[too-many-positional-arguments]

    assert_backend_dispatched(model, backend)
    assert_chunking_dispatched(model, chunk_size)

    with counted_flash_attention() as flash_calls:
        result = run_forward(model, ESMFOLD2_SEQUENCES[MATRIX_LENGTH], flash=flash)
    if flash:
        assert _layers.flash_attn_func is not None, "flash-attn is not installed"
        assert flash_calls, "flash=True but no flash-attn kernel ran"
    else:
        assert not flash_calls

    label = f"{backend}/flash={flash}/chunk={chunk_size}"
    assert_close_to_reference(result, cpu_fp32_result, GPU_TOLERANCE, label)


# ---------------------------------------------------------------------------
# Torch.compile
# ---------------------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.parametrize("mode", ["fixed_seqlen", "dynamic_seqlen"])
def test_torch_compile_matches_eager(kernel_esmfold2_config, mode):
    """Both compile modes must reproduce the eager result on the same device.

    Compared against an eager *CUDA* run, not the CPU reference: the question
    ``apply_torch_compile`` raises is whether Inductor changes the answer, and
    mixing in the device difference would bury it under a bf16-sized tolerance.
    ``apply_torch_compile`` rebinds ``module.forward`` in place and cannot be
    undone, so the compiled model is a second, freshly-built copy.

    The tolerances are deliberately absent rather than guessed: nothing
    compiles, so there is nothing to measure. ``raises=`` pins the *reason* -
    any other exception is reported as a real failure, not an expected one.
    """
    eager = build_model(kernel_esmfold2_config, device="cuda")
    expected = run_forward(eager, ESMFOLD2_SEQUENCES[MATRIX_LENGTH], flash=False)

    # Dynamo caches on the *code object*, which every test in this file shares.
    # Without a reset a compiled entry from an earlier test satisfies this one's
    # guards and no compilation happens at all - which silently turned this into
    # a pass once already.
    torch._dynamo.reset()
    compiled = build_model(kernel_esmfold2_config, device="cuda")
    compiled.apply_torch_compile(mode=mode)
    actual = run_forward(compiled, ESMFOLD2_SEQUENCES[MATRIX_LENGTH], flash=False)

    assert_close_to_reference(actual, expected, GPU_TOLERANCE, f"compile[{mode}]")


@pytest.mark.nightly
def test_torch_compile_does_not_stack_with_the_fused_backend(kernel_esmfold2_config):
    """``apply_torch_compile``'s "does not stack with Triton" claim is unenforced.

    "Does NOT stack with our Triton kernels - call ``set_kernel_backend(None)``
    before compiling" is a comment and nothing else: the method inspects nothing,
    raises nothing and warns nothing at call time. That is the assertion here,
    and a future guard would have to change it.

    What the stacked forward then *does* is platform-dependent (see the module
    docstring), so this only requires that it produces a usable result on
    whichever platform reached it. The numeric question - does compiling change
    the answer - is ``test_torch_compile_matches_eager``'s job, on the
    unstacked build where a clean eager reference exists.
    """
    skip_if_backend_missing("fused")

    # The fused build itself has to work, or the rest of this proves nothing.
    eager = build_model(kernel_esmfold2_config, device="cuda")
    eager.set_kernel_backend("fused")
    expected = run_forward(eager, ESMFOLD2_SEQUENCES[MATRIX_LENGTH], flash=False)

    torch._dynamo.reset()
    stacked = build_model(kernel_esmfold2_config, device="cuda")
    stacked.set_kernel_backend("fused")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stacked.apply_torch_compile(mode="fixed_seqlen")
    about_kernels = [
        str(w.message)
        for w in caught
        if "kernel" in str(w.message).lower() or "fused" in str(w.message).lower()
    ]
    assert not about_kernels, about_kernels

    actual = run_forward(stacked, ESMFOLD2_SEQUENCES[MATRIX_LENGTH], flash=False)
    for key in GPU_TOLERANCE:
        assert actual[key].shape == expected[key].shape
        assert torch.isfinite(actual[key]).all(), f"{key} is not finite"
