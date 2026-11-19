"""``set_memory_mode("big")`` must not change what the model computes."""

import pytest
import torch

from tests.conftest import esmfold2_inputs

# Two token bands of 256, the last one short.
SEQUENCE = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG" * 4
)[:300]
FOLD = dict(num_loops=1, num_sampling_steps=2, num_diffusion_samples=2)


@pytest.mark.parametrize(
    "device,backend",
    [
        ("cpu", None),
        pytest.param("cuda", "fused", marks=pytest.mark.gpu),
        pytest.param("cuda", "fast", marks=pytest.mark.gpu),
    ],
)
def test_big_folds_like_the_default(request, deterministic, device, backend):
    model = request.getfixturevalue(
        "tiny_esmfold2" if device == "cpu" else "wide_esmfold2"
    )
    model.set_kernel_backend(backend)
    features, lm_hidden_states = esmfold2_inputs(model, SEQUENCE)
    features = {k: v.to(device) for k, v in features.items()}
    outputs = {}
    for mode in (None, "big"):
        model.set_memory_mode(mode)
        torch.manual_seed(0)
        out = model(**features, lm_hidden_states=lm_hidden_states.to(device), **FOLD)
        outputs[mode] = {
            k: v.cpu() for k, v in out.items() if isinstance(v, torch.Tensor)
        }

    ref, got = outputs[None], outputs["big"]
    assert got.keys() == ref.keys()
    # Bands change the GEMMs' M, so the last bits move with the BLAS kernel
    # choice. Measured worst: 1.2e-5 of the coordinate range, on L40.
    for key in ref:
        scale = ref[key].double().abs().max().clamp_min(1e-6).item()
        torch.testing.assert_close(
            got[key], ref[key], atol=1e-4 * scale, rtol=0, msg=key
        )
