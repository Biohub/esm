"""Parity tests for opt-in ESMFold2 low-memory inference paths.

The tests use tiny randomly initialized CPU models and never download weights.
They cover the topology and sample-ordering contracts that are easy to break
while reducing inference memory; the stored execution reference separately pins
the default monomer forward bit-for-bit.
"""

from __future__ import annotations

import pytest
import torch

from esm.models.esmc import EsmcModel
from esm.models.esmfold2 import layers as _layers
from esm.models.esmfold2.protein_utils import prepare_protein_features


@pytest.fixture(autouse=True)
def _force_reference_attention(monkeypatch):
    monkeypatch.setattr(_layers, "FLASH_ATTN_AVAILABLE", False)


def _topology_features(chains: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Protein-only forward features with chain identity set explicitly."""
    features = prepare_protein_features("".join(chains))
    length = sum(map(len, chains))
    asym = torch.empty(1, length, dtype=torch.long)
    residue = torch.empty(1, length, dtype=torch.long)
    entity = torch.empty(1, length, dtype=torch.long)
    sym = torch.empty(1, length, dtype=torch.long)
    entity_for_sequence: dict[str, int] = {}
    copies_per_entity: dict[int, int] = {}
    offset = 0
    for chain_id, sequence in enumerate(chains):
        stop = offset + len(sequence)
        entity_id = entity_for_sequence.setdefault(sequence, len(entity_for_sequence))
        sym_id = copies_per_entity.get(entity_id, 0)
        copies_per_entity[entity_id] = sym_id + 1
        asym[:, offset:stop] = chain_id
        residue[:, offset:stop] = torch.arange(len(sequence))
        entity[:, offset:stop] = entity_id
        sym[:, offset:stop] = sym_id
        if offset:
            features["token_bonds"][:, offset - 1, offset] = 0
            features["token_bonds"][:, offset, offset - 1] = 0
        offset = stop
    features.update(asym_id=asym, residue_index=residue, entity_id=entity, sym_id=sym)
    return features


def _synthetic_lm_states(model, length: int, batch_size: int = 1) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(
        batch_size, length, model.config.lm_num_layers + 1, model.config.lm_d_model
    )


def _repeat_batch(
    features: dict[str, torch.Tensor], batch_size: int
) -> dict[str, torch.Tensor]:
    return {
        name: value.expand(batch_size, *value.shape[1:]).clone()
        for name, value in features.items()
    }


def _assert_forward_outputs_close(
    baseline: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> None:
    assert baseline.keys() == candidate.keys()
    for name in baseline:
        if name in {
            "sample_atom_coords",
            "distogram_logits",
            "atom_pad_mask",
            "residue_index",
            "entity_id",
        }:
            torch.testing.assert_close(baseline[name], candidate[name], atol=0, rtol=0)
        else:
            # Changing the confidence-head batch shape can select a different
            # reduction order; observed CPU drift is below 4e-7 absolute.
            torch.testing.assert_close(
                baseline[name], candidate[name], atol=1e-6, rtol=1e-6
            )


@pytest.mark.parametrize(
    "chains",
    [(("ACDEFG",)), (("ACDEFG", "HIKLM")), (("ACDEFG", "ACDEFG"))],
    ids=["monomer", "heteromer", "homomer"],
)
def test_confidence_sample_chunking_preserves_all_outputs(tiny_esmfold2, chains):
    features = _topology_features(chains)
    lm_states = _synthetic_lm_states(tiny_esmfold2, sum(map(len, chains)))
    settings = dict(num_loops=1, num_sampling_steps=2, num_diffusion_samples=3)

    torch.manual_seed(19)
    baseline = tiny_esmfold2(**features, lm_hidden_states=lm_states.clone(), **settings)
    torch.manual_seed(19)
    chunked = tiny_esmfold2(
        **features,
        lm_hidden_states=lm_states.clone(),
        confidence_sample_chunk_size=1,
        **settings,
    )

    _assert_forward_outputs_close(baseline, chunked)


def test_confidence_chunking_preserves_batch_sample_order_with_remainder(tiny_esmfold2):
    features = _repeat_batch(_topology_features(("ACDEFG",)), batch_size=2)
    lm_states = _synthetic_lm_states(tiny_esmfold2, 6, batch_size=2)
    settings = dict(num_loops=1, num_sampling_steps=2, num_diffusion_samples=3)

    torch.manual_seed(31)
    baseline = tiny_esmfold2(**features, lm_hidden_states=lm_states.clone(), **settings)
    torch.manual_seed(31)
    chunked = tiny_esmfold2(
        **features,
        lm_hidden_states=lm_states.clone(),
        confidence_sample_chunk_size=2,
        **settings,
    )

    _assert_forward_outputs_close(baseline, chunked)


def test_pair_encodings_are_recomputed_at_their_use_sites(tiny_esmfold2):
    features = _topology_features(("ACDEFG",))
    lm_states = _synthetic_lm_states(tiny_esmfold2, 6)
    calls = {"relative": 0, "bonds": 0}
    handles = [
        tiny_esmfold2.rel_pos.register_forward_hook(
            lambda *_: calls.__setitem__("relative", calls["relative"] + 1)
        ),
        tiny_esmfold2.token_bonds.register_forward_hook(
            lambda *_: calls.__setitem__("bonds", calls["bonds"] + 1)
        ),
    ]
    try:
        torch.manual_seed(3)
        tiny_esmfold2(
            **features,
            lm_hidden_states=lm_states,
            num_loops=1,
            num_sampling_steps=2,
            num_diffusion_samples=1,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"relative": 2, "bonds": 2}


def test_recomputed_pair_encodings_match_cached_upstream_output(
    tiny_esmfold2, monkeypatch
):
    features = _topology_features(("ACDEFG",))
    lm_states = _synthetic_lm_states(tiny_esmfold2, 6)
    settings = dict(num_loops=1, num_sampling_steps=2, num_diffusion_samples=1)

    torch.manual_seed(43)
    recomputed = tiny_esmfold2(
        **features, lm_hidden_states=lm_states.clone(), **settings
    )

    real_compute_pair_encodings = tiny_esmfold2._compute_pair_encodings
    cached_relative = None
    cached_bonds = None

    def use_upstream_cached_pair_encodings(*, token_bonds=None, **kwargs):
        nonlocal cached_relative, cached_bonds
        if cached_relative is None:
            cached_relative, cached_bonds = real_compute_pair_encodings(
                token_bonds=features["token_bonds"], **kwargs
            )
        return cached_relative, cached_bonds if token_bonds is not None else None

    monkeypatch.setattr(
        tiny_esmfold2, "_compute_pair_encodings", use_upstream_cached_pair_encodings
    )
    torch.manual_seed(43)
    upstream_semantics = tiny_esmfold2(
        **features, lm_hidden_states=lm_states.clone(), **settings
    )

    _assert_forward_outputs_close(upstream_semantics, recomputed)


def test_distogram_is_materialized_after_diffusion_and_confidence(
    tiny_esmfold2, monkeypatch
):
    features = _topology_features(("ACDEFG",))
    lm_states = _synthetic_lm_states(tiny_esmfold2, 6)
    order: list[str] = []
    real_sample = tiny_esmfold2.structure_head.sample

    def record_sample(**kwargs):
        order.append("diffusion")
        return real_sample(**kwargs)

    monkeypatch.setattr(tiny_esmfold2.structure_head, "sample", record_sample)
    handles = [
        tiny_esmfold2.confidence_head.register_forward_hook(
            lambda *_: order.append("confidence")
        ),
        tiny_esmfold2.distogram_head.register_forward_hook(
            lambda *_: order.append("distogram")
        ),
    ]
    try:
        torch.manual_seed(5)
        tiny_esmfold2(
            **features,
            lm_hidden_states=lm_states,
            num_loops=1,
            num_sampling_steps=2,
            num_diffusion_samples=1,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert order[-3:] == ["diffusion", "confidence", "distogram"]


@pytest.mark.parametrize(
    "chains",
    [("ACDEFG",), ("ACDEFG", "HIKLM"), ("ACDEFG", "ACDEFG")],
    ids=["monomer", "heteromer", "homomer"],
)
def test_esmc_offload_option_preserves_output_and_requests_cpu(
    tiny_esmfold2, tiny_esmc_config, monkeypatch, chains
):
    features = _topology_features(chains)
    torch.manual_seed(0)
    tiny_esmfold2.esmc = EsmcModel(tiny_esmc_config).eval()
    requested_devices: list[torch.device] = []
    real_move = tiny_esmfold2._move_esmc_to

    def record_move(device):
        requested_devices.append(torch.device(device))
        real_move(torch.device(device))

    monkeypatch.setattr(tiny_esmfold2, "_move_esmc_to", record_move)

    torch.manual_seed(23)
    baseline = tiny_esmfold2(
        **features, num_loops=1, num_sampling_steps=2, num_diffusion_samples=1
    )
    torch.manual_seed(23)
    offloaded = tiny_esmfold2(
        **features,
        offload_esmc_after_lm=True,
        num_loops=1,
        num_sampling_steps=2,
        num_diffusion_samples=1,
    )

    assert torch.device("cpu") in requested_devices
    for name in baseline:
        torch.testing.assert_close(baseline[name], offloaded[name], atol=0, rtol=0)


def test_esmc_offload_rejects_fp8(tiny_esmfold2):
    features = _topology_features(("ACDEFG",))
    lm_states = _synthetic_lm_states(tiny_esmfold2, 6)
    tiny_esmfold2._esmc_fp8 = True

    with pytest.raises(ValueError, match="not supported.*FP8"):
        tiny_esmfold2(
            **features,
            lm_hidden_states=lm_states,
            offload_esmc_after_lm=True,
            num_loops=1,
            num_sampling_steps=2,
            num_diffusion_samples=1,
        )


def test_explicit_offload_false_overrides_low_memory_preset(tiny_esmfold2, monkeypatch):
    features = _topology_features(("ACDEFG",))
    lm_states = _synthetic_lm_states(tiny_esmfold2, 6)
    tiny_esmfold2.esmc = torch.nn.Linear(1, 1, bias=False).eval()
    requested_devices: list[torch.device] = []
    monkeypatch.setattr(
        tiny_esmfold2,
        "_move_esmc_to",
        lambda device: requested_devices.append(torch.device(device)),
    )

    torch.manual_seed(37)
    tiny_esmfold2(
        **features,
        lm_hidden_states=lm_states,
        low_memory_mode=True,
        offload_esmc_after_lm=False,
        num_loops=1,
        num_sampling_steps=2,
        num_diffusion_samples=1,
    )

    assert requested_devices == []


def test_low_memory_mode_matches_explicit_combined_options(tiny_esmfold2, monkeypatch):
    features = _topology_features(("ACDEFG", "ACDEFG"))
    lm_states = _synthetic_lm_states(tiny_esmfold2, 12)
    tiny_esmfold2.esmc = torch.nn.Linear(1, 1, bias=False).eval()
    requested_devices: list[torch.device] = []
    confidence_chunks: list[int | None] = []
    real_move = tiny_esmfold2._move_esmc_to
    real_confidence = tiny_esmfold2._run_confidence_head

    def record_move(device):
        requested_devices.append(torch.device(device))
        real_move(torch.device(device))

    def record_confidence(*, sample_chunk_size, **kwargs):
        confidence_chunks.append(sample_chunk_size)
        return real_confidence(sample_chunk_size=sample_chunk_size, **kwargs)

    monkeypatch.setattr(tiny_esmfold2, "_move_esmc_to", record_move)
    monkeypatch.setattr(tiny_esmfold2, "_run_confidence_head", record_confidence)
    settings = dict(num_loops=1, num_sampling_steps=2, num_diffusion_samples=3)

    torch.manual_seed(29)
    explicit = tiny_esmfold2(
        **features,
        lm_hidden_states=lm_states.clone(),
        offload_esmc_after_lm=True,
        confidence_sample_chunk_size=1,
        **settings,
    )
    torch.manual_seed(29)
    umbrella = tiny_esmfold2(
        **features, lm_hidden_states=lm_states.clone(), low_memory_mode=True, **settings
    )

    assert requested_devices == [torch.device("cpu"), torch.device("cpu")]
    assert confidence_chunks == [1, 1]
    for name in explicit:
        torch.testing.assert_close(explicit[name], umbrella[name], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_recomputed_pair_encodings_preserve_autocast_and_default_output(
    tiny_esmfold2,
):
    model = tiny_esmfold2.cuda().eval()
    features = {
        name: value.cuda() for name, value in _topology_features(("ACDEFG",)).items()
    }
    lm_states = _synthetic_lm_states(model, 6).cuda()
    settings = dict(num_loops=1, num_sampling_steps=2, num_diffusion_samples=1)
    observations: list[tuple[str, torch.Tensor, bool]] = []

    def record_pair_output(name):
        def hook(_module, _args, output):
            observations.append(
                (name, output.detach().clone(), torch.is_autocast_enabled("cuda"))
            )

        return hook

    handles = [
        model.rel_pos.register_forward_hook(record_pair_output("relative")),
        model.token_bonds.register_forward_hook(record_pair_output("bonds")),
    ]
    torch.manual_seed(41)
    model(**features, lm_hidden_states=lm_states, **settings)
    for handle in handles:
        handle.remove()

    assert [name for name, _, _ in observations] == [
        "relative",
        "bonds",
        "relative",
        "bonds",
    ]
    assert all(value.dtype == torch.bfloat16 for _, value, _ in observations)
    assert all(autocast_enabled for _, _, autocast_enabled in observations)
    relative_outputs = [value for name, value, _ in observations if name == "relative"]
    bond_outputs = [value for name, value, _ in observations if name == "bonds"]
    for recomputed in relative_outputs[1:]:
        torch.testing.assert_close(relative_outputs[0], recomputed, atol=0, rtol=0)
    torch.testing.assert_close(bond_outputs[0], bond_outputs[1], atol=0, rtol=0)
