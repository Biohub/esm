from unittest.mock import MagicMock

import pytest
import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProteinTensor, LogitsConfig, LogitsOutput


def _model():
    tokenizers = MagicMock()
    tokenizers.sequence.mask_token_id = 32
    return ESM3(
        d_model=64,
        n_heads=4,
        v_heads=4,
        n_layers=1,
        structure_encoder_fn=MagicMock(),
        structure_decoder_fn=MagicMock(),
        function_decoder_fn=MagicMock(),
        tokenizers=tokenizers,
    ).eval()


def _outputs(result: LogitsOutput):
    assert result.logits is not None
    return {
        "sequence": result.logits.sequence,
        "structure": result.logits.structure,
        "secondary_structure": result.logits.secondary_structure,
        "sasa": result.logits.sasa,
        "function": result.logits.function,
        "residue_annotations": result.residue_annotation_logits,
        "embeddings": result.embeddings,
    }


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_logits_preserves_confidence_scalar(default_dtype):
    model = _model()
    protein = ESMProteinTensor(sequence=torch.tensor([0, 4, 5, 6, 2]))
    previous_dtype = torch.get_default_dtype()
    observed = []

    def record_confidence(module, args, kwargs):
        observed.append(kwargs["average_plddt"])

    handle = model.register_forward_pre_hook(record_confidence, with_kwargs=True)
    try:
        torch.set_default_dtype(default_dtype)
        expected = torch.tensor(1.0)
        model.logits(protein)
    finally:
        handle.remove()
        torch.set_default_dtype(previous_dtype)

    assert len(observed) == 1
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_coordinates", [False, True])
def test_logits_cuda_graph_replays_changed_inputs(dtype, with_coordinates):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("ESM3 logits uses CUDA BF16 autocast")

    torch.manual_seed(0)
    model = _model().to(device="cuda", dtype=dtype)
    config = LogitsConfig(
        sequence=True,
        structure=True,
        secondary_structure=True,
        sasa=True,
        function=True,
        residue_annotations=True,
        return_embeddings=True,
    )
    proteins = [
        ESMProteinTensor(
            sequence=torch.randint(4, 24, (8,), device="cuda"),
            coordinates=torch.randn(8, 3, 3, device="cuda")
            if with_coordinates
            else None,
        )
        for _ in range(3)
    ]
    static_input = proteins[0].copy()
    references = [_outputs(model.logits(protein, config)) for protein in proteins]
    for reference in references:
        for value in reference.values():
            assert value is not None
            assert torch.isfinite(value).all()
    assert not torch.equal(references[0]["sequence"], references[1]["sequence"])

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            model.logits(static_input, config)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model.logits(static_input, config)

    for protein, reference in zip(proteins, references):
        static_input.sequence.copy_(protein.sequence)
        if with_coordinates:
            static_input.coordinates.copy_(protein.coordinates)
        graph.replay()
        torch.testing.assert_close(_outputs(output), reference)
