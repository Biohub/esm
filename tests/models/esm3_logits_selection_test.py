from unittest.mock import MagicMock

import pytest
import torch

from esm.models.esm3 import ESM3, ESMOutput
from esm.sdk.api import ESMProteinTensor, LogitsConfig

_FIELDS = {
    "sequence": "sequence_logits",
    "structure": "structure_logits",
    "secondary_structure": "secondary_structure_logits",
    "sasa": "sasa_logits",
    "function": "function_logits",
    "residue_annotations": "residue_logits",
    "return_embeddings": "embeddings",
}


@pytest.fixture(scope="module")
def model():
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


@pytest.mark.parametrize(
    "requested", [(), tuple(_FIELDS), *((key,) for key in _FIELDS)]
)
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_logits_only_converts_requested_outputs(
    model, monkeypatch, requested, output_dtype
):
    protein = ESMProteinTensor(sequence=torch.tensor([0, 4, 5, 6, 2]))
    config = LogitsConfig(**dict.fromkeys(requested, True))
    raw_outputs = {}
    converted = []
    head_outputs = []
    tensor_to = torch.Tensor.to

    def record_heads(module, args, output):
        head_outputs.append(output)

    def record_forward(module, args, output):
        # Exercise the low-precision outputs produced by CUDA autocast on CPU too.
        output = ESMOutput(
            attentions=output.attentions,
            **{
                field: tensor_to(getattr(output, field), dtype=output_dtype)
                for field in _FIELDS.values()
            },
        )
        raw_outputs.update(
            {key: getattr(output, field) for key, field in _FIELDS.items()}
        )
        return output

    def record_conversion(tensor, *args, **kwargs):
        converted.extend(key for key, value in raw_outputs.items() if tensor is value)
        return tensor_to(tensor, *args, **kwargs)

    head_hook = model.output_heads.register_forward_hook(record_heads)
    forward_hook = model.register_forward_hook(record_forward)
    monkeypatch.setattr(torch.Tensor, "to", record_conversion)
    try:
        result = model.logits(protein, config)
    finally:
        head_hook.remove()
        forward_hook.remove()

    # The public forward and output-head hooks still receive every raw output.
    assert len(head_outputs) == 1
    assert isinstance(head_outputs[0], ESMOutput)
    assert all(
        isinstance(getattr(head_outputs[0], field), torch.Tensor)
        for field in _FIELDS.values()
    )
    assert set(converted) == set(requested)
    assert result.logits is not None
    selected = {
        "sequence": result.logits.sequence,
        "structure": result.logits.structure,
        "secondary_structure": result.logits.secondary_structure,
        "sasa": result.logits.sasa,
        "function": result.logits.function,
        "residue_annotations": result.residue_annotation_logits,
        "return_embeddings": result.embeddings,
    }
    for key, value in selected.items():
        if key in requested:
            expected = tensor_to(
                raw_outputs[key], device=protein.device, dtype=torch.float32
            )
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        else:
            assert value is None
