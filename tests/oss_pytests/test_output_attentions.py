import torch

from esm.models.esm3 import ESM3, ESMOutput
from esm.models.esmc import ESMC, ESMCOutput
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.constants import esm3 as C

B, L = 2, 10
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2


def make_esmc() -> ESMC:
    tokenizer = EsmSequenceTokenizer()
    return ESMC(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        tokenizer=tokenizer,
        use_flash_attn=False,
    )


def make_esm3() -> ESM3:
    from unittest.mock import MagicMock

    tokenizers = MagicMock()
    tokenizers.sequence.mask_token_id = 32

    return ESM3(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        v_heads=4,
        n_layers=N_LAYERS,
        structure_encoder_fn=MagicMock(),
        structure_decoder_fn=MagicMock(),
        function_decoder_fn=MagicMock(),
        tokenizers=tokenizers,
    )


def test_esmc_output_attentions_shape():
    model = make_esmc()
    tokens = torch.randint(
        C.SEQUENCE_STANDARD_AA_MIN_TOKEN, C.SEQUENCE_STANDARD_AA_MAX_TOKEN, (B, L)
    )

    with torch.no_grad():
        out: ESMCOutput = model(sequence_tokens=tokens, output_attentions=True)

    assert out.attentions is not None
    assert len(out.attentions) == N_LAYERS
    for layer_attn in out.attentions:
        assert layer_attn.shape == (B, N_HEADS, L, L)


def test_esmc_output_attentions_none_by_default():
    model = make_esmc()
    tokens = torch.randint(
        C.SEQUENCE_STANDARD_AA_MIN_TOKEN, C.SEQUENCE_STANDARD_AA_MAX_TOKEN, (B, L)
    )

    with torch.no_grad():
        out: ESMCOutput = model(sequence_tokens=tokens)

    assert out.attentions is None


def test_esmc_output_attentions_numerically_consistent():
    """Logits from manual attention path should match the sdpa path."""
    model = make_esmc()
    tokens = torch.randint(
        C.SEQUENCE_STANDARD_AA_MIN_TOKEN, C.SEQUENCE_STANDARD_AA_MAX_TOKEN, (B, L)
    )

    with torch.no_grad():
        out_attn: ESMCOutput = model(sequence_tokens=tokens, output_attentions=True)
        out_sdpa: ESMCOutput = model(sequence_tokens=tokens)

    assert torch.allclose(out_attn.sequence_logits, out_sdpa.sequence_logits, atol=1e-4)


def test_esmc_attention_weights_sum_to_one():
    model = make_esmc()
    tokens = torch.randint(
        C.SEQUENCE_STANDARD_AA_MIN_TOKEN, C.SEQUENCE_STANDARD_AA_MAX_TOKEN, (B, L)
    )

    with torch.no_grad():
        out: ESMCOutput = model(sequence_tokens=tokens, output_attentions=True)

    assert out.attentions is not None
    for layer_attn in out.attentions:
        # Each row of the attention matrix should sum to 1.
        row_sums = layer_attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_esm3_output_attentions_shape():
    model = make_esm3()
    sequence_tokens = torch.full((B, L), C.SEQUENCE_MASK_TOKEN, dtype=torch.long)
    sequence_tokens[:, 0] = C.SEQUENCE_BOS_TOKEN
    sequence_tokens[:, -1] = C.SEQUENCE_EOS_TOKEN

    with torch.no_grad():
        out: ESMOutput = model(sequence_tokens=sequence_tokens, output_attentions=True)

    assert out.attentions is not None
    assert len(out.attentions) == N_LAYERS
    for layer_attn in out.attentions:
        assert layer_attn.shape == (B, N_HEADS, L, L)


def test_esm3_output_attentions_none_by_default():
    model = make_esm3()
    sequence_tokens = torch.full((B, L), C.SEQUENCE_MASK_TOKEN, dtype=torch.long)

    with torch.no_grad():
        out: ESMOutput = model(sequence_tokens=sequence_tokens)

    assert out.attentions is None


def test_esm3_output_attentions_numerically_consistent():
    """Logits from manual attention path should match the sdpa path."""
    model = make_esm3()
    sequence_tokens = torch.full((B, L), C.SEQUENCE_MASK_TOKEN, dtype=torch.long)

    with torch.no_grad():
        out_attn: ESMOutput = model(
            sequence_tokens=sequence_tokens, output_attentions=True
        )
        out_sdpa: ESMOutput = model(sequence_tokens=sequence_tokens)

    assert torch.allclose(out_attn.sequence_logits, out_sdpa.sequence_logits, atol=1e-4)
