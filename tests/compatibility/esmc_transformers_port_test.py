"""``transformers``' own ESMC port loads the published checkpoints next to ``esm``.

The model cards say ``pip install esm`` and then load ``biohub/ESMC-*`` with
``AutoTokenizer`` / ``AutoModelForMaskedLM``; these check that path agrees with
this package. Skipped on ``transformers`` < 5.16, which has no ESMC port.
"""

import pytest
import torch

pytest.importorskip("transformers.models.esmc")

from transformers import AutoModelForMaskedLM, AutoTokenizer

from tests.conftest import MEDIUM_SEQUENCE


def test_auto_tokenizer_matches_ours(esmc_300m_dir, esmc_tokenizer):
    theirs = AutoTokenizer.from_pretrained(esmc_300m_dir)
    assert type(theirs).__module__.startswith("transformers.models.esmc")
    ours = esmc_tokenizer(MEDIUM_SEQUENCE)["input_ids"]
    assert theirs(MEDIUM_SEQUENCE)["input_ids"] == ours


def test_auto_model_matches_ours(esmc_300m_dir, esmc_300m, esmc_tokenizer):
    theirs = AutoModelForMaskedLM.from_pretrained(esmc_300m_dir).eval()
    assert type(theirs).__module__.startswith("transformers.models.esmc")
    input_ids = esmc_tokenizer(MEDIUM_SEQUENCE, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        # Both are fp32 on CPU, but the attention kernels differ: up to 8.4e-5 apart.
        torch.testing.assert_close(
            theirs(input_ids=input_ids).logits,
            esmc_300m.eval()(input_ids=input_ids).logits,
            atol=1e-4,
            rtol=1e-4,
        )
