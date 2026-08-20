import random

import torch
import torch.nn.functional as F

from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_sm_open_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization import get_esm3_model_tokenizers
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer as EsmFunctionTokenizer,
)
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation


@torch.no_grad()
def inverse_folding_example():
    tokenizer = EsmSequenceTokenizer()
    encoder = ESM3_structure_encoder_v0("cuda")
    model = ESM3_sm_open_v0("cuda")

    chain = ProteinChain.from_rcsb("1utn", "A")
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()
    coords = coords.cuda()
    plddt = plddt.cuda()
    residue_index = residue_index.cuda()
    _, structure_tokens = encoder.encode(coords, residue_index=residue_index)

    # Add BOS/EOS padding
    coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
    plddt = F.pad(plddt, (1, 1), value=0)
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    structure_tokens[:, 0] = 4098
    structure_tokens[:, -1] = 4097

    output = model.forward(
        structure_coords=coords, per_res_plddt=plddt, structure_tokens=structure_tokens
    )
    sequence_tokens = torch.argmax(output.sequence_logits, dim=-1)
    sequence = tokenizer.decode(sequence_tokens[0])
    print(sequence)


@torch.no_grad()
def conditioned_prediction_example():
    tokenizers = get_esm3_model_tokenizers()

    model = ESM3_sm_open_v0("cuda")

    # PDB 1UTN
    sequence = "MKTFIFLALLGAAVAFPVDDDDKIVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN"
    tokens = tokenizers.sequence.encode(sequence)

    # Calculate the number of tokens to replace, excluding the first and last token
    num_to_replace = int((len(tokens) - 2) * 0.75)

    # Randomly select indices to replace, excluding the first and last index
    indices_to_replace = random.sample(range(1, len(tokens) - 1), num_to_replace)

    # Replace selected indices with 32
    assert tokenizers.sequence.mask_token_id is not None
    for idx in indices_to_replace:
        tokens[idx] = tokenizers.sequence.mask_token_id
    sequence_tokens = torch.tensor(tokens, dtype=torch.int64)

    function_annotations = [
        # Peptidase S1A, chymotrypsin family
        FunctionAnnotation(label="peptidase", start=100, end=114),
        FunctionAnnotation(label="chymotrypsin", start=190, end=202),
    ]
    function_tokens = tokenizers.function.tokenize(function_annotations, len(sequence))
    function_tokens = tokenizers.function.encode(function_tokens)

    function_tokens = function_tokens.cuda().unsqueeze(0)
    sequence_tokens = sequence_tokens.cuda().unsqueeze(0)

    output = model.forward(
        sequence_tokens=sequence_tokens, function_tokens=function_tokens
    )
    return sequence, output, sequence_tokens


@torch.no_grad()
def decode(sequence, output, sequence_tokens):
    # To save on VRAM, we load these in separate functions
    decoder = ESM3_structure_decoder_v0("cuda")
    function_decoder = ESM3_function_decoder_v0("cuda")
    function_tokenizer = EsmFunctionTokenizer()

    # Generally not recommended to just argmax the logits, decode iteratively!
    # For quick demonstration only:
    structure_tokens = torch.argmax(output.structure_logits, dim=-1)
    structure_tokens = (
        structure_tokens.where(sequence_tokens != 0, 4098)  # BOS
        .where(sequence_tokens != 2, 4097)  # EOS
        .where(sequence_tokens != 31, 4100)  # Chainbreak
    )

    bb_coords = (
        decoder.decode(
            structure_tokens,
            torch.ones_like(sequence_tokens),
            torch.zeros_like(sequence_tokens),
        )["bb_pred"]
        .detach()
        .cpu()
    )

    chain = ProteinChain.from_backbone_atom_coordinates(
        bb_coords, sequence="X" + sequence + "X"
    )
    chain.infer_oxygen().to_pdb("hello.pdb")

    # Function prediction
    p_none_threshold = 0.05
    log_p = F.log_softmax(output.function_logits[:, 1:-1, :], dim=3).squeeze(0)

    # Choose which positions have no predicted function.
    log_p_nones = log_p[:, :, function_tokenizer.vocab_to_index["<none>"]]
    p_none = torch.exp(log_p_nones).mean(dim=1)  # "Ensemble of <none> predictions"
    where_none = p_none > p_none_threshold  # (length,)

    log_p[~where_none, :, function_tokenizer.vocab_to_index["<none>"]] = -torch.inf
    function_token_ids = torch.argmax(log_p, dim=2)
    function_token_ids[where_none, :] = function_tokenizer.vocab_to_index["<none>"]

    predicted_function = function_decoder.decode(
        function_token_ids,
        tokenizer=function_tokenizer,
        annotation_threshold=0.1,
        annotation_min_length=5,
        annotation_gap_merge_max=3,
    )

    print("function prediction:")
    print(predicted_function["interpro_preds"].nonzero())
    print(predicted_function["function_keywords"])


if __name__ == "__main__":
    inverse_folding_example()

    sequence, output, sequence_tokens = conditioned_prediction_example()
    torch.cuda.empty_cache()
    # And then decode from tokenized representation to outputs:
    decode(sequence, output, sequence_tokens)
