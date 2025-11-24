from __future__ import annotations

import contextlib
from functools import partial
from typing import Callable

import attr
import einops
import torch
import torch.nn as nn
from attr import dataclass

from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinTensor,
    ForwardAndSampleOutput,
    ForwardTrackData,
    GenerationConfig,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
    SamplingConfig,
)
from esm.tokenization import TokenizerCollectionProtocol
from esm.utils import encoding
from esm.utils.constants import esm3 as C
from esm.utils.constants.models import ESM3_OPEN_SMALL, normalize_model_name
from esm.utils.decoding import decode_protein_tensor
from esm.utils.generation import (
    _batch_forward,
    _sample_per_prompt,
    _slice_tensor_dataclass,
    iterative_sampling_raw,
    iterative_sampling_tokens,
)
from esm.utils.misc import rbf
from esm.utils.sampling import (
    _BatchedESMProteinTensor,
    get_default_sampling_config,
    validate_sampling_config,
)
from esm.utils.structure.affine3d import build_affine3d_from_coordinates


@dataclass
class ESMOutput:
    sequence_logits: torch.Tensor
    structure_logits: torch.Tensor
    secondary_structure_logits: torch.Tensor
    sasa_logits: torch.Tensor
    function_logits: torch.Tensor
    residue_logits: torch.Tensor
    embeddings: torch.Tensor


class EncodeInputs(nn.Module):
    """
    Module for encoding input features in the ESM-3 model.

    Args:
        d_model (int): The dimensionality of the model's hidden states.
    """

    def __init__(self, d_model: int):
        super().__init__()

        # Sequence
        self.sequence_embed = nn.Embedding(64, d_model)
        # Mandatory information
        self.plddt_projection = nn.Linear(16, d_model)
        self.structure_per_res_plddt_projection = nn.Linear(16, d_model)

        # Structure
        self.structure_tokens_embed = nn.Embedding(4096 + 5, d_model)

        # "Structural" features
        self.ss8_embed = nn.Embedding(8 + 3, d_model)
        self.sasa_embed = nn.Embedding(16 + 3, d_model)

        # "Functional" features
        self.function_embed = nn.ModuleList(
            [nn.Embedding(260, d_model // 8, padding_idx=0) for _ in range(8)]
        )

        self.residue_embed = nn.EmbeddingBag(1478, d_model, mode="sum", padding_idx=0)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        average_plddt: torch.Tensor,
        per_res_plddt: torch.Tensor,
        ss8_tokens: torch.Tensor,
        sasa_tokens: torch.Tensor,
        function_tokens: torch.Tensor,
        residue_annotation_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sequence_embed = self.sequence_embed(sequence_tokens)

        rbf_16_fn = partial(rbf, v_min=0.0, v_max=1.0, n_bins=16)
        # the `masked_fill(padding_mask.unsqueeze(2), 0)` for the two below is unnecessary
        # as pad tokens never even interact with the "real" tokens (due to sequence_id)
        plddt_embed = self.plddt_projection(rbf_16_fn(average_plddt))
        structure_per_res_plddt = self.structure_per_res_plddt_projection(
            rbf_16_fn(per_res_plddt)
        )

        # Structure + "structural features" embeds
        structure_embed = self.structure_tokens_embed(structure_tokens)
        ss8_embed = self.ss8_embed(ss8_tokens)
        sasa_embed = self.sasa_embed(sasa_tokens)

        # "Functional" features embeds
        function_embed = torch.cat(
            [
                embed_fn(funcs)
                for embed_fn, funcs in zip(
                    self.function_embed, function_tokens.unbind(-1)
                )
            ],
            -1,
        )

        # Residue embeds
        B, L, N = residue_annotation_tokens.shape
        residue_embed = self.residue_embed(
            einops.rearrange(
                residue_annotation_tokens, "B L N -> (B L) N", B=B, L=L, N=N
            )
        )
        residue_embed = einops.rearrange(residue_embed, "(B L) D -> B L D", B=B, L=L)

        return (
            sequence_embed
            + plddt_embed
            + structure_per_res_plddt
            + structure_embed
            + ss8_embed
            + sasa_embed
            + function_embed
            + residue_embed
        )


class OutputHeads(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_head = RegressionHead(d_model, 64)
        self.structure_head = RegressionHead(d_model, 4096)
        self.ss8_head = RegressionHead(d_model, 8 + 3)
        self.sasa_head = RegressionHead(d_model, 16 + 3)
        self.function_head = RegressionHead(d_model, 260 * 8)
        self.residue_head = RegressionHead(d_model, 1478)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> ESMOutput:
        sequence_logits = self.sequence_head(x)
        structure_logits = self.structure_head(x)
        secondary_structure_logits = self.ss8_head(x)
        sasa_logits = self.sasa_head(x)
        function_logits = self.function_head(x)
        function_logits = einops.rearrange(function_logits, "... (k v) -> ... k v", k=8)

        residue_logits = self.residue_head(x)

        return ESMOutput(
            sequence_logits=sequence_logits,
            structure_logits=structure_logits,
            secondary_structure_logits=secondary_structure_logits,
            sasa_logits=sasa_logits,
            function_logits=function_logits,
            residue_logits=residue_logits,
            embeddings=embed,
        )


class ESM3(nn.Module, ESM3InferenceClient):
    """
    ESM3 model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        v_heads (int): The number of attention heads in the variational transformer layers.
        n_layers (int): The number of transformer layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int,
        n_layers: int,
        structure_encoder_fn: Callable[[torch.device | str], StructureTokenEncoder],
        structure_decoder_fn: Callable[[torch.device | str], StructureTokenDecoder],
        function_decoder_fn: Callable[[torch.device | str], FunctionTokenDecoder],
        tokenizers: TokenizerCollectionProtocol,
    ):
        super().__init__()
        self.encoder = EncodeInputs(d_model)
        self.transformer = TransformerStack(
            d_model, n_heads, v_heads, n_layers, mask_and_zero_frameless=True
        )
        self.output_heads = OutputHeads(d_model)

        self.structure_encoder_fn = structure_encoder_fn
        self.structure_decoder_fn = structure_decoder_fn
        self.function_decoder_fn = function_decoder_fn

        self._structure_encoder = None
        self._structure_decoder = None
        self._function_decoder = None

        self.tokenizers = tokenizers

    @classmethod
    def from_pretrained(
        cls, model_name: str = ESM3_OPEN_SMALL, device: torch.device | None = None
    ) -> ESM3:
        from esm.pretrained import load_local_model

        model_name = normalize_model_name(model_name)
        if not model_name:
            raise ValueError(f"Model name {model_name} is not a valid ESM3 model name.")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_local_model(model_name, device=device)
        if device.type != "cpu":
            model = model.to(torch.bfloat16)
        assert isinstance(model, ESM3)
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def raw_model(self):
        return self

    def get_structure_encoder(self) -> StructureTokenEncoder:
        if self._structure_encoder is None:
            self._structure_encoder = self.structure_encoder_fn(self.device)
        return self._structure_encoder

    def get_structure_decoder(self) -> StructureTokenDecoder:
        if self._structure_decoder is None:
            self._structure_decoder = self.structure_decoder_fn(self.device)
        return self._structure_decoder

    def get_function_decoder(self) -> FunctionTokenDecoder:
        if self._function_decoder is None:
            self._function_decoder = self.function_decoder_fn(self.device)
        return self._function_decoder

    def forward(
        self,
        *,
        sequence_tokens: torch.Tensor | None = None,
        structure_tokens: torch.Tensor | None = None,
        ss8_tokens: torch.Tensor | None = None,
        sasa_tokens: torch.Tensor | None = None,
        function_tokens: torch.Tensor | None = None,
        residue_annotation_tokens: torch.Tensor | None = None,
        average_plddt: torch.Tensor | None = None,
        per_res_plddt: torch.Tensor | None = None,
        structure_coords: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
    ) -> ESMOutput:
        """
        Performs forward pass through the ESM3 model. Check utils to see how to tokenize inputs from raw data.

        Args:
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            structure_tokens (torch.Tensor, optional): The structure tokens.
            ss8_tokens (torch.Tensor, optional): The secondary structure tokens.
            sasa_tokens (torch.Tensor, optional): The solvent accessible surface area tokens.
            function_tokens (torch.Tensor, optional): The function tokens.
            residue_annotation_tokens (torch.Tensor, optional): The residue annotation tokens.
            average_plddt (torch.Tensor, optional): The average plddt across the entire sequence.
            per_res_plddt (torch.Tensor, optional): The per residue plddt, if you want to specify exact plddts, use this,
                otherwise, use average_plddt.
            structure_coords (torch.Tensor, optional): The structure coordinates, in the form of (B, L, 3, 3).
            chain_id (torch.Tensor, optional): The chain ID
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            ESMOutput: The output of the ESM3 model.

        Raises:
            ValueError: If at least one of the inputs is None.

        """
        # Reasonable defaults:
        try:
            L, device = next(
                (x.shape[1], x.device)
                for x in [
                    sequence_tokens,
                    structure_tokens,
                    ss8_tokens,
                    sasa_tokens,
                    structure_coords,
                    function_tokens,
                    residue_annotation_tokens,
                ]
                if x is not None
            )
        except StopIteration:
            raise ValueError("At least one of the inputs must be non-None")

        t = self.tokenizers
        defaults = lambda x, tok: (
            torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
        )
        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
            )

        if function_tokens is None:
            function_tokens = torch.full(
                (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
            )

        if structure_coords is None:
            structure_coords = torch.full(
                (1, L, 3, 3), float("nan"), dtype=torch.float, device=device
            )

        structure_coords = structure_coords[
            ..., :3, :
        ]  # In case we pass in an atom14 or atom37 repr
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        assert structure_tokens is not None
        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )

        x = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        x, embedding, _ = self.transformer(
            x, sequence_id, affine, affine_mask, chain_id
        )
        return self.output_heads(x, embedding)

    # The following methods are for the ESM3InferenceClient interface
    def generate(self, input: ProteinType, config: GenerationConfig) -> ProteinType:
        """Wrap around batched generation."""
        proteins = self.batch_generate([input], [config])
        assert len(proteins) == 1
        return proteins[0]

    def batch_generate(
        self, inputs: list[ProteinType], configs: list[GenerationConfig]
    ) -> list[ProteinType]:
        assert len(inputs) == len(
            configs
        ), "Must have the same number of prompts and configs."

        if inputs == []:
            # Nothing to do.
            return []

        # Make sure prompts are of the same type.
        t = type(inputs[0])
        for i in range(1, len(inputs)):
            assert isinstance(inputs[i], t), (
                "Prompts must have the same type. Got "
                f"{t.__name__ and type(inputs[i]).__name__} instead."
            )

        if isinstance(inputs[0], ESMProtein):
            return iterative_sampling_raw(self, inputs, configs)  # type: ignore
        elif isinstance(inputs[0], ESMProteinTensor):
            return iterative_sampling_tokens(
                self,
                inputs,  # type: ignore
                configs,
                self.tokenizers,  # type: ignore
            )
        else:
            raise ValueError("Input must be an ESMProtein or ESMProteinTensor")

    def encode(self, input: ESMProtein) -> ESMProteinTensor:
        input = attr.evolve(input)  # Make a copy

        sequence_tokens = None
        structure_tokens = None
        secondary_structure_tokens = None
        sasa_tokens = None
        function_tokens = None
        residue_annotation_tokens = None

        coordinates = None

        if input.sequence is not None:
            sequence_tokens = encoding.tokenize_sequence(
                input.sequence, self.tokenizers.sequence, add_special_tokens=True
            )
        if input.secondary_structure is not None:
            secondary_structure_tokens = encoding.tokenize_secondary_structure(
                input.secondary_structure,
                self.tokenizers.secondary_structure,
                add_special_tokens=True,
            )
        if input.sasa is not None:
            sasa_tokens = encoding.tokenize_sasa(
                input.sasa, self.tokenizers.sasa, add_special_tokens=True
            )

        # Infer input length
        sequence_length = -1
        if sequence_tokens is not None:
            sequence_length = len(sequence_tokens)
        elif secondary_structure_tokens is not None:
            sequence_length = len(secondary_structure_tokens)
        elif sasa_tokens is not None:
            sequence_length = len(sasa_tokens)

        # Try to infer input length from structure data
        if input.coordinates is not None:
            coordinates, _, structure_tokens = encoding.tokenize_structure(
                input.coordinates,
                self.get_structure_encoder(),
                structure_tokenizer=self.tokenizers.structure,
                reference_sequence=input.sequence or "",
                add_special_tokens=True,
            )
            if sequence_length == -1:
                sequence_length = len(structure_tokens)

        if sequence_length == -1:
            raise ValueError(
                "Cannot infer input length from input data. Please provide one of: sequence, structure, secondary_structure, sasa.\n"
                "To condition on sequence length only, use ESM3LocalInferenceClient.get_default_sequence(sequence_length) to generate a default sequence input."
            )

        # Function and Residue annotations
        if input.function_annotations is not None:
            if input.sequence is None:
                reference_sequence = encoding.get_default_sequence(sequence_length - 2)
            else:
                reference_sequence = input.sequence
            (function_tokens, residue_annotation_tokens) = (
                encoding.tokenize_function_annotations(
                    input.function_annotations,
                    reference_sequence=reference_sequence,
                    function_tokenizer=self.tokenizers.function,
                    residue_annotation_tokenizer=self.tokenizers.residue_annotations,
                    add_special_tokens=True,
                )
            )

        return ESMProteinTensor(
            sequence=sequence_tokens,
            structure=structure_tokens,
            secondary_structure=secondary_structure_tokens,
            sasa=sasa_tokens,
            function=function_tokens,
            residue_annotations=residue_annotation_tokens,
            coordinates=coordinates,
            potential_sequence_of_concern=input.potential_sequence_of_concern,
        ).to(next(self.parameters()).device)

    def decode(self, input: ESMProteinTensor) -> ESMProtein:
        return decode_protein_tensor(
            input=input,
            tokenizers=self.tokenizers,
            structure_token_decoder=self.get_structure_decoder(),
            function_token_decoder=self.get_function_decoder(),
        )

    def logits(
        self,
        input: ESMProteinTensor | _BatchedESMProteinTensor,
        config: LogitsConfig = LogitsConfig(),
    ) -> LogitsOutput:
        if not isinstance(input, _BatchedESMProteinTensor):
            # Create batch dimension if necessary.
            input = _BatchedESMProteinTensor.from_protein_tensor(input)

        device = torch.device(input.device)

        # Default plddt conditioning for inference. 1s where coordinates are provided.
        if input.coordinates is None:
            per_res_plddt = None
        else:
            # 1.0 if all coordinates at specific indices have valid non-nan values.
            per_res_plddt = input.coordinates.isfinite().all(dim=-1).any(dim=-1).float()

        with (
            torch.no_grad(),  # Assume no gradients for now...
            torch.autocast(enabled=True, device_type=device.type, dtype=torch.bfloat16)  # type: ignore
            if device.type == "cuda"
            else contextlib.nullcontext(),
        ):
            output = self.forward(
                sequence_tokens=input.sequence,
                structure_tokens=input.structure,
                ss8_tokens=input.secondary_structure,
                sasa_tokens=input.sasa,
                function_tokens=input.function,
                residue_annotation_tokens=input.residue_annotations,
                average_plddt=torch.tensor(1.0, device=input.device),
                per_res_plddt=per_res_plddt,
                structure_coords=input.coordinates,
                chain_id=None,
                sequence_id=None,
            )

        output = ESMOutput(
            **{k: v.to(device).to(torch.float32) for k, v in vars(output).items()}
        )

        return LogitsOutput(
            logits=ForwardTrackData(
                sequence=output.sequence_logits if config.sequence else None,
                structure=output.structure_logits if config.structure else None,
                secondary_structure=output.secondary_structure_logits
                if config.secondary_structure
                else None,
                sasa=output.sasa_logits if config.sasa else None,
                function=output.function_logits if config.function else None,
            ),
            residue_annotation_logits=output.residue_logits
            if config.residue_annotations
            else None,
            embeddings=output.embeddings if config.return_embeddings else None,
        )

    def forward_and_sample(
        self, input: ESMProteinTensor, sampling_configuration: SamplingConfig
    ) -> ForwardAndSampleOutput:
        validate_sampling_config(sampling_configuration, on_invalid="warn")

        protein_tensor = attr.evolve(input)  # Make a copy

        device = next(self.parameters()).device

        sampling_config = sampling_configuration
        if sampling_config is None:
            sampling_config = get_default_sampling_config(self.tokenizers)

        # Initialize default values for missing tracks
        default_protein_tensor = ESMProteinTensor.empty(
            len(input) - 2, tokenizers=self.tokenizers, device=input.device
        )
        for track in attr.fields(ESMProteinTensor):
            if getattr(protein_tensor, track.name, None) is None:
                setattr(
                    protein_tensor,
                    track.name,
                    getattr(default_protein_tensor, track.name, None),
                )

        if len(protein_tensor) <= 0:
            raise ValueError("No input data provided")

        # Move input protein to proper device.
        batched_protein = _BatchedESMProteinTensor.from_protein_tensor(protein_tensor)
        batched_protein.to(device)

        logits_output: LogitsOutput = _batch_forward(self, batched_protein)
        forward_and_sample_out: ForwardAndSampleOutput = _sample_per_prompt(
            batched_protein, logits_output, sampling_config, self.tokenizers
        )

        # There is only 1 prompt to sample for.
        return _slice_tensor_dataclass(forward_and_sample_out, 0)
