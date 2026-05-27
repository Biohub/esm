from __future__ import annotations

import asyncio
import base64
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence, TypeAlias, Union, cast

import torch

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMCInferenceClient,
    ESMProtein,
    ESMProteinError,
    ESMProteinTensor,
    FoldingConfig,
    ForwardAndSampleOutput,
    ForwardTrackData,
    GenerationConfig,
    InverseFoldingConfig,
    LogitsConfig,
    LogitsOutput,
    ProteinComplex,
    ProteinType,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.sdk.base_forge_client import _BaseForgeInferenceClient
from esm.sdk.retry import retry_decorator
from esm.utils.constants.api import MIMETYPE_ES_PICKLE
from esm.utils.constants.models import ESMFOLD2_FAST, ESMFOLD2_MAX_MSA_SEQS
from esm.utils.misc import deserialize_tensors, maybe_list, maybe_tensor
from esm.utils.msa import MSA
from esm.utils.structure.input_builder import (
    ProteinInput,
    StructurePredictionInput,
    serialize_structure_prediction_input,
)
from esm.utils.structure.molecular_complex import (
    MolecularComplex,
    MolecularComplexResult,
)
from esm.utils.types import FunctionAnnotation

# fmt: off
MSAInput: TypeAlias = Union[
    MSA,
    None,
]
# fmt: on


def _list_to_function_annotations(l) -> list[FunctionAnnotation] | None:
    if l is None or len(l) <= 0:
        return None
    return [FunctionAnnotation(*t) for t in l]


def _maybe_logits(data: dict[str, Any], track: str):
    return maybe_tensor(data.get("logits", {}).get(track, None))


def _maybe_b64_decode(obj, return_bytes: bool):
    return (
        deserialize_tensors(base64.b64decode(obj))
        if return_bytes and isinstance(obj, str)
        else obj
    )


def _validate_protein_tensor_input(input):
    if isinstance(input, ESMProteinError):
        raise ValueError(
            f"Input must be an ESMProteinTensor instance, but received an ESMProteinError instead: {input.error_code} {input.error_msg}"
        )
    if not isinstance(input, ESMProteinTensor):
        raise ValueError(
            f"Input must be an ESMProteinTensor instance, but received {type(input)} instead. "
            "Use encode() API to encode an ESMProtein into ESMProteinTensor."
        )


class SequenceStructureForgeInferenceClient(_BaseForgeInferenceClient):
    def __init__(
        self,
        url: str = "https://biohub.ai",
        model: str | None = None,
        token: str = "",
        request_timeout: int | None = None,
        min_retry_wait: int = 1,
        max_retry_wait: int = 10,
        max_retry_attempts: int = 5,
    ):
        """
        Forge/Biohub Platform client for folding and inverse folding between sequence and structure spaces.

        Args:
            url: URL of the Forge/Biohub Platform server.
            model: Name of the model to be used for folding / inv folding.
            token: API token.
            request_timeout: Override the system default request timeout, in seconds.
        """
        super().__init__(
            model=model or "",
            url=url,
            token=token,
            request_timeout=request_timeout,
            min_retry_wait=min_retry_wait,
            max_retry_wait=max_retry_wait,
            max_retry_attempts=max_retry_attempts,
        )

    @staticmethod
    def _process_fold_request(
        sequence: str,
        msa: MSAInput,
        config: FoldingConfig,
        target_structure: ProteinComplex | None,
        model_name: str | None,
    ):
        request: dict[str, Any] = {"sequence": sequence}

        if msa is None:
            request["msa"] = None
        elif isinstance(msa, MSA):
            if model_name == ESMFOLD2_FAST:
                warnings.warn(
                    f"Model '{model_name}' was not trained with MSA and will ignore "
                    "the provided MSA. Remove the MSA to suppress this warning.",
                    UserWarning,
                    stacklevel=4,
                )
            elif len(msa.sequences) > ESMFOLD2_MAX_MSA_SEQS:
                warnings.warn(
                    f"MSA depth ({len(msa.sequences)}) exceeds the maximum of "
                    f"{ESMFOLD2_MAX_MSA_SEQS}. The MSA will be truncated to "
                    f"{ESMFOLD2_MAX_MSA_SEQS} sequences by the server.",
                    UserWarning,
                    stacklevel=4,
                )
            request["msa"] = {"sequences": msa.sequences}
        else:
            error_msg = f"MSA must be None or MSA. Got {msa} instead."
            raise AttributeError(error_msg)

        request["include_distogram"] = config.include_distogram
        request["include_pae"] = config.include_pae
        request["include_pair_chains_iptm"] = config.include_pair_chains_iptm
        request["num_sampling_steps"] = config.num_sampling_steps
        request["num_loops"] = config.num_loops
        request["include_embeddings"] = config.include_embeddings

        request["model"] = model_name

        return request

    @staticmethod
    def _process_fold_response(data: dict[str, Any], sequence: str) -> ESMProtein:
        return ESMProtein(
            sequence=sequence,
            coordinates=maybe_tensor(data["coordinates"], convert_none_to_nan=True),
            ptm=maybe_tensor(data.get("ptm", None)),
            plddt=maybe_tensor(data.get("plddt", None), convert_none_to_nan=True),
            pae=maybe_tensor(data.get("pae", None)),
            interface_ptm=maybe_tensor(data.get("interface_ptm", None)),
            pair_chains_iptm=maybe_tensor(data.get("pair_chains_iptm", None)),
            output_embedding_sequence=maybe_tensor(
                data.get("output_embedding_sequence", None), convert_none_to_nan=True
            ),
            output_embedding_pair_pooled=maybe_tensor(
                data.get("output_embedding_pair_pooled", None), convert_none_to_nan=True
            ),
            residue_index=maybe_tensor(
                data.get("residue_index", None), convert_none_to_nan=True
            ),
            entity_id=maybe_tensor(
                data.get("entity_id", None), convert_none_to_nan=True
            ),
        )

    @staticmethod
    def process_inverse_fold_request(
        coordinates: torch.Tensor,
        sequence: str | None,
        config: InverseFoldingConfig,
        model_name: str | None,
    ):
        inverse_folding_config = {
            "invalid_ids": config.invalid_ids,
            "temperature": config.temperature,
        }
        request = {
            "coordinates": maybe_list(coordinates, convert_nan_to_none=True),
            "inverse_folding_config": inverse_folding_config,
            "sequence": sequence,
        }
        if model_name is not None:
            request["model"] = model_name

        return request

    @retry_decorator
    async def async_fold(
        self,
        sequence: str,
        potential_sequence_of_concern: bool = False,
        model_name: str | None = None,
        msa: MSAInput = None,
        config: FoldingConfig = FoldingConfig(),
        target_structure: ProteinComplex | None = None,
    ) -> ESMProtein | ESMProteinError:
        """Predict coordinates for a protein sequence.

        Args:
            sequence: Protein sequence to be folded.
            model_name: Override the client level model name if needed.
            msa: Optional multi-sequence alignment data that may help some models fold the sequence.
            config: Optional configuration for folding parameters.
            target_structure: Optional target structure to use for distogram conditioning.

        Deprecated:
            potential_sequence_of_concern: this parameter is largely deprecated
                and ignored by the folding endpoint.
        """
        del potential_sequence_of_concern

        request = self._process_fold_request(
            sequence,
            msa,
            config,
            target_structure,
            model_name if model_name is not None else self.model,
        )

        try:
            data = await self._async_post("fold", request)
        except ESMProteinError as e:
            return e

        return self._process_fold_response(data, sequence)

    @retry_decorator
    def fold(
        self,
        sequence: str,
        potential_sequence_of_concern: bool = False,
        model_name: str | None = None,
        msa: MSAInput = None,
        config: FoldingConfig = FoldingConfig(),
        target_structure: ProteinComplex | None = None,
    ) -> ESMProtein | ESMProteinError:
        """Predict coordinates for a protein sequence.

        Args:
            sequence: Protein sequence to be folded.
            model_name: Override the client level model name if needed.
            msa: Optional multi-sequence alignment data that may help some models fold the sequence.
            config: Optional configuration for folding parameters.
            target_structure: Optional target structure to use for distogram conditioning.

        Deprecated:
            potential_sequence_of_concern: this parameter is largely deprecated
                and ignored by the folding endpoint.
        """

        del potential_sequence_of_concern

        request = self._process_fold_request(
            sequence,
            msa,
            config,
            target_structure,
            model_name if model_name is not None else self.model,
        )

        try:
            data = self._post("fold", request)
        except ESMProteinError as e:
            return e

        return self._process_fold_response(data, sequence)

    @retry_decorator
    async def async_fold_all_atom(
        self,
        all_atom_input: StructurePredictionInput,
        config: FoldingConfig = FoldingConfig(),
        model_name: str | None = None,
    ) -> MolecularComplexResult | list[MolecularComplexResult] | ESMProteinError:
        """Fold a molecular complex containing proteins, nucleic acids, and/or ligands.

        Args:
            all_atom_input: StructurePredictionInput containing sequences for different molecule types
            config: Optional configuration for folding parameters.
            model_name: Override the client level model name if needed
        """
        request = self._process_fold_all_atom_request(
            all_atom_input, config, model_name if model_name is not None else self.model
        )

        try:
            data = await self._async_post("fold_all_atom", request)
        except ESMProteinError as e:
            return e

        return self._process_fold_all_atom_response(data)

    @retry_decorator
    def fold_all_atom(
        self,
        all_atom_input: StructurePredictionInput,
        model_name: str | None = None,
        config: FoldingConfig = FoldingConfig(),
    ) -> MolecularComplexResult | list[MolecularComplexResult] | ESMProteinError:
        """Predict coordinates for a molecular complex containing proteins, dna, rna, and/or ligands.

        Args:
            all_atom_input: StructurePredictionInput containing sequences for different molecule types
            model_name: Override the client level model name if needed
            config: Optional configuration for folding parameters.
        """
        request = self._process_fold_all_atom_request(
            all_atom_input, config, model_name if model_name is not None else self.model
        )

        try:
            data = self._post("fold_all_atom", request)
        except ESMProteinError as e:
            return e

        return self._process_fold_all_atom_response(data)

    @staticmethod
    def _process_fold_all_atom_request(
        all_atom_input: StructurePredictionInput,
        config: FoldingConfig,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        for seq in all_atom_input.sequences:
            if not isinstance(seq, ProteinInput) or seq.msa is None:
                continue
            if model_name == ESMFOLD2_FAST:
                warnings.warn(
                    f"Model '{model_name}' was not trained with MSA and will ignore "
                    "any MSA provided in ProteinInput. Remove the MSA to suppress this warning.",
                    UserWarning,
                    stacklevel=4,
                )
                break
            if (
                isinstance(seq.msa, MSA)
                and len(seq.msa.sequences) > ESMFOLD2_MAX_MSA_SEQS
            ):
                chain_id = seq.id if seq.id is not None else "unknown"
                warnings.warn(
                    f"MSA depth ({len(seq.msa.sequences)}) for chain '{chain_id}' "
                    f"exceeds the maximum of {ESMFOLD2_MAX_MSA_SEQS}. The MSA will "
                    f"be truncated to {ESMFOLD2_MAX_MSA_SEQS} sequences by the server.",
                    UserWarning,
                    stacklevel=4,
                )

        request: dict[str, Any] = {
            "all_atom_input": serialize_structure_prediction_input(all_atom_input),
            "model": model_name,
        }

        request["include_distogram"] = config.include_distogram
        request["include_pae"] = config.include_pae
        request["num_sampling_steps"] = config.num_sampling_steps
        request["num_loops"] = config.num_loops
        request["include_embeddings"] = config.include_embeddings

        return request

    @staticmethod
    def _process_fold_all_atom_response(data: dict[str, Any]) -> MolecularComplexResult:
        complex_data = data.get("complex")
        molecular_complex = MolecularComplex.from_state_dict(complex_data)
        return MolecularComplexResult(
            complex=molecular_complex,
            plddt=maybe_tensor(data.get("plddt"), convert_none_to_nan=True),
            ptm=data.get("ptm", None),
            pae=maybe_tensor(data.get("pae"), convert_none_to_nan=True),
            iptm=data.get("interface_ptm", None),
            output_embedding_sequence=maybe_tensor(
                data.get("output_embedding_sequence"), convert_none_to_nan=True
            ),
            output_embedding_pair_pooled=maybe_tensor(
                data.get("output_embedding_pair_pooled"), convert_none_to_nan=True
            ),
            residue_index=maybe_tensor(
                data.get("residue_index"), convert_none_to_nan=True
            ),
            entity_id=maybe_tensor(data.get("entity_id"), convert_none_to_nan=True),
            distogram=maybe_tensor(data.get("distogram"), convert_none_to_nan=True),
        )

    @retry_decorator
    async def async_inverse_fold(
        self,
        coordinates: torch.Tensor,
        config: InverseFoldingConfig,
        potential_sequence_of_concern: bool,
        sequence: str | None = None,
        model_name: str | None = None,
    ) -> ESMProtein | ESMProteinError:
        """Generate protein sequence from its structure.

        This endpoint is only supported by generative models like ESM3.

        Args:
            coordinates: Protein sequence coordinates to be inversely folded.
            config: Configurations related to inverse folding generation.
            potential_sequence_of_concern: Self disclosed potential_of_concern bit.
                Requires special permission to use.
            model_name: Override the client level model name if needed.
        """
        request = self.process_inverse_fold_request(
            coordinates,
            sequence,
            config,
            model_name if model_name is not None else self.model,
        )

        try:
            data = await self._async_post(
                "inverse_fold", request, potential_sequence_of_concern
            )
        except ESMProteinError as e:
            return e

        return ESMProtein(sequence=data["sequence"])

    @retry_decorator
    def inverse_fold(
        self,
        coordinates: torch.Tensor,
        config: InverseFoldingConfig,
        potential_sequence_of_concern: bool,
        sequence: str | None = None,
        model_name: str | None = None,
    ) -> ESMProtein | ESMProteinError:
        """Generate protein sequence from its structure.

        This endpoint is only supported by generative models like ESM3.

        Args:
            coordinates: Protein sequence coordinates to be inversely folded.
            config: Configurations related to inverse folding generation.
            potential_sequence_of_concern: Self disclosed potential_of_concern bit.
                Requires special permission to use.
            model_name: Override the client level model name if needed.
        """
        request = self.process_inverse_fold_request(
            coordinates,
            sequence,
            config,
            model_name if model_name is not None else self.model,
        )

        try:
            data = self._post("inverse_fold", request, potential_sequence_of_concern)
        except ESMProteinError as e:
            return e

        return ESMProtein(sequence=data["sequence"])


class ESM3ForgeInferenceClient(ESM3InferenceClient, _BaseForgeInferenceClient):
    def __init__(
        self,
        model: str,
        url: str = "https://biohub.ai",
        token: str = "",
        request_timeout: int | None = None,
        min_retry_wait: int = 1,
        max_retry_wait: int = 10,
        max_retry_attempts: int = 5,
    ):
        ESM3InferenceClient.__init__(self)
        _BaseForgeInferenceClient.__init__(
            self,
            model,
            url,
            token,
            request_timeout,
            min_retry_wait,
            max_retry_wait,
            max_retry_attempts,
        )

    @staticmethod
    def _process_generate_protein_request(
        input: ESMProtein, config: GenerationConfig, model_name: str
    ) -> dict[str, Any]:
        req = {}
        req["sequence"] = input.sequence
        req["secondary_structure"] = input.secondary_structure
        req["sasa"] = input.sasa
        if input.function_annotations is not None:
            req["function"] = [x.to_tuple() for x in input.function_annotations]
        req["coordinates"] = maybe_list(input.coordinates, convert_nan_to_none=True)

        request = {
            "model": model_name,
            "inputs": req,
            "track": config.track,
            "invalid_ids": config.invalid_ids,
            "schedule": config.schedule,
            "num_steps": config.num_steps,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "condition_on_coordinates_only": config.condition_on_coordinates_only,
            "strategy": config.strategy,
            "temperature_annealing": config.temperature_annealing,
        }
        return request

    @staticmethod
    def _process_generate_protein_tensor_request(
        input: ESMProteinTensor, config: GenerationConfig, model_name: str
    ) -> dict[str, Any]:
        req = {}
        req["sequence"] = maybe_list(input.sequence)
        req["structure"] = maybe_list(input.structure)
        req["secondary_structure"] = maybe_list(input.secondary_structure)
        req["sasa"] = maybe_list(input.sasa)
        req["function"] = maybe_list(input.function)
        req["coordinates"] = maybe_list(input.coordinates, convert_nan_to_none=True)
        req["residue_annotation"] = maybe_list(input.residue_annotations)

        request = {
            "model": model_name,
            "inputs": req,
            "track": config.track,
            "invalid_ids": config.invalid_ids,
            "schedule": config.schedule,
            "num_steps": config.num_steps,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "condition_on_coordinates_only": config.condition_on_coordinates_only,
            "strategy": config.strategy,
            "temperature_annealing": config.temperature_annealing,
        }
        return request

    @staticmethod
    def _process_generate_protein_response(data: dict[str, Any]) -> ESMProtein:
        return ESMProtein(
            sequence=data["outputs"]["sequence"],
            secondary_structure=data["outputs"]["secondary_structure"],
            sasa=data["outputs"]["sasa"],
            function_annotations=_list_to_function_annotations(
                data["outputs"]["function"]
            ),
            coordinates=maybe_tensor(
                data["outputs"]["coordinates"], convert_none_to_nan=True
            ),
            plddt=maybe_tensor(data["outputs"]["plddt"]),
            ptm=maybe_tensor(data["outputs"]["ptm"]),
        )

    @staticmethod
    def _process_generate_protein_tensor_response(
        data: dict[str, Any],
    ) -> ESMProteinTensor:
        def _field_to_tensor(field, convert_none_to_nan: bool = False):
            if field not in data["outputs"]:
                return None
            return maybe_tensor(
                data["outputs"][field], convert_none_to_nan=convert_none_to_nan
            )

        return ESMProteinTensor(
            sequence=_field_to_tensor("sequence"),
            structure=_field_to_tensor("structure"),
            secondary_structure=_field_to_tensor("secondary_structure"),
            sasa=_field_to_tensor("sasa"),
            function=_field_to_tensor("function"),
            residue_annotations=_field_to_tensor("residue_annotation"),
            coordinates=_field_to_tensor("coordinates", convert_none_to_nan=True),
        )

    @staticmethod
    def _process_forward_and_sample_request(
        input: ESMProteinTensor, sampling_configuration: SamplingConfig, model_name: str
    ) -> dict[str, Any]:
        from esm.utils.sampling import validate_sampling_config

        _validate_protein_tensor_input(input)
        validate_sampling_config(sampling_configuration, on_invalid="raise")

        req = {}
        sampling_config = {}
        embedding_config = {
            "sequence": sampling_configuration.return_mean_embedding,
            "per_residue": sampling_configuration.return_per_residue_embeddings,
        }

        req["sequence"] = maybe_list(input.sequence)
        req["structure"] = maybe_list(input.structure)
        req["secondary_structure"] = maybe_list(input.secondary_structure)
        req["sasa"] = maybe_list(input.sasa)
        req["function"] = maybe_list(input.function)
        req["coordinates"] = maybe_list(input.coordinates, convert_nan_to_none=True)
        req["residue_annotation"] = maybe_list(input.residue_annotations)

        def do_track(t: str):
            track: SamplingTrackConfig | None
            if (track := getattr(sampling_configuration, t, None)) is None:
                sampling_config[t] = None
            else:
                sampling_config[t] = {
                    "temperature": track.temperature,
                    "top_p": track.top_p,
                    "only_sample_masked_tokens": track.only_sample_masked_tokens,
                    "invalid_ids": track.invalid_ids,
                    "topk_logprobs": track.topk_logprobs,
                }

        do_track("sequence")
        do_track("structure")
        do_track("secondary_structure")
        do_track("sasa")
        do_track("function")

        request = {
            "model": model_name,
            "inputs": req,
            "sampling_config": sampling_config,
            "embedding_config": embedding_config,
        }

        return request

    @staticmethod
    def _process_forward_and_sample_response(
        data: dict[str, Any],
    ) -> ForwardAndSampleOutput:
        def get(k, field):
            if data[k] is None:
                return None
            v = data[k][field]
            return torch.tensor(v) if v is not None else None

        tokens = ESMProteinTensor(
            sequence=get("sequence", "tokens"),
            structure=get("structure", "tokens"),
            secondary_structure=get("secondary_structure", "tokens"),
            sasa=get("sasa", "tokens"),
            function=get("function", "tokens"),
        )

        def get_track(field):
            return ForwardTrackData(
                sequence=get("sequence", field),
                structure=get("structure", field),
                secondary_structure=get("secondary_structure", field),
                sasa=get("sasa", field),
                function=get("function", field),
            )

        def operate_on_track(track: ForwardTrackData, fn):
            apply = lambda x: fn(x) if x is not None else None
            return ForwardTrackData(
                sequence=apply(track.sequence),
                structure=apply(track.structure),
                secondary_structure=apply(track.secondary_structure),
                sasa=apply(track.sasa),
                function=apply(track.function),
            )

        logprob = get_track("logprobs")
        return ForwardAndSampleOutput(
            protein_tensor=tokens,
            logprob=logprob,
            prob=operate_on_track(logprob, torch.exp),
            entropy=get_track("entropy"),
            topk_logprob=get_track("topk_logprobs"),
            topk_tokens=get_track("topk_tokens"),
            per_residue_embedding=data["embeddings"]["per_residue"],
            mean_embedding=data["embeddings"]["sequence"],
        )

    @staticmethod
    def _process_encode_request(input: ESMProtein, model_name: str) -> dict[str, Any]:
        tracks = {}
        tracks["sequence"] = input.sequence
        tracks["secondary_structure"] = input.secondary_structure
        tracks["sasa"] = input.sasa
        if input.function_annotations is not None:
            tracks["function"] = [x.to_tuple() for x in input.function_annotations]
        tracks["coordinates"] = maybe_list(input.coordinates, convert_nan_to_none=True)

        request = {"inputs": tracks, "model": model_name}
        return request

    @staticmethod
    def _process_encode_response(data: dict[str, Any]) -> ESMProteinTensor:
        return ESMProteinTensor(
            sequence=maybe_tensor(data["outputs"]["sequence"]),
            structure=maybe_tensor(data["outputs"]["structure"]),
            coordinates=maybe_tensor(
                data["outputs"]["coordinates"], convert_none_to_nan=True
            ),
            secondary_structure=maybe_tensor(data["outputs"]["secondary_structure"]),
            sasa=maybe_tensor(data["outputs"]["sasa"]),
            function=maybe_tensor(data["outputs"]["function"]),
            residue_annotations=maybe_tensor(data["outputs"]["residue_annotation"]),
            potential_sequence_of_concern=data["potential_sequence_of_concern"],
        )

    @staticmethod
    def _process_decode_request(
        input: ESMProteinTensor, model_name: str
    ) -> dict[str, Any]:
        _validate_protein_tensor_input(input)

        tokens = {}
        tokens["sequence"] = maybe_list(input.sequence)
        tokens["structure"] = maybe_list(input.structure)
        tokens["secondary_structure"] = maybe_list(input.secondary_structure)
        tokens["sasa"] = maybe_list(input.sasa)
        tokens["function"] = maybe_list(input.function)
        tokens["residue_annotation"] = maybe_list(input.residue_annotations)
        tokens["coordinates"] = maybe_list(input.coordinates, convert_nan_to_none=True)

        request = {"model": model_name, "inputs": tokens}
        return request

    @staticmethod
    def _process_decode_response(data: dict[str, Any]) -> ESMProtein:
        return ESMProtein(
            sequence=data["outputs"]["sequence"],
            secondary_structure=data["outputs"]["secondary_structure"],
            sasa=data["outputs"]["sasa"],
            function_annotations=_list_to_function_annotations(
                data["outputs"]["function"]
            ),
            coordinates=maybe_tensor(
                data["outputs"]["coordinates"], convert_none_to_nan=True
            ),
            plddt=maybe_tensor(data["outputs"]["plddt"]),
            ptm=maybe_tensor(data["outputs"]["ptm"]),
        )

    @staticmethod
    def _process_logits_request(
        input: ESMProteinTensor, config: LogitsConfig, model_name: str
    ) -> dict[str, Any]:
        _validate_protein_tensor_input(input)
        # Note: using raw model forwards is discouraged because of the byte size
        # of the logits.
        # Please use forward_and_sample instead.
        req = {}
        req["sequence"] = maybe_list(input.sequence)
        req["structure"] = maybe_list(input.structure)
        req["secondary_structure"] = maybe_list(input.secondary_structure)
        req["sasa"] = maybe_list(input.sasa)
        req["function"] = maybe_list(input.function)
        req["coordinates"] = maybe_list(input.coordinates, convert_nan_to_none=True)
        req["residue_annotation"] = maybe_list(input.residue_annotations)

        logits_config = {
            "sequence": config.sequence,
            "structure": config.structure,
            "secondary_structure": config.secondary_structure,
            "sasa": config.sasa,
            "function": config.function,
            "residue_annotations": config.residue_annotations,
            "return_embeddings": config.return_embeddings,
            "return_mean_embedding": config.return_mean_embedding,
            "return_hidden_states": config.return_hidden_states,
            "return_mean_hidden_states": config.return_mean_hidden_states,
            "ith_hidden_layer": config.ith_hidden_layer,
        }

        request = {"model": model_name, "inputs": req, "logits_config": logits_config}
        return request

    @staticmethod
    def _process_logits_response(
        data: dict[str, Any], return_bytes: bool
    ) -> LogitsOutput:
        logits = _maybe_b64_decode(data["logits"], return_bytes)
        data["logits"] = dict(logits) if logits is not None else logits
        data["embeddings"] = _maybe_b64_decode(data["embeddings"], return_bytes)
        data["hidden_states"] = _maybe_b64_decode(data["hidden_states"], return_bytes)

        return LogitsOutput(
            logits=ForwardTrackData(
                sequence=_maybe_logits(data, "sequence"),
                structure=_maybe_logits(data, "structure"),
                secondary_structure=_maybe_logits(data, "secondary_structure"),
                sasa=_maybe_logits(data, "sasa"),
                function=_maybe_logits(data, "function"),
            ),
            embeddings=maybe_tensor(data["embeddings"]),
            mean_embedding=data["mean_embedding"],
            residue_annotation_logits=_maybe_logits(data, "residue_annotation"),
            hidden_states=maybe_tensor(data["hidden_states"]),
            mean_hidden_state=maybe_tensor(data["mean_hidden_state"]),
        )

    @retry_decorator
    async def async_generate(
        self, input: ProteinType, config: GenerationConfig
    ) -> ProteinType:
        if isinstance(input, ESMProteinError):
            raise ValueError(
                f"Input must be an ESMProtein or ESMProteinTensor instance, but received an ESMProteinError instead: {input.error_code} {input.error_msg}"
            )
        assert isinstance(input, ESMProtein) or isinstance(input, ESMProteinTensor)
        if input.sequence is not None and config.num_steps > len(input.sequence):
            config.num_steps = len(input.sequence)
            print(
                "Warning: num_steps cannot exceed sequence length. Setting num_steps to sequence length."
            )
        if isinstance(input, ESMProtein):
            output = await self.__async_generate_protein(input, config)
        elif isinstance(input, ESMProteinTensor):
            output = await self.__async_generate_protein_tensor(input, config)
        else:
            return ESMProteinError(
                error_code=500, error_msg=f"Unknown input type {type(input)}"
            )

        if (
            isinstance(output, ESMProtein)
            and isinstance(input, ESMProtein)
            and config.track not in ["function", "residue_annotations"]
        ):
            # Function and residue annotation encoding/decoding is lossy
            # There is no guarantee that decoding encoded tokens will yield the same input
            output.function_annotations = input.function_annotations

        return output

    @retry_decorator
    def generate(self, input: ProteinType, config: GenerationConfig) -> ProteinType:
        if isinstance(input, ESMProteinError):
            raise ValueError(
                f"Input must be an ESMProtein or ESMProteinTensor instance, but received an ESMProteinError instead: {input.error_code} {input.error_msg}"
            )
        assert isinstance(input, ESMProtein) or isinstance(input, ESMProteinTensor)
        if input.sequence is not None and config.num_steps > len(input.sequence):
            config.num_steps = len(input.sequence)
            print(
                "Warning: num_steps cannot exceed sequence length. Setting num_steps to sequence length."
            )
        if isinstance(input, ESMProtein):
            output = self.__generate_protein(input, config)
        elif isinstance(input, ESMProteinTensor):
            output = self.__generate_protein_tensor(input, config)
        else:
            return ESMProteinError(
                error_code=500, error_msg=f"Unknown input type {type(input)}"
            )

        if (
            isinstance(output, ESMProtein)
            and isinstance(input, ESMProtein)
            and config.track not in ["function", "residue_annotations"]
        ):
            # Function and residue annotation encoding/decoding is lossy
            # There is no guarantee that decoding encoded tokens will yield the same input
            output.function_annotations = input.function_annotations

        return output

    async def async_batch_generate(
        self, inputs: Sequence[ProteinType], configs: Sequence[GenerationConfig]
    ) -> Sequence[ProteinType]:
        """Forge/Biohub Platform supports auto-batching. So batch_generate() for the Forge/Biohub Platform client
        is as simple as running a collection of generate() in parallel using asyncio.
        """

        async def safe_generate(protein, config):
            try:
                res = self.async_generate(protein, config)
                return await res
            except ESMProteinError as e:
                return e

        tasks = [
            safe_generate(protein, config) for protein, config in zip(inputs, configs)
        ]
        results = await asyncio.gather(*tasks)
        return results

    def batch_generate(
        self, inputs: Sequence[ProteinType], configs: Sequence[GenerationConfig]
    ) -> Sequence[ProteinType]:
        """Forge/Biohub Platform supports auto-batching. So batch_generate() for the Forge/Biohub Platform client
        is as simple as running a collection of generate() in parallel using a threadpool.
        """
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.generate, protein, config)
                for protein, config in zip(inputs, configs)
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(ESMProteinError(error_code=500, error_msg=str(e)))
        return results

    async def __async_generate_protein(
        self, input: ESMProtein, config: GenerationConfig
    ) -> ESMProtein | ESMProteinError:
        request = self._process_generate_protein_request(input, config, self.model)
        try:
            data = await self._async_post(
                "generate", request, input.potential_sequence_of_concern
            )
        except ESMProteinError as e:
            return e

        return self._process_generate_protein_response(data)

    async def __async_generate_protein_tensor(
        self, input: ESMProteinTensor, config: GenerationConfig
    ) -> ESMProteinTensor | ESMProteinError:
        request = self._process_generate_protein_tensor_request(
            input, config, self.model
        )

        try:
            data = await self._async_post(
                "generate_tensor", request, input.potential_sequence_of_concern
            )
        except ESMProteinError as e:
            return e

        return self._process_generate_protein_tensor_response(data)

    def __generate_protein(
        self, input: ESMProtein, config: GenerationConfig
    ) -> ESMProtein | ESMProteinError:
        request = self._process_generate_protein_request(input, config, self.model)
        try:
            data = self._post("generate", request, input.potential_sequence_of_concern)
        except ESMProteinError as e:
            return e

        return self._process_generate_protein_response(data)

    def __generate_protein_tensor(
        self, input: ESMProteinTensor, config: GenerationConfig
    ) -> ESMProteinTensor | ESMProteinError:
        request = self._process_generate_protein_tensor_request(
            input, config, self.model
        )

        try:
            data = self._post(
                "generate_tensor", request, input.potential_sequence_of_concern
            )
        except ESMProteinError as e:
            return e

        return self._process_generate_protein_tensor_response(data)

    @retry_decorator
    async def async_forward_and_sample(
        self, input: ESMProteinTensor, sampling_configuration: SamplingConfig
    ) -> ForwardAndSampleOutput | ESMProteinError:
        request = self._process_forward_and_sample_request(
            input, sampling_configuration, self.model
        )
        try:
            data = await self._async_post(
                "forward_and_sample",
                request,
                input.potential_sequence_of_concern,
                headers={
                    "Accept": f"{MIMETYPE_ES_PICKLE};protocol={pickle.HIGHEST_PROTOCOL}, application/json"
                },
            )
        except ESMProteinError as e:
            return e

        return self._process_forward_and_sample_response(data)

    @retry_decorator
    def forward_and_sample(
        self, input: ESMProteinTensor, sampling_configuration: SamplingConfig
    ) -> ForwardAndSampleOutput | ESMProteinError:
        request = self._process_forward_and_sample_request(
            input, sampling_configuration, self.model
        )
        try:
            data = self._post(
                "forward_and_sample",
                request,
                input.potential_sequence_of_concern,
                headers={
                    "Accept": f"{MIMETYPE_ES_PICKLE};protocol={pickle.HIGHEST_PROTOCOL}, application/json"
                },
            )
        except ESMProteinError as e:
            return e

        return self._process_forward_and_sample_response(data)

    @retry_decorator
    async def async_encode(
        self, input: ESMProtein
    ) -> ESMProteinTensor | ESMProteinError:
        request = self._process_encode_request(input, self.model)

        try:
            data = await self._async_post(
                "encode", request, input.potential_sequence_of_concern
            )
        except ESMProteinError as e:
            return e

        return self._process_encode_response(data)

    @retry_decorator
    def encode(self, input: ESMProtein) -> ESMProteinTensor | ESMProteinError:
        request = self._process_encode_request(input, self.model)

        try:
            data = self._post("encode", request, input.potential_sequence_of_concern)
        except ESMProteinError as e:
            return e

        return self._process_encode_response(data)

    @retry_decorator
    async def async_decode(
        self, input: ESMProteinTensor
    ) -> ESMProtein | ESMProteinError:
        request = self._process_decode_request(input, self.model)

        try:
            data = await self._async_post(
                "decode", request, input.potential_sequence_of_concern
            )
        except ESMProteinError as e:
            return e

        return self._process_decode_response(data)

    @retry_decorator
    def decode(self, input: ESMProteinTensor) -> ESMProtein | ESMProteinError:
        request = self._process_decode_request(input, self.model)

        try:
            data = self._post("decode", request, input.potential_sequence_of_concern)
        except ESMProteinError as e:
            return e

        return self._process_decode_response(data)

    @retry_decorator
    async def async_logits(
        self,
        input: ESMProteinTensor,
        config: LogitsConfig = LogitsConfig(),
        return_bytes: bool = True,
    ) -> LogitsOutput | ESMProteinError:
        request = self._process_logits_request(input, config, self.model)

        try:
            data = await self._async_post(
                "logits",
                request,
                input.potential_sequence_of_concern,
                return_bytes=return_bytes,
                headers={
                    "Accept": f"{MIMETYPE_ES_PICKLE};protocol={pickle.HIGHEST_PROTOCOL}, application/json"
                },
            )
        except ESMProteinError as e:
            return e

        return self._process_logits_response(data, return_bytes)

    @retry_decorator
    def logits(
        self,
        input: ESMProteinTensor,
        config: LogitsConfig = LogitsConfig(),
        return_bytes: bool = True,
    ) -> LogitsOutput | ESMProteinError:
        request = self._process_logits_request(input, config, self.model)

        try:
            data = self._post(
                "logits",
                request,
                input.potential_sequence_of_concern,
                return_bytes=return_bytes,
                headers={
                    "Accept": f"{MIMETYPE_ES_PICKLE};protocol={pickle.HIGHEST_PROTOCOL}, application/json"
                },
            )
        except ESMProteinError as e:
            return e

        return self._process_logits_response(data, return_bytes)

    @property
    def raw_model(self):
        raise NotImplementedError(
            f"Can not get underlying remote model {self.model} from a Forge/Biohub Platform client."
        )


class ESMCForgeInferenceClient(ESMCInferenceClient, _BaseForgeInferenceClient):
    def __init__(
        self,
        model: str,
        url: str = "https://biohub.ai",
        token: str = "",
        request_timeout: int | None = None,
        min_retry_wait: int = 1,
        max_retry_wait: int = 10,
        max_retry_attempts: int = 5,
    ):
        ESMCInferenceClient.__init__(self)
        _BaseForgeInferenceClient.__init__(
            self,
            model,
            url,
            token,
            request_timeout,
            min_retry_wait,
            max_retry_wait,
            max_retry_attempts,
        )

    @staticmethod
    def _process_logits_request(
        input: ESMProteinTensor, config: LogitsConfig, model_name: str
    ) -> dict[str, Any]:
        _validate_protein_tensor_input(input)

        req = {}
        req["sequence"] = maybe_list(input.sequence)

        logits_config = {
            "sequence": config.sequence,
            "return_embeddings": config.return_embeddings,
            "return_mean_embedding": config.return_mean_embedding,
            "return_mean_hidden_states": config.return_mean_hidden_states,
            "return_hidden_states": config.return_hidden_states,
            "ith_hidden_layer": config.ith_hidden_layer,
            "sae_config": {
                "models": config.sae_config.models,
                "normalize_features": config.sae_config.normalize_features,
            }
            if config.sae_config
            else None,
        }
        request = {"model": model_name, "inputs": req, "logits_config": logits_config}
        return request

    @staticmethod
    def _process_logits_response(
        data: dict[str, Any], return_bytes: bool
    ) -> LogitsOutput:
        logits = _maybe_b64_decode(data["logits"], return_bytes)
        data["logits"] = dict(logits) if logits is not None else logits
        data["embeddings"] = _maybe_b64_decode(data["embeddings"], return_bytes)
        data["hidden_states"] = _maybe_b64_decode(data["hidden_states"], return_bytes)

        # sae outputs are always encoded
        # NOTE: leave this intact for application/json since it's always
        # base64 bytes, even with return_bytes=False
        def _b64_decode(obj):
            return (
                deserialize_tensors(base64.b64decode(obj)) if obj is not None else obj
            )

        sae_outputs = data["sae_outputs"]
        if isinstance(sae_outputs, str):
            # sae outputs are always encoded for application/json
            sae_outputs = _b64_decode(sae_outputs)
            sae_outputs = (
                {k: v.to(torch.float32) for k, v in sae_outputs.items()}
                if sae_outputs
                else None
            )
            sae_outputs = cast(dict[str, torch.Tensor] | None, sae_outputs)
        output = LogitsOutput(
            logits=ForwardTrackData(sequence=_maybe_logits(data, "sequence")),
            embeddings=maybe_tensor(data["embeddings"]),
            mean_embedding=data["mean_embedding"],
            hidden_states=maybe_tensor(data["hidden_states"]),
            mean_hidden_state=maybe_tensor(data["mean_hidden_state"]),
            sae_outputs=sae_outputs,
        )
        return output

    @retry_decorator
    async def async_encode(
        self, input: ESMProtein
    ) -> ESMProteinTensor | ESMProteinError:
        tracks = {}
        tracks["sequence"] = input.sequence

        request = {"inputs": tracks, "model": self.model}

        try:
            data = await self._async_post(
                "encode", request, input.potential_sequence_of_concern
            )
        except ESMProteinError as e:
            return e

        return ESMProteinTensor(
            sequence=maybe_tensor(data["outputs"]["sequence"]),
            potential_sequence_of_concern=data["potential_sequence_of_concern"],
        )

    @retry_decorator
    def encode(self, input: ESMProtein) -> ESMProteinTensor | ESMProteinError:
        tracks = {}
        tracks["sequence"] = input.sequence

        request = {"inputs": tracks, "model": self.model}

        try:
            data = self._post("encode", request, input.potential_sequence_of_concern)
        except ESMProteinError as e:
            return e

        return ESMProteinTensor(
            sequence=maybe_tensor(data["outputs"]["sequence"]),
            potential_sequence_of_concern=data["potential_sequence_of_concern"],
        )

    @retry_decorator
    async def async_decode(
        self, input: ESMProteinTensor
    ) -> ESMProtein | ESMProteinError:
        _validate_protein_tensor_input(input)

        tokens = {}
        tokens["sequence"] = maybe_list(input.sequence)

        request = {"model": self.model, "inputs": tokens}

        try:
            data = await self._async_post(
                "decode", request, input.potential_sequence_of_concern
            )
        except ESMProteinError as e:
            return e

        return ESMProtein(sequence=data["outputs"]["sequence"])

    @retry_decorator
    def decode(self, input: ESMProteinTensor) -> ESMProtein | ESMProteinError:
        _validate_protein_tensor_input(input)

        tokens = {}
        tokens["sequence"] = maybe_list(input.sequence)

        request = {"model": self.model, "inputs": tokens}

        try:
            data = self._post("decode", request, input.potential_sequence_of_concern)
        except ESMProteinError as e:
            return e

        return ESMProtein(sequence=data["outputs"]["sequence"])

    @retry_decorator
    async def async_logits(
        self,
        input: ESMProteinTensor,
        config: LogitsConfig = LogitsConfig(),
        return_bytes: bool = True,
    ) -> LogitsOutput | ESMProteinError:
        request = self._process_logits_request(input, config, self.model)
        try:
            data = await self._async_post(
                "logits",
                request,
                input.potential_sequence_of_concern,
                return_bytes=return_bytes,
                headers={
                    "Accept": f"{MIMETYPE_ES_PICKLE};protocol={pickle.HIGHEST_PROTOCOL}, application/json"
                },
            )
        except ESMProteinError as e:
            return e

        return self._process_logits_response(data, return_bytes)

    @retry_decorator
    def logits(
        self,
        input: ESMProteinTensor,
        config: LogitsConfig = LogitsConfig(),
        return_bytes: bool = True,
    ) -> LogitsOutput | ESMProteinError:
        request = self._process_logits_request(input, config, self.model)
        try:
            data = self._post(
                "logits",
                request,
                input.potential_sequence_of_concern,
                return_bytes=return_bytes,
                headers={
                    "Accept": f"{MIMETYPE_ES_PICKLE};protocol={pickle.HIGHEST_PROTOCOL}, application/json"
                },
            )
        except ESMProteinError as e:
            return e

        return self._process_logits_response(data, return_bytes)

    @property
    def raw_model(self):
        raise NotImplementedError(
            f"Can not get underlying remote model {self.model} from a Forge/Biohub Platform client."
        )
