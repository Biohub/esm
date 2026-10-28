from __future__ import annotations

import asyncio
import base64
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence, TypeAlias, Union, cast

import attr
import torch

from esm.sdk.api import (
    ChainInput,
    ESM3InferenceClient,
    ESMCInferenceClient,
    ESMProtein,
    ESMProteinError,
    ESMProteinTensor,
    FoldAllAtomRequest,
    FoldingConfig,
    FoldMaxAccuracyConfig,
    FoldMaxAccuracyRequest,
    FoldRequest,
    ForwardAndSampleOutput,
    ForwardTrackData,
    GenerationConfig,
    InverseFoldingConfig,
    LogitsConfig,
    LogitsOutput,
    ProteinComplex,
    ProteinType,
    ResidueGeometry,
    ResidueMetrics,
    ResidueRef,
    ResolvedResidue,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.sdk.base_forge_client import (
    EndpointHandler,
    _BaseForgeBatchClient,
    _BaseForgeInferenceClient,
)
from esm.sdk.batch import BatchSubmission
from esm.sdk.retry import retry_decorator
from esm.sdk.validation import validate_fold_max_accuracy_input
from esm.utils.compression import compress_state_dict
from esm.utils.constants.api import ACCEPT_ES_PICKLE
from esm.utils.constants.models import (
    DEFAULT_ESMFOLD2_FAST_LM_MASK_PCT,
    ESMFOLD2,
    ESMFOLD2_CUTOFF_2025,
    ESMFOLD2_FAST,
    ESMFOLD2_MAX_MSA_SEQS,
)
from esm.utils.misc import deserialize_tensors, maybe_list, maybe_tensor, to_float32
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


def _resolve_lm_mask_pct(lm_mask_pct: float | None, model_name: str | None) -> float:
    f"""
    An explicit value is always honored. When unset (``None``),
    {ESMFOLD2_FAST} (or no model name) defaults to 0.1,
    {ESMFOLD2} defaults to 0.0.
    """
    if lm_mask_pct is not None:
        return lm_mask_pct
    is_fast_or_default = model_name is None or model_name == ESMFOLD2_FAST
    return DEFAULT_ESMFOLD2_FAST_LM_MASK_PCT if is_fast_or_default else 0.0


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


def _attrs_from_dict(cls, data: dict | None):
    """Build an attrs response type from the API's dict.

    Unknown keys are dropped, so a field the API adds does not break an older SDK.
    A required key that the API stops sending raises rather than reads as a
    plausible value.
    """
    if data is None:
        return None
    names = {field.name for field in attr.fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in names})


def _fold_target(
    sequence: str | None, chains: list[ChainInput] | None
) -> tuple[dict[str, Any], str]:
    """The request fields naming what to fold, and the '|'-joined sequence.

    The joined form is what an MSA lookup takes, so a chain-addressed request
    still needs it.
    """
    if (sequence is None) == (chains is None):
        raise ValueError("Supply exactly one of sequence or chains.")
    if chains is not None:
        if not chains:
            raise ValueError("chains is empty. Please provide a valid input.")
        return (
            {"chains": [attr.asdict(chain) for chain in chains]},
            "|".join(chain.sequence for chain in chains),
        )
    assert sequence is not None
    if len(sequence) == 0:
        raise ValueError("Input sequence length is 0. Please provide a valid input.")
    return {"sequence": sequence}, sequence


def _chain_plddt(data: dict[str, Any]) -> list[torch.Tensor] | None:
    """Per-chain pLDDT as tensors, with NaN where the model gave no confidence."""
    chain_plddt = data.get("chain_plddt", None)
    if chain_plddt is None:
        return None
    return [
        torch.tensor([float("nan") if value is None else value for value in chain])
        for chain in chain_plddt
    ]


def _residue_metrics(data: dict[str, Any]) -> ResidueMetrics | None:
    """The per-residue block of a fold reply, or None when none was asked for."""
    resolved = data.get("resolved_residues", None)
    if resolved is None:
        return None
    geometry = data.get("residue_geometry", None)
    return ResidueMetrics(
        resolved=[_attrs_from_dict(ResolvedResidue, residue) for residue in resolved],
        plddt=data.get("residue_plddt", None) or [],
        geometry=_attrs_from_dict(ResidueGeometry, geometry),
        min_distances_to_interface_angstrom=(
            data.get("min_distances_to_interface_angstrom", None) or []
        ),
        relative_sasa=data.get("residue_relative_sasa", None),
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
        request_timeout: int | None = 125,
        min_retry_wait: int = 1,
        max_retry_wait: int = 10,
        max_retry_attempts: int = 3,
    ):
        """
        Forge/Biohub Platform client for folding and inverse folding between sequence and structure spaces.

        Args:
            url: URL of the Forge/Biohub Platform server.
            model: Name of the model to be used for folding / inv folding.
            token: API token.
            request_timeout: Override the system default request timeout (125s), in seconds.
                Note the API has a 120s timeout
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
        sequence: str | None,
        msa: MSAInput,
        config: FoldingConfig,
        target_structure: ProteinComplex | None,
        model_name: str | None,
        chains: list[ChainInput] | None = None,
        residues_of_interest: list[ResidueRef] | None = None,
        interface_chain_id: str | None = None,
    ):
        if config.include_distogram:
            raise ValueError(
                "include_distogram is not supported for Forge/Biohub Platform right now"
            )

        target, _ = _fold_target(sequence, chains)
        request: dict[str, Any] = dict(target)

        if residues_of_interest is not None:
            # Omitting the field and sending an empty list are different requests to
            # the server: one wants no metrics, the other asks for metrics on nothing.
            if not residues_of_interest:
                raise ValueError(
                    "residues_of_interest is empty. Omit it to ask for no residue "
                    "metrics."
                )
            request["residues_of_interest"] = [
                attr.asdict(residue) for residue in residues_of_interest
            ]
        if interface_chain_id is not None:
            request["interface_chain_id"] = interface_chain_id

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
                raise ValueError(
                    f"MSA depth ({len(msa.sequences)}) exceeds the maximum of "
                    f"{ESMFOLD2_MAX_MSA_SEQS}. Truncate the alignment using "
                    "`select_sequences` before submitting."
                )
            request["msa"] = compress_state_dict(msa)
        else:
            error_msg = f"MSA must be None or MSA. Got {msa} instead."
            raise AttributeError(error_msg)

        request["include_distogram"] = config.include_distogram
        request["include_pae"] = config.include_pae
        request["include_pair_chains_iptm"] = config.include_pair_chains_iptm
        request["num_sampling_steps"] = config.num_sampling_steps
        request["num_loops"] = config.num_loops
        request["lm_dropout"] = config.lm_dropout
        request["lm_mask_pct"] = _resolve_lm_mask_pct(config.lm_mask_pct, model_name)
        request["msa_max_depth"] = config.msa_max_depth
        request["msa_column_mask_rate"] = config.msa_column_mask_rate
        request["target_structure"] = (
            target_structure.state_dict(json_serializable=True)
            if target_structure
            else None
        )
        request["include_embeddings"] = config.include_embeddings

        request["model"] = model_name

        return request

    @staticmethod
    def _process_fold_response(data: dict[str, Any]) -> ESMProtein:
        return ESMProtein(
            # The server echoes the request's raw separators, and ':' is also one, but
            # ESMProtein splits chains only on '|'.
            sequence=data["sequence"].replace(":", "|"),
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
            chains=data.get("chains", None),
            chain_plddt=_chain_plddt(data),
            residue_metrics=_residue_metrics(data),
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
        sequence: str | None = None,
        potential_sequence_of_concern: bool = False,
        model_name: str | None = None,
        msa: MSAInput = None,
        config: FoldingConfig = FoldingConfig(),
        target_structure: ProteinComplex | None = None,
        chains: list[ChainInput] | None = None,
        residues_of_interest: list[ResidueRef] | None = None,
        interface_chain_id: str | None = None,
        disable_cache: bool = False,
    ) -> ESMProtein | ESMProteinError:
        """Predict coordinates for a protein sequence.

        Args:
            sequence: Protein sequence to be folded; '|' separates chains.
            model_name: Override the client level model name if needed.
            msa: Optional multi-sequence alignment data that may help some models fold the sequence.
            config: Optional configuration for folding parameters.
            target_structure: Optional target structure to use for distogram conditioning.
            chains: The complex as named chains, the addressable form of `sequence`.
                Supply this or `sequence`, not both.
            residues_of_interest: Residues to measure, each addressed as
                {chain, position}. The result then carries `residue_metrics`.
            interface_chain_id: Chain to measure interface distances against. Every
                other chain is the far side. Defaults to the last chain.
            disable_cache: Optional flag to disable caching

        Deprecated:
            potential_sequence_of_concern: this parameter is largely deprecated
                and ignored by the folding endpoint.
        """
        del potential_sequence_of_concern
        _, folded_sequence = _fold_target(sequence, chains)

        request = self._process_fold_request(
            sequence,
            msa,
            config,
            target_structure,
            model_name if model_name is not None else self.model,
            chains,
            residues_of_interest,
            interface_chain_id,
        )

        try:
            data = await self._async_post(
                "fold",
                request,
                headers={"Accept": ACCEPT_ES_PICKLE},
                disable_cache=disable_cache,
            )
        except ESMProteinError as e:
            return e

        return self._process_fold_response(data)

    @retry_decorator
    def fold(
        self,
        sequence: str | None = None,
        potential_sequence_of_concern: bool = False,
        model_name: str | None = None,
        msa: MSAInput = None,
        config: FoldingConfig = FoldingConfig(),
        target_structure: ProteinComplex | None = None,
        chains: list[ChainInput] | None = None,
        residues_of_interest: list[ResidueRef] | None = None,
        interface_chain_id: str | None = None,
        disable_cache: bool = False,
    ) -> ESMProtein | ESMProteinError:
        """Predict coordinates for a protein sequence.

        Args:
            sequence: Protein sequence to be folded; '|' separates chains.
            model_name: Override the client level model name if needed.
            msa: Optional multi-sequence alignment data that may help some models fold the sequence.
            config: Optional configuration for folding parameters.
            target_structure: Optional target structure to use for distogram conditioning.
            chains: The complex as named chains, the addressable form of `sequence`.
                Supply this or `sequence`, not both.
            residues_of_interest: Residues to measure, each addressed as
                {chain, position}. The result then carries `residue_metrics`.
            interface_chain_id: Chain to measure interface distances against. Every
                other chain is the far side. Defaults to the last chain.
            disable_cache: Optional flag to disable caching

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
            chains,
            residues_of_interest,
            interface_chain_id,
        )

        try:
            data = self._post(
                "fold",
                request,
                headers={"Accept": ACCEPT_ES_PICKLE},
                disable_cache=disable_cache,
            )
        except ESMProteinError as e:
            return e

        return self._process_fold_response(data)

    @retry_decorator
    async def async_fold_all_atom(
        self,
        all_atom_input: StructurePredictionInput,
        config: FoldingConfig = FoldingConfig(),
        model_name: str | None = None,
        disable_cache: bool = False,
    ) -> MolecularComplexResult | list[MolecularComplexResult] | ESMProteinError:
        """Fold a molecular complex containing proteins, nucleic acids, and/or ligands.

        Args:
            all_atom_input: StructurePredictionInput containing sequences for different molecule types
            config: Optional configuration for folding parameters.
            model_name: Override the client level model name if needed
            disable_cache: Optional flag to disable caching
        """
        request = self._process_fold_all_atom_request(
            all_atom_input, config, model_name if model_name is not None else self.model
        )

        try:
            data = await self._async_post(
                "fold_all_atom", request, disable_cache=disable_cache
            )
        except ESMProteinError as e:
            return e

        return self._process_fold_all_atom_response(data)

    @retry_decorator
    def fold_all_atom(
        self,
        all_atom_input: StructurePredictionInput,
        model_name: str | None = None,
        config: FoldingConfig = FoldingConfig(),
        disable_cache: bool = False,
    ) -> MolecularComplexResult | list[MolecularComplexResult] | ESMProteinError:
        """Predict coordinates for a molecular complex containing proteins, dna, rna, and/or ligands.

        Args:
            all_atom_input: StructurePredictionInput containing sequences for different molecule types
            model_name: Override the client level model name if needed
            config: Optional configuration for folding parameters.
            disable_cache: Optional flag to disable caching
        """
        request = self._process_fold_all_atom_request(
            all_atom_input, config, model_name if model_name is not None else self.model
        )

        try:
            data = self._post("fold_all_atom", request, disable_cache=disable_cache)
        except ESMProteinError as e:
            return e

        return self._process_fold_all_atom_response(data)

    @staticmethod
    def _process_fold_all_atom_request(
        all_atom_input: StructurePredictionInput,
        config: FoldingConfig,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        if config.include_distogram:
            raise ValueError(
                "include_distogram is not supported for Forge/Biohub Platform right now"
            )

        if len(all_atom_input.sequences) == 0:
            raise ValueError(
                "Input sequence length is 0. Please provide a valid input."
            )

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
                raise ValueError(
                    f"MSA depth ({len(seq.msa.sequences)}) for chain '{chain_id}' "
                    f"exceeds the maximum of {ESMFOLD2_MAX_MSA_SEQS}. Truncate the "
                    "alignment using `select_sequences` before submitting."
                )

        request: dict[str, Any] = {
            "all_atom_input": serialize_structure_prediction_input(all_atom_input),
            "model": model_name,
        }

        request["include_distogram"] = config.include_distogram
        request["include_pae"] = config.include_pae
        request["num_sampling_steps"] = config.num_sampling_steps
        request["num_loops"] = config.num_loops
        request["lm_dropout"] = config.lm_dropout
        request["lm_mask_pct"] = _resolve_lm_mask_pct(config.lm_mask_pct, model_name)
        request["msa_max_depth"] = config.msa_max_depth
        request["msa_column_mask_rate"] = config.msa_column_mask_rate
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
        potential_sequence_of_concern: bool = False,
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
                "inverse_fold",
                request,
                potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
            )
        except ESMProteinError as e:
            return e

        return ESMProtein(sequence=data["sequence"])

    @retry_decorator
    def inverse_fold(
        self,
        coordinates: torch.Tensor,
        config: InverseFoldingConfig,
        potential_sequence_of_concern: bool = False,
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
            data = self._post(
                "inverse_fold",
                request,
                potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
            )
        except ESMProteinError as e:
            return e

        return ESMProtein(sequence=data["sequence"])


class ESM3ForgeInferenceClient(ESM3InferenceClient, _BaseForgeInferenceClient):
    def __init__(
        self,
        model: str,
        url: str = "https://biohub.ai",
        token: str = "",
        request_timeout: int | None = 125,
        min_retry_wait: int = 1,
        max_retry_wait: int = 10,
        max_retry_attempts: int = 3,
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
            "only_compute_backbone_rmsd": config.only_compute_backbone_rmsd,
            "include_pae": config.include_pae,
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
            pae=maybe_tensor(data["outputs"]["pae"]),
            crmsd=maybe_tensor(data["outputs"]["crmsd"]),
            globularity=maybe_tensor(data["outputs"]["globularity"]),
            interface_ptm=maybe_tensor(data["outputs"]["interface_ptm"]),
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
                "generate",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
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
                "generate_tensor",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
            )
        except ESMProteinError as e:
            return e

        return self._process_generate_protein_tensor_response(data)

    def __generate_protein(
        self, input: ESMProtein, config: GenerationConfig
    ) -> ESMProtein | ESMProteinError:
        request = self._process_generate_protein_request(input, config, self.model)
        try:
            data = self._post(
                "generate",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
            )
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
                "generate_tensor",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
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
                headers={"Accept": ACCEPT_ES_PICKLE},
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
                headers={"Accept": ACCEPT_ES_PICKLE},
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
                "encode",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
            )
        except ESMProteinError as e:
            return e

        return self._process_encode_response(data)

    @retry_decorator
    def encode(self, input: ESMProtein) -> ESMProteinTensor | ESMProteinError:
        request = self._process_encode_request(input, self.model)

        try:
            data = self._post(
                "encode",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
            )
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
                "decode",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
            )
        except ESMProteinError as e:
            return e

        return self._process_decode_response(data)

    @retry_decorator
    def decode(self, input: ESMProteinTensor) -> ESMProtein | ESMProteinError:
        request = self._process_decode_request(input, self.model)

        try:
            data = self._post(
                "decode",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
            )
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
                headers={"Accept": ACCEPT_ES_PICKLE},
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
                headers={"Accept": ACCEPT_ES_PICKLE},
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
        request_timeout: int | None = 125,
        min_retry_wait: int = 1,
        max_retry_wait: int = 10,
        max_retry_attempts: int = 3,
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
            logits=ForwardTrackData(
                sequence=to_float32(_maybe_logits(data, "sequence"))
            ),
            embeddings=to_float32(maybe_tensor(data["embeddings"])),
            mean_embedding=to_float32(data["mean_embedding"]),
            hidden_states=to_float32(maybe_tensor(data["hidden_states"])),
            mean_hidden_state=to_float32(maybe_tensor(data["mean_hidden_state"])),
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
                "encode",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
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
            data = self._post(
                "encode",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
            )
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
                "decode",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
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
            data = self._post(
                "decode",
                request,
                input.potential_sequence_of_concern,
                headers={"Accept": ACCEPT_ES_PICKLE},
            )
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
                headers={"Accept": ACCEPT_ES_PICKLE},
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
                headers={"Accept": ACCEPT_ES_PICKLE},
            )
        except ESMProteinError as e:
            return e

        return self._process_logits_response(data, return_bytes)

    @property
    def raw_model(self):
        raise NotImplementedError(
            f"Can not get underlying remote model {self.model} from a Forge/Biohub Platform client."
        )


class FoldMaxAccuracyHandler(
    EndpointHandler[FoldMaxAccuracyRequest, MolecularComplexResult]
):
    """Fold a molecular complex at the highest-accuracy settings."""

    request_type = FoldMaxAccuracyRequest
    poll_interval = 10
    poll_max_interval = 60
    default_timeout = 2 * 60 * 60  # 2 hours

    def __init__(self, batch_client: _BaseForgeBatchClient):
        super().__init__(batch_client)

    @property
    def endpoint_name(self) -> str:
        return "fold_max_accuracy"

    def _prepare_request(
        self, request: FoldMaxAccuracyRequest, *, model_name: str | None = None
    ) -> dict[str, Any]:
        if model_name is not None:
            raise TypeError(
                "fold_max_accuracy selects its own model; model_name= is not supported"
            )
        config = (
            request.config if request.config is not None else FoldMaxAccuracyConfig()
        )
        validate_fold_max_accuracy_input(request.all_atom_input)
        return {
            "all_atom_input": serialize_structure_prediction_input(
                request.all_atom_input
            ),
            "model": ESMFOLD2_CUTOFF_2025,
            **attr.asdict(config),
        }

    def _process_response(self, response: dict) -> MolecularComplexResult:
        result = self._batch_client._read_response(response)
        return SequenceStructureForgeInferenceClient._process_fold_all_atom_response(
            result
        )

    async def _async_process_response(self, response: dict) -> MolecularComplexResult:
        result = await self._batch_client._async_read_response(response)
        return SequenceStructureForgeInferenceClient._process_fold_all_atom_response(
            result
        )


def _require_model(endpoint: str, model: str | None) -> str:
    # A null model fails the submit route's body parse with an unhelpful message.
    if model is None or not model.strip():
        raise ValueError(
            f"{endpoint} requires a non-empty model. Pass model= when constructing "
            "ForgeBatchClient, or model_name= to submit()/run()."
        )
    return model


class FoldHandler(EndpointHandler[FoldRequest, ESMProtein]):
    request_type = FoldRequest

    def __init__(self, batch_client: _BaseForgeBatchClient, model: str | None = None):
        super().__init__(batch_client)
        self.model = model

    @property
    def endpoint_name(self) -> str:
        return "fold"

    def _prepare_request(
        self, request: FoldRequest, *, model_name: str | None = None
    ) -> dict[str, Any]:
        model = model_name if model_name is not None else self.model
        return SequenceStructureForgeInferenceClient._process_fold_request(
            request.sequence,
            request.msa,
            request.config if request.config is not None else FoldingConfig(),
            request.target_structure,
            _require_model(self.endpoint_name, model),
            request.chains,
            request.residues_of_interest,
            request.interface_chain_id,
        )

    def _process_response(self, response: dict) -> ESMProtein:
        result = self._batch_client._read_response(response)
        return SequenceStructureForgeInferenceClient._process_fold_response(result)

    async def _async_process_response(self, response: dict) -> ESMProtein:
        result = await self._batch_client._async_read_response(response)
        return SequenceStructureForgeInferenceClient._process_fold_response(result)


class FoldAllAtomHandler(EndpointHandler[FoldAllAtomRequest, MolecularComplexResult]):
    request_type = FoldAllAtomRequest

    def __init__(self, batch_client: _BaseForgeBatchClient, model: str | None = None):
        super().__init__(batch_client)
        self.model = model

    @property
    def endpoint_name(self) -> str:
        return "fold_all_atom"

    def _prepare_request(
        self, request: FoldAllAtomRequest, *, model_name: str | None = None
    ) -> dict[str, Any]:
        model = model_name if model_name is not None else self.model
        return SequenceStructureForgeInferenceClient._process_fold_all_atom_request(
            request.all_atom_input,
            request.config if request.config is not None else FoldingConfig(),
            _require_model(self.endpoint_name, model),
        )

    def _process_response(self, response: dict) -> MolecularComplexResult:
        result = self._batch_client._read_response(response)
        # Use the same logic as _process_fold_all_atom_response
        return SequenceStructureForgeInferenceClient._process_fold_all_atom_response(
            result
        )

    async def _async_process_response(self, response: dict) -> MolecularComplexResult:
        result = await self._batch_client._async_read_response(response)
        # Use the same logic as _process_fold_all_atom_response
        return SequenceStructureForgeInferenceClient._process_fold_all_atom_response(
            result
        )


class ForgeBatchClient:
    """Submit one or many tasks and recover them through saved job identifiers.

    Endpoint handlers accept SDK inputs and return decoded Python objects.
    This client provides shared status, waiting, listing, and cancellation
    operations. Each method has an ``async_`` equivalent.
    """

    def __init__(
        self,
        url: str = "https://biohub.ai",
        token: str = "",
        request_timeout: int | None = 15,
        model: str | None = None,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
        min_retry_wait: int = 2,
        max_retry_wait: int = 2,
        max_retry_attempts: int = 3,
        transfer_timeout: int | None = 60,
    ):
        self._batch_client = _BaseForgeBatchClient(
            url=url,
            token=token,
            request_timeout=request_timeout,
            min_retry_wait=min_retry_wait,
            max_retry_wait=max_retry_wait,
            max_retry_attempts=max_retry_attempts,
            poll_interval=poll_interval,
            poll_max_interval=poll_max_interval,
            transfer_timeout=transfer_timeout,
        )
        self.model = model

        self._fold_max_accuracy: FoldMaxAccuracyHandler | None = None
        self._fold: FoldHandler | None = None
        self._fold_all_atom: FoldAllAtomHandler | None = None

    def close(self) -> None:
        self._batch_client.close()

    async def aclose(self) -> None:
        await self._batch_client.aclose()
        self._batch_client.close()

    def __enter__(self) -> ForgeBatchClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    async def __aenter__(self) -> ForgeBatchClient:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    def get_status(self, task_id: str) -> dict[str, Any]:
        """Read task state, result location, errors, billing, and available artifacts."""
        return self._batch_client.get_status(task_id)

    async def async_get_status(self, task_id: str) -> dict[str, Any]:
        return await self._batch_client.async_get_status(task_id)

    def wait(
        self,
        submission: BatchSubmission,
        timeout: int = 1800,
        *,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Wait for a batch, returning its aggregate summary.

        A polling timeout raises ``BatchTimeoutError`` with the submission and leaves
        work running. The budget excludes submission and result downloads.
        """
        return self._batch_client.wait(
            submission, timeout, poll_interval, poll_max_interval
        )

    async def async_wait(
        self,
        submission: BatchSubmission,
        timeout: int = 1800,
        *,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        return await self._batch_client.async_wait(
            submission, timeout, poll_interval, poll_max_interval
        )

    def cancel(
        self, task_id: str | None = None, *, batch_id: str | None = None
    ) -> dict[str, Any]:
        """Cancel one task or queued work in a batch; running tasks may still finish."""
        return self._batch_client.cancel(task_id, batch_id=batch_id)

    async def async_cancel(
        self, task_id: str | None = None, *, batch_id: str | None = None
    ) -> dict[str, Any]:
        return await self._batch_client.async_cancel(task_id, batch_id=batch_id)

    def list_batch(
        self,
        batch_id: str,
        *,
        status: list[str] | None = None,
        include_responses: bool = False,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Read one batch's progress and, optionally, a page of its tasks.

        Parameters
        ----------
        batch_id
            Saved batch ID returned by submission.
        status
            Task states to include, such as ``["done", "failed"]``. Only filters
            the item page; aggregate counts always describe the whole batch.
        include_responses
            Include task entries and pagination information. S3 results are
            marked ``response_available``; this call does not download them.
        limit
            Page size. Omit to use the server default.
        cursor
            Opaque cursor from the previous page. Use the same filters when
            requesting subsequent pages.

        Returns
        -------
        dict[str, Any]
            ``batch_id``, ``endpoint``, ``status``, ``counts``, and ``total``.
            With ``include_responses=True``, also ``items`` and ``cursor``;
            a null cursor marks the last page. Item entries contain task IDs,
            states, and available inline responses or errors.

        Notes
        -----
        HTTP errors are raised as ``ESMProteinError``.
        """
        return self._batch_client.list_batch(
            batch_id,
            status=status,
            include_responses=include_responses,
            limit=limit,
            cursor=cursor,
        )

    async def async_list_batch(
        self,
        batch_id: str,
        *,
        status: list[str] | None = None,
        include_responses: bool = False,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Asynchronous equivalent of :meth:`list_batch`, with the same options."""
        return await self._batch_client.async_list_batch(
            batch_id,
            status=status,
            include_responses=include_responses,
            limit=limit,
            cursor=cursor,
        )

    @property
    def fold_max_accuracy(self) -> FoldMaxAccuracyHandler:
        if self._fold_max_accuracy is None:
            self._fold_max_accuracy = FoldMaxAccuracyHandler(self._batch_client)
        return self._fold_max_accuracy

    @property
    def fold(self) -> FoldHandler:
        if self._fold is None:
            self._fold = FoldHandler(self._batch_client, self.model)
        return self._fold

    @property
    def fold_all_atom(self) -> FoldAllAtomHandler:
        if self._fold_all_atom is None:
            self._fold_all_atom = FoldAllAtomHandler(self._batch_client, self.model)
        return self._fold_all_atom
