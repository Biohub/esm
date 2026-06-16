from __future__ import annotations

import warnings
from abc import ABC
from copy import deepcopy
from typing import Sequence

import attr
import torch
from attr import asdict, define

import esm.utils.constants.api as C
from esm.tokenization import TokenizerCollectionProtocol, get_esm3_model_tokenizers
from esm.utils import encoding
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.misc import get_chainbreak_boundaries_from_sequence
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import SINGLE_LETTER_CHAIN_IDS, ProteinComplex
from esm.utils.types import FunctionAnnotation, PathOrBuffer


class ProteinType(ABC): ...


## Basic Types
@define
class ESMProtein(ProteinType):
    # Tracks
    sequence: str | None = None
    secondary_structure: str | None = None
    sasa: list[float | None] | None = None
    function_annotations: list[FunctionAnnotation] | None = None
    coordinates: torch.Tensor | None = None

    # Metrics
    plddt: torch.Tensor | None = None
    ptm: torch.Tensor | None = None
    pae: torch.Tensor | None = None

    crmsd: torch.Tensor | None = None
    globularity: torch.Tensor | None = None
    interface_annotations: list[str] | None = None
    interface_ptm: torch.Tensor | None = None
    pair_chains_iptm: torch.Tensor | None = None
    output_embedding_sequence: torch.Tensor | None = None
    output_embedding_pair_pooled: torch.Tensor | None = None
    residue_index: torch.Tensor | None = None
    entity_id: torch.Tensor | None = None

    # When calling EvolutionaryScale API, use this flag to disclose any
    # sequences that may potentially have concerns.
    # Such sequences may not go through standard safety filter for approved users.
    # Reach out if interested in using this.
    potential_sequence_of_concern: bool = False

    def __len__(self):
        if self.sequence is not None:
            return len(self.sequence)
        elif self.secondary_structure is not None:
            return len(self.secondary_structure)
        elif self.sasa is not None:
            return len(self.sasa)
        elif self.coordinates is not None:
            return self.coordinates.size(0)
        else:
            raise ValueError("No track to determine length from.")

    @classmethod
    def from_pdb(
        cls,
        path: PathOrBuffer,
        chain_id: str = "all",
        id: str | None = None,
        is_predicted: bool = False,
    ) -> ESMProtein:
        """Return an ESMProtein object from a pdb file.

        Args:
            path (str | Path | io.TextIO): Path or buffer to read pdb file from. Should be uncompressed.
            chain_id (str, optional): Select a chain corresponding to (author) chain id. "all" uses all chains,
            "detect" uses the first detected chain
            id (str, optional): String identifier to assign to structure. Will attempt to infer otherwise.
            is_predicted (bool): If True, reads b factor as the confidence readout. Default: False.
        """
        if chain_id == "all":
            protein_complex = ProteinComplex.from_pdb(
                path=path, id=id, is_predicted=is_predicted
            )
            return cls.from_protein_complex(protein_complex)
        else:
            protein_chain = ProteinChain.from_pdb(
                path=path, chain_id=chain_id, id=id, is_predicted=is_predicted
            )
            return cls.from_protein_chain(protein_chain)

    @classmethod
    def from_protein_chain(
        cls, protein_chain: ProteinChain, with_annotations: bool = False
    ) -> ESMProtein:
        if with_annotations:
            return ESMProtein(
                sequence=protein_chain.sequence,
                sasa=protein_chain.sasa().tolist(),
                function_annotations=None,
                coordinates=torch.tensor(protein_chain.atom37_positions),
                plddt=torch.tensor(protein_chain.confidence),
            )
        else:
            return ESMProtein(
                sequence=protein_chain.sequence,
                secondary_structure=None,
                sasa=None,
                function_annotations=None,
                coordinates=torch.tensor(protein_chain.atom37_positions),
                plddt=torch.tensor(protein_chain.confidence),
            )

    @classmethod
    def from_protein_complex(
        cls, protein_complex: ProteinComplex, with_annotations: bool = False
    ) -> ESMProtein:
        if with_annotations:
            raise NotImplementedError(
                "Annotations are not supported for ProteinComplex yet."
            )

        return ESMProtein(
            sequence=protein_complex.sequence,
            secondary_structure=None,
            sasa=None,
            function_annotations=None,
            coordinates=torch.tensor(
                protein_complex.atom37_positions, dtype=torch.float32
            ),
            plddt=torch.tensor(protein_complex.confidence),
        )

    def to_pdb(self, pdb_path: PathOrBuffer) -> None:
        # Note: Will work for single chains as well and produce same pdb file
        protein_complex = self.to_protein_complex().infer_oxygen()
        protein_complex.to_pdb(pdb_path)

    def to_pdb_string(self) -> str:
        # Note: This was modified to match .to_pdb() behavior. We can revisit this at some point
        protein_chain = self.to_protein_complex().infer_oxygen()
        return protein_chain.to_pdb_string()

    def to_protein_chain(self) -> ProteinChain:
        if self.coordinates is None:
            raise ValueError("Coordinates are required to convert to a ProteinChain.")
        protein_chain = ProteinChain.from_atom37(
            atom37_positions=self.coordinates.to("cpu").numpy(),
            id=None,
            sequence=None if self.sequence is None else self.sequence.replace("_", "X"),
            chain_id=None,
            entity_id=None,
            residue_index=None,
            insertion_code=None,
            confidence=None
            if self.plddt is None
            else self.plddt.detach().cpu().numpy(),
        )
        return protein_chain

    def to_protein_complex(
        self, copy_annotations_from_ground_truth: ProteinComplex | None = None
    ) -> ProteinComplex:
        assert (
            self.sequence is not None
        ), "ESMProtein must have a sequence to convert to ProteinComplex"
        assert (
            self.coordinates is not None
        ), "ESMProtein must have coordinates to convert to ProteinComplex"
        coords = self.coordinates.to("cpu").numpy()

        chain_boundaries = get_chainbreak_boundaries_from_sequence(self.sequence)
        if copy_annotations_from_ground_truth is not None:
            gt_chains = list(copy_annotations_from_ground_truth.chain_iter())
        else:
            gt_chains = None

        # Expand pLDDT to match sequence length if needed, inserting NaN at chain breaks
        # This handles the case where the server doesn't include chain breaks in pLDDT
        # We should fix this in the server side.
        if self.plddt is not None and len(self.plddt) != len(self.sequence):
            # Only expand if there's a mismatch (likely due to chain breaks)
            if "|" in self.sequence:
                # Create expanded pLDDT with NaN at chain break positions
                expanded_plddt = torch.full((len(self.sequence),), float("nan"))
                plddt_idx = 0
                for i, aa in enumerate(self.sequence):
                    if aa != "|":
                        if plddt_idx < len(self.plddt):
                            expanded_plddt[i] = self.plddt[plddt_idx]
                        plddt_idx += 1
                plddt = expanded_plddt
            else:
                # Mismatch but no chain breaks - shouldn't happen but preserve original
                plddt = self.plddt
        else:
            plddt = self.plddt

        pred_chains = []
        for i, (start, end) in enumerate(chain_boundaries):
            if i >= len(SINGLE_LETTER_CHAIN_IDS):
                raise ValueError(
                    f"Too many chains to convert to ProteinComplex. The maximum number of chains is {len(SINGLE_LETTER_CHAIN_IDS)}"
                )

            pred_chain = ProteinChain.from_atom37(
                atom37_positions=coords[start:end],
                sequence=self.sequence[start:end],
                chain_id=gt_chains[i].chain_id
                if gt_chains is not None
                else SINGLE_LETTER_CHAIN_IDS[i],
                residue_index=self.residue_index[start:end]
                if self.residue_index is not None
                else None,
                entity_id=gt_chains[i].entity_id if gt_chains is not None else None,
                confidence=plddt[start:end] if plddt is not None else None,
            )
            pred_chains.append(pred_chain)
        return ProteinComplex.from_chains(pred_chains)

    def copy(self) -> "ESMProtein":
        """Create a deep copy of the ESMProtein instance."""
        return deepcopy(self)


@define
class ESMProteinTensor(ProteinType):
    sequence: torch.Tensor | None = None
    structure: torch.Tensor | None = None
    secondary_structure: torch.Tensor | None = None
    sasa: torch.Tensor | None = None
    function: torch.Tensor | None = None
    residue_annotations: torch.Tensor | None = None
    coordinates: torch.Tensor | None = None

    # When calling EvolutionaryScale API, use this flag to disclose any
    # sequences that may potentially have concerns.
    # Such sequences may not go through standard safety filter for approved users.
    # Reach out if interested in using this.
    potential_sequence_of_concern: bool = False

    def _detect_attribute(self, func, msg):
        mapped = {
            k: func(k, v)
            for k, v in asdict(self).items()
            if isinstance(v, torch.Tensor)
        }
        s = set(mapped.values())
        if len(s) <= 0:
            return None
        if len(s) != 1:
            raise ValueError(f"Either no tracks or inconsistent {msg}: {mapped}")
        return next(iter(s))

    def __len__(self) -> int:
        l = self._detect_attribute(lambda _, x: x.size(0), "length")
        return l if l is not None else 0

    @property
    def device(self) -> str | torch.device:
        d = self._detect_attribute(lambda _, x: x.device, "device")
        assert d is not None
        return d

    def to(self, device_or_dtype: str | torch.device | torch.dtype) -> ESMProteinTensor:
        def _to(name):
            v = getattr(self, name)
            if v is not None and isinstance(v, torch.Tensor):
                setattr(self, name, v.to(device_or_dtype))

        for n in attr.fields(ESMProteinTensor):
            _to(n.name)

        return self

    @classmethod
    def empty(
        cls,
        length: int,
        tokenizers: TokenizerCollectionProtocol | None = None,
        device: torch.device | str = "cpu",
    ) -> ESMProteinTensor:
        if tokenizers is None:
            tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)

        return ESMProteinTensor(
            sequence=encoding.get_default_sequence_tokens(
                length, tokenizers.sequence
            ).to(device),
            structure=encoding.get_default_structure_tokens(
                length, tokenizers.structure
            ).to(device),
            secondary_structure=encoding.get_default_secondary_structure_tokens(
                length, tokenizers.secondary_structure
            ).to(device),
            sasa=encoding.get_default_sasa_tokens(length, tokenizers.sasa).to(device),
            function=encoding.get_default_function_tokens(
                length, tokenizers.function
            ).to(device),
            residue_annotations=encoding.get_default_residue_annotation_tokens(
                length, tokenizers.residue_annotations
            ).to(device),
        )

    def copy(self) -> ESMProteinTensor:
        """Create a deep copy of the ESMProteinTensor instance."""
        return deepcopy(self)


@define
class ESMProteinError(Exception, ProteinType):
    error_code: int  # Error code follows HTTP convention, i.e., 404 NotFoundError, 500 InternalError.
    error_msg: str


## High Level Endpoint Types
@define
class GenerationConfig:
    """
    track (str): Track to generate: sequence, structure, secondary_structure, sasa,
        or function.
    invalid_ids (Sequence[int]): Token indices that should not be sampled.
    schedule (str): Unmasking schedule for generation. Controls the number of tokens
        to unmask during each round of iterative generation.
    strategy (str): Unmasking strategy to use. Controls which tokens to unmask
        during each round of iterative generation. 'random' will unmask a correct
        number of tokens randomly. 'entropy' will unmask the tokens with the lowest
        logit entropy first. Default was random. Updated on 02/14/2025.
    num_steps (int): Number of steps for generation. There is diminishing return for
        decoding steps more than 20. Note that this needs to be less than or equal
        to the sequence length. Default was 8. Updated on 02/14/2025.
    temperature (float): Temperature for sampling. Default was 1.0. Updated on
        02/14/2025.
    temperature_annealing (bool): Whether temperature should be annealed during
        generation. Default was False. Updated on 02/14/2025.
    top_p (float): Top-p sampling.
    condition_on_coordinates_only (bool): Use coordinates instead of structure
        tokens as generation conditioning.
    only_compute_backbone_rmsd (bool): Only compute the RMSD of the backbone atoms.
        Affects the returned crmsd.
    """

    track: str = ""
    invalid_ids: Sequence[int] = []
    schedule: str = attr.field(
        validator=attr.validators.in_(["cosine", "linear"]), default="cosine"
    )
    strategy: str = attr.field(
        validator=attr.validators.in_(["random", "entropy"]), default="random"
    )
    num_steps: int = 20
    temperature: float = 1.0
    temperature_annealing: bool = True
    top_p: float = 1.0
    condition_on_coordinates_only: bool = True
    only_compute_backbone_rmsd: bool = False

    def use_entropy_based_unmasking_strategy(self):
        """Use entropy based unmasking strategy during generation."""
        self.schedule = "cosine"
        self.strategy = "entropy"
        self.temperature_annealing = False

    def use_generative_unmasking_strategy(self):
        """Use an unmasking strategy that produces more variety of generations."""
        self.schedule = "cosine"
        self.strategy = "random"
        self.temperature_annealing = True


@define
class InverseFoldingConfig:
    """
    invalid_ids (Sequence[int]): Token indices that should not be sampled.
    temperature (float): Temperature for sampling. For inverse folding models, we
        recommend getting diverse predictions by changing the seed and not by
        increasing the temperature.
    """

    invalid_ids: Sequence[int] = []
    temperature: float = 0.1


@define
class FoldingConfig:
    """
    include_distogram (bool): (ESMFold2) Whether to include distogram predictions in
        the response.
    include_pae (bool): (ESMFold2) Whether to include Predicted Aligned Error (PAE)
        matrix in the response.
    include_pair_chains_iptm (bool): (ESMFold2) Whether to include pair-chain IPTM
        predictions in the response.
    num_sampling_steps (int): (ESMFold2) Diffusion ODE solver steps. Lower for
        speed, higher for quality.
    num_loops (int): (ESMFold2) Number of trunk loops for iterative refinement.
    lm_dropout (float): (ESMFold2) Dropout probability on LM pair embeddings. When >
        0, dropout is applied.
    lm_mask_pct (float | None): (ESMFold2) Fraction of sequence residues randomly
        masked before the PLM backbone. If not provided, defaults to 0.1 for
        ESMFOLD2_FAST and 0.0 for ESMFOLD2
    msa_max_depth (int | None): (ESMFold2) Number of MSA rows randomly subsampled
        each loop. Set to null to disable (sets msa_subsample_at_inference to
        False).
    msa_column_mask_rate (float): (ESMFold2) Fraction of MSA columns randomly masked
        in non-query rows for inference-time diversity.
    include_embeddings (bool): (ESMFold2) Whether to include sequence and pair
        embeddings in the response.
    """

    include_distogram: bool = False
    include_pae: bool = False
    include_pair_chains_iptm: bool = False
    num_sampling_steps: int = 100
    num_loops: int = 20
    lm_dropout: float = 0.3
    lm_mask_pct: float | None = None
    msa_max_depth: int | None = 1024
    msa_column_mask_rate: float = 0.1
    include_embeddings: bool = False


## Low Level Endpoint Types
@define
class SamplingTrackConfig:
    """
    temperature (float): Temperature for sampling.
    top_p (float): Sample from logits within the top-p probability.
    only_sample_masked_tokens (bool): Only sample for masked tokens.
    invalid_ids (Sequence[int]): Token indices that should not be sampled.
    topk_logprobs (int): Number of top ranking prediction and logprobs to return.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    only_sample_masked_tokens: bool = True
    invalid_ids: Sequence[int] = []
    topk_logprobs: int = 0


@define
class SamplingConfig:
    """
    sequence (SamplingTrackConfig | None): Sampling configuration for the sequence
        track.
    structure (SamplingTrackConfig | None): Sampling configuration for the structure
        track.
    secondary_structure (SamplingTrackConfig | None): Sampling configuration for the
        secondary structure track.
    sasa (SamplingTrackConfig | None): Sampling configuration for the SASA track.
    function (SamplingTrackConfig | None): Sampling configuration for the function
        annotation track.
    return_per_residue_embeddings (bool): Whether to return per-residue embeddings.
    return_mean_embedding (bool): Whether to return the embedding mean-pooled over
        the sequence length.
    """

    sequence: SamplingTrackConfig | None = attr.field(
        default=None, metadata={"max_topk": C.MAX_TOPK_SEQUENCE}
    )
    structure: SamplingTrackConfig | None = attr.field(
        default=None, metadata={"max_topk": C.MAX_TOPK_STRUCTURE}
    )
    secondary_structure: SamplingTrackConfig | None = attr.field(
        default=None, metadata={"max_topk": C.MAX_TOPK_SECONDARY_STRUCTURE}
    )
    sasa: SamplingTrackConfig | None = attr.field(
        default=None, metadata={"max_topk": C.MAX_TOPK_SASA}
    )
    function: SamplingTrackConfig | None = attr.field(
        default=None, metadata={"max_topk": C.MAX_TOPK_FUNCTION}
    )

    return_per_residue_embeddings: bool = False
    return_mean_embedding: bool = False


@define
class ForwardTrackData:
    """
    sequence (torch.Tensor | None): Sequence track logits.
    structure (torch.Tensor | None): Structure track logits.
    secondary_structure (torch.Tensor | None): Secondary structure track logits.
    sasa (torch.Tensor | None): Solvent accessible surface area (SASA) track logits.
    function (torch.Tensor | None): Function annotations logits.
    """

    sequence: torch.Tensor | None = None
    structure: torch.Tensor | None = None
    secondary_structure: torch.Tensor | None = None
    sasa: torch.Tensor | None = None
    function: torch.Tensor | None = None


@define
class LogitsConfig:
    """
    sequence (bool): Return sequence logits.
    structure (bool): Return structure logits.
    secondary_structure (bool): Return secondary structure logits.
    sasa (bool): Return sasa logits.
    function (bool): Return function logits.
    residue_annotations (bool): Return residue annotations logits.
    return_embeddings (bool): Whether embeddings should be returned.
    return_hidden_states (bool): Whether to return per-residue hidden states. With
        ith_hidden_layer=-1, returns all layers as a tensor of shape [n_layers + 1,
        B, L, D]. With ith_hidden_layer!= -1, returns the selected layer as a tensor
        of shape [1, B, L, D].
    return_mean_embedding (bool): Whether mean embeddings should be returned.
    return_mean_hidden_states (bool): Whether hidden states mean-pooled along the
        sequence length (L) dimension should be returned. Returns a tensor of shape
        [B, n_layers + 1, D].
    ith_hidden_layer (int): Valid values for ith_hidden_layer are 0 to
        max_ith_hidden_layer (inclusive), where index 0 is the embedding layer. -1
        returns all layers, but is not supported for ESMC 6B or any ESM3 model. Here
        is the max_ith_hidden_layer for each ESMC and ESM3 model (except ESM3
        Large).
        | Model Name                    | max_ith_hidden_layer           |
        |-------------------------------|--------------------------------|
        | esmc-300-2024-12              | 30                             |
        | esmc-600-2024-12              | 36                             |
        | esmc-6b-2024-12               | 80                             |
        | esm3-small-2024-03            | 48                             |
        | esm3-small-2024-08            | 48                             |
        | esm3-medium-2024-03           | 96                             |
        | esm3-medium-2024-08           | 96                             |
    sae_config (SAEConfig | None): SAE config. Only applies to ESMC models.
    """

    # Logits.
    sequence: bool = False

    # Note that getting logits for tracks other than sequence
    # are not supported by Forge/Biohub Platform today, due to their impractical
    # data sizes.
    # These are of course supported when running local OSS models.
    structure: bool = False
    secondary_structure: bool = False
    sasa: bool = False
    function: bool = False
    residue_annotations: bool = False

    # Embeddings.
    return_embeddings: bool = False
    return_hidden_states: bool = False
    return_mean_embedding: bool = False
    return_mean_hidden_states: bool = False
    ith_hidden_layer: int = -1

    sae_config: SAEConfig | None = None


@define
class SAEConfig:
    """
    models (list[str]): List of SAE models with specific layer and codebook size.
    normalize_features (bool): Normalize computed features before return. Default to
        True.
    model (str | None): Deprecated, use 'models' instead. SAE model with specific
        layer and codebook size.
    """

    models: list[str] = attr.Factory(list)
    normalize_features: bool = True
    model: str | None = None

    def __attrs_post_init__(self):
        if self.model is not None:
            if self.models:
                raise ValueError(
                    "Cannot specify both 'model' and 'models' in SAEConfig. "
                    "Use 'models' only."
                )
            warnings.warn(
                "SAEConfig(model=...) is deprecated, use SAEConfig(models=[...]) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.models = [self.model]

        if self.normalize_features:
            unsupported = [m for m in self.models if "300m" in m.lower()]
            if unsupported:
                raise ValueError(
                    f"normalize_features=True is not supported for ESMC 300M SAE models: {unsupported}. "
                    "Set normalize_features=False when using 300M SAE models."
                )


@define
class LogitsOutput:
    """
    logits (ForwardTrackData | None): Per-track categorical logits, populated for each
        track requested via LogitsConfig.
    embeddings (torch.Tensor | None): Per-residue embeddings (final hidden state).
        Returned when LogitsConfig.return_embeddings is set.
    mean_embedding (torch.Tensor | None): Embedding mean-pooled over the sequence
        length. Returned when LogitsConfig.return_mean_embedding is set.
    residue_annotation_logits (torch.Tensor | None): Residue annotation logits. These
        are multi-hot (bernoulli), so they are kept separate from `logits` (which holds
        categorical per-track logits).
    hidden_states (torch.Tensor | None): Hidden states for the requested layer(s).
        Returned when LogitsConfig.return_hidden_states is set.
    mean_hidden_state (torch.Tensor | None): Hidden states mean-pooled over the
        sequence length. Returned when LogitsConfig.return_mean_hidden_states is set.
    sae_outputs (dict[str, torch.Tensor] | None): SAE activations keyed by SAE model
        name. Returned when LogitsConfig.sae_config is set.
    """

    logits: ForwardTrackData | None = None
    embeddings: torch.Tensor | None = None
    mean_embedding: torch.Tensor | None = None

    # Residue annotations is multi-hot, so deserves special treatment
    # It's not a categorical distribution, but instead a bernoulli, so
    # softmax across the last dimension is _wrong_
    residue_annotation_logits: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    mean_hidden_state: torch.Tensor | None = None
    sae_outputs: dict[str, torch.Tensor] | None = None


@define
class ForwardAndSampleOutput(LogitsOutput):
    """Output of forward_and_sample. Extends LogitsOutput with the sampled tokens and
    per-position sampling statistics (each ForwardTrackData holds one value per track).

    protein_tensor (ESMProteinTensor): The sampled tokens.
    entropy (ForwardTrackData | None): Per-position entropy of the predicted
        distribution, per track.
    prob (ForwardTrackData | None): Probability of the sampled token at each position.
    logprob (ForwardTrackData | None): Log-probability of the sampled token at each
        position.
    top_prob (ForwardTrackData | None): Highest token probability at each position.
    topk_logprob (ForwardTrackData | None): Log-probabilities of the top-k tokens at
        each position. Populated when PerTrackSamplingConfig.topk_logprobs is set.
    topk_tokens (ForwardTrackData | None): Token ids of the top-k tokens at each
        position. Populated when PerTrackSamplingConfig.topk_logprobs is set.
    per_residue_embedding (torch.Tensor | None): Per-residue embeddings. Returned when
        SamplingConfig.return_per_residue_embeddings is set.
    mean_embedding (torch.Tensor | None): Embedding mean-pooled over the sequence
        length. Returned when SamplingConfig.return_mean_embedding is set.
    """

    protein_tensor: ESMProteinTensor = ESMProteinTensor()

    entropy: ForwardTrackData | None = None
    prob: ForwardTrackData | None = None
    logprob: ForwardTrackData | None = None
    top_prob: ForwardTrackData | None = None
    topk_logprob: ForwardTrackData | None = None
    topk_tokens: ForwardTrackData | None = None
    per_residue_embedding: torch.Tensor | None = None
    mean_embedding: torch.Tensor | None = None


class ESM3InferenceClient(ABC):
    model: str

    def generate(self, input: ProteinType, config: GenerationConfig) -> ProteinType:
        # This is the easiest and most flexible way to run ESM3. Generate will
        # iteratively sample tokens an provide an output with the track specified
        # completely filled out, according to the GenerationConfig provided.
        # It is a local function wrapping calls for encode -> iterative_sampling -> decode.
        # if a ESMProteinTensor is provided, encode and decode are skipped
        raise NotImplementedError

    async def async_generate(
        self, input: ProteinType, config: GenerationConfig
    ) -> ProteinType:
        raise NotImplementedError

    def batch_generate(
        self, inputs: Sequence[ProteinType], configs: Sequence[GenerationConfig]
    ) -> Sequence[ProteinType]:
        # Same as generate(...), but generates a batch of proteins at once.
        raise NotImplementedError

    async def async_batch_generate(
        self, inputs: Sequence[ProteinType], configs: Sequence[GenerationConfig]
    ) -> Sequence[ProteinType]:
        raise NotImplementedError

    def encode(self, input: ESMProtein) -> ESMProteinTensor:
        # Encode allows for encoding RawRepresentation into TokenizedRepresentation.
        # This runs the structure_token_encoder, as well as dealing with PDB => atom37 conversion
        raise NotImplementedError

    async def async_encode(self, input: ESMProtein) -> ESMProteinTensor:
        raise NotImplementedError

    def decode(self, input: ESMProteinTensor) -> ESMProtein:
        # Decode is the inverse of encode, and runs a structure_token_decoder to output coordinates
        raise NotImplementedError

    async def async_decode(self, input: ESMProteinTensor) -> ESMProtein:
        raise NotImplementedError

    def logits(
        self, input: ESMProteinTensor, config: LogitsConfig = LogitsConfig()
    ) -> LogitsOutput:
        # Our API generally discourages using raw forwards.
        # This is because sending logits can be prohibitively expensive.
        # Please use forward_and_sample instead.
        raise NotImplementedError

    async def async_logits(
        self, input: ESMProteinTensor, config: LogitsConfig = LogitsConfig()
    ) -> LogitsOutput:
        raise NotImplementedError

    def forward_and_sample(
        self, input: ESMProteinTensor, sampling_configuration: SamplingConfig
    ) -> ForwardAndSampleOutput:
        # forward_and_sample runs a single model forward, sampling tokens according to `SamplingConfiguration`.
        # This is the way for power users to run ESM3. We hope to design this in a way to enable high throughput
        # inference, as well as arbitrary chain-of-though invocations of ESM3.
        raise NotImplementedError

    async def async_forward_and_sample(
        self, input: ESMProteinTensor, sampling_configuration: SamplingConfig
    ) -> ForwardAndSampleOutput:
        raise NotImplementedError

    @property
    def raw_model(self):
        # Get underlying esm3 model of an inference client.
        raise NotImplementedError


class ESMCInferenceClient(ABC):
    model: str

    def encode(self, input: ESMProtein) -> ESMProteinTensor:
        # Encode allows for encoding RawRepresentation into TokenizedRepresentation.
        raise NotImplementedError

    async def async_encode(self, input: ESMProtein) -> ESMProteinTensor:
        raise NotImplementedError

    def decode(self, input: ESMProteinTensor) -> ESMProtein:
        # Decode is the inverse of encode
        raise NotImplementedError

    async def async_decode(self, input: ESMProteinTensor) -> ESMProtein:
        raise NotImplementedError

    def logits(
        self, input: ESMProteinTensor, config: LogitsConfig = LogitsConfig()
    ) -> LogitsOutput:
        raise NotImplementedError

    async def async_logits(
        self, input: ESMProteinTensor, config: LogitsConfig = LogitsConfig()
    ) -> LogitsOutput:
        raise NotImplementedError

    @property
    def raw_model(self):
        # Get underlying esmc model of an inference client.
        raise NotImplementedError
