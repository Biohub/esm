import os
from typing import Any, Callable, Sequence
from warnings import warn

import attr
import torch
from tqdm import tqdm

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    ESMProteinTensor,
    ForwardAndSampleOutput,
    ForwardTrackData,
    GenerationConfig,
    LogitsConfig,
    LogitsOutput,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.tokenization import EsmTokenizerBase, TokenizerCollectionProtocol
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
from esm.utils.constants import esm3 as C
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.noise_schedules import NOISE_SCHEDULE_REGISTRY
from esm.utils.sampling import (
    _BatchedESMProteinTensor,
    get_sampling_mask,
    sample_function_logits,
    sample_logits,
    sample_residue_annotation_logits,
    sample_sasa_logits,
)


def _trim_sequence_tensor_dataclass(o: Any, sequence_len: int):
    """Trim tensors on the sequence dimension.

    This util assume that input tensor class has batch dimension.
    """
    assert attr.has(o.__class__)

    sliced = {}
    for k, v in attr.asdict(o, recurse=False).items():
        if k in ["mean_hidden_state", "mean_embedding"]:
            sliced[k] = v
        elif v is None:
            sliced[k] = None
        elif isinstance(v, torch.Tensor):
            # Trim padding.
            sliced[k] = v[:, :sequence_len]
        elif isinstance(v, tuple) and all(isinstance(t, torch.Tensor) for t in v):
            # Trim padding for a list of tensors
            sliced[k] = [t[:, :sequence_len] for t in v]
        elif attr.has(v.__class__):
            # Recursively slice the child attribute.
            sliced[k] = _trim_sequence_tensor_dataclass(v, sequence_len)
        else:
            # Otherwise, simply copy the entire data bit over.
            sliced[k] = v

    return attr.evolve(o, **sliced)


def _slice_tensor_dataclass(o: Any, i: int, keep_dim: bool = False) -> Any:
    """Take a slice out of any attr defined Tensor objects along the batch dimension.

    Args:
        o: input tensor object to be sliced.
        i: index of the row to be sliced.
        keep_dim: whether to keep the batch dim after slicing.
            For example, given a tensor of shape (5, 8), if keep_dim is True,
            return a sliced tensor of shape (1, 8). Return a tensor of shape
            (8,) instead if keep_dim is False. The default is False.
    """
    assert attr.has(o.__class__)

    sliced = {}
    for k, v in attr.asdict(o, recurse=False).items():
        if v is None:
            sliced[k] = None
        elif isinstance(v, torch.Tensor):
            # Select the i-th row of each tensor.
            row = v.select(0, i)
            if keep_dim:
                row = row.unsqueeze(0)
            sliced[k] = row
        elif attr.has(v.__class__):
            # Recursively slice the child attribute.
            sliced[k] = _slice_tensor_dataclass(v, i, keep_dim)
        else:
            # Otherwise, simply copy the entire data bit over.
            sliced[k] = v

    return attr.evolve(o, **sliced)


def iterative_sampling_raw(
    client: ESM3InferenceClient,
    proteins: list[ESMProtein],
    configs: list[GenerationConfig],
) -> list[ESMProtein | ESMProteinError]:
    # Keep structure tokens
    input_tokens = [client.encode(protein) for protein in proteins]

    output_tokens_list = client.batch_generate(input_tokens, configs)

    raw_proteins: list[ESMProtein | ESMProteinError] = []
    for output_tokens in output_tokens_list:
        if isinstance(output_tokens, ESMProteinTensor):
            raw_proteins.append(client.decode(output_tokens))
        elif isinstance(output_tokens, ESMProteinError):
            raw_proteins.append(output_tokens)
        else:
            raise ValueError(f"Unknown output type {type(output_tokens)}")

    for input_protein, raw_protein, config in zip(proteins, raw_proteins, configs):
        if isinstance(raw_protein, ESMProteinError):
            # If this generation errored out.
            continue
        if config.track not in ["function", "residue_annotations"]:
            # Function and residue annotation encoding/decoding is lossy
            # There is no guarantee that decoding encoded tokens will yield the same input
            raw_protein.function_annotations = input_protein.function_annotations

    return raw_proteins


def _make_masked_inputs(
    track: str, sequence_length: int, tokenizers: TokenizerCollectionProtocol
):
    get_tokenizer: Callable[[str], EsmTokenizerBase] = lambda s: getattr(tokenizers, s)
    has_tokenizer: Callable[[str], bool] = lambda s: hasattr(tokenizers, s)

    if track == "coordinates":
        dims = (sequence_length, 3, 3)
    elif track == "confidence":
        dims = (sequence_length,)
    elif track == "attention_mask":
        dims = (sequence_length,)
    elif track == "function":
        dims = (sequence_length, tokenizers.function.depth)
    elif track == "residue_annotations":
        dims = (sequence_length, C.MAX_RESIDUE_ANNOTATIONS)
    else:
        dims = (sequence_length,)

    if track == "coordinates":
        masked_tokens = torch.full(dims, torch.inf, dtype=torch.float)
    elif track == "confidence":
        # All-mask dummy input for confidence track.
        masked_tokens = torch.full(dims, 0.0)
    elif track == "attention_mask":
        masked_tokens = torch.full(dims, 1, dtype=torch.bool)
    elif has_tokenizer(track):
        masked_tokens = torch.full(
            dims, get_tokenizer(track).mask_token_id, dtype=torch.long
        )
        masked_tokens[0] = get_tokenizer(track).bos_token_id
        masked_tokens[-1] = get_tokenizer(track).eos_token_id
    else:
        # Does not know how to create the dummy all masked input.
        return None

    return masked_tokens


def _stack_protein_tensors(
    input_tokens: list[ESMProteinTensor],
    sequence_lengths: list[int],
    tokenizers: TokenizerCollectionProtocol,
    device: str | torch.device,
) -> _BatchedESMProteinTensor:
    o = _BatchedESMProteinTensor()

    def _maybe_mock_input(fn, t, l):
        if t is not None:
            return t

        # Try create dummy masked input for this prompt.
        t = _make_masked_inputs(fn, l, tokenizers)
        if t is not None:
            t = t.to(device)

        return t

    def _stack_field(fn: str):
        tensors = [getattr(tokens, fn) for tokens in input_tokens]

        # Create all mask mock inputs for any tensors that are None.
        tensors = [
            _maybe_mock_input(fn, t, l) for t, l in zip(tensors, sequence_lengths)
        ]

        # Handle any track that has all None as the input.
        # We can't meaningfully stack tensors in this case, so simply batched
        # them as None in _BatchedESMProteinTensor.
        if all([t is None for t in tensors]):
            setattr(o, fn, None)
            return

        if fn == "coordinates":
            mask_token_id = torch.inf
        else:
            mask_token_id = getattr(tokenizers, fn).pad_token_id

        setattr(
            o,
            fn,
            stack_variable_length_tensors(
                sequences=tensors, constant_value=mask_token_id
            ),
        )

    for f in attr.fields(ESMProteinTensor):
        # We do not batch potential_sequence_of_concern field.
        if f.name == "potential_sequence_of_concern":
            continue
        _stack_field(f.name)

    return o


def _get_masked_positions(
    track: str, tokens: torch.Tensor, mask_token_id: int
) -> torch.Tensor:
    if track == "function":
        mask = torch.all(tokens == mask_token_id, dim=-1).to(tokens.device)
    else:
        mask = tokens == mask_token_id

    # Should not sample BOS and EOS positions.
    mask[..., 0] = False
    mask[..., -1] = False

    return mask


def _get_iterative_sampling_mask_for_prompt_and_step(
    cur_sampled: _BatchedESMProteinTensor,
    sequence_lengths: torch.Tensor,
    total_to_sample: torch.Tensor,
    step: int,
    entropy: ForwardTrackData,
    config: GenerationConfig,
    tokenizers: TokenizerCollectionProtocol,
) -> torch.Tensor:
    """Get sampling mask based on forward output and config.

    Returns:
        Sampling mask and num of positions sampled.
    """
    track_to_sample = config.track
    tokens = getattr(cur_sampled, track_to_sample)
    device = tokens.device

    shape = tokens.shape
    B, L = shape[0], shape[1]

    # TODO: figure out why we want this function to work with
    # _BatchedESMProteinTensor in the first place. Logics below
    # don't really work for batched tensors.
    assert B == 1

    sampling_mask = torch.ones((B, L), dtype=torch.bool, device=device)
    sampling_mask[:, 0] = False  # BOS
    # EOS and all padding tokens.
    sampling_mask &= (
        torch.arange(L).repeat(B, 1) < (sequence_lengths - 1).unsqueeze(-1)
    ).to(device)

    is_mask = _get_masked_positions(
        track_to_sample, tokens, getattr(tokenizers, track_to_sample).mask_token_id
    )
    if not is_mask.any().item():
        raise ValueError(f"Cannot sample {config.track} when input has no masks.")
    sampling_mask = sampling_mask & is_mask

    # Initialize schedule and masks
    decoding_schedule = NOISE_SCHEDULE_REGISTRY[config.schedule]

    # Calculate number of tokens to sample
    still_masked = torch.sum(sampling_mask).int()
    perc_masked_after_this_step = decoding_schedule(
        torch.tensor((step + 1) / config.num_steps)
    )
    num_tokens_masked_after_this_step = (
        # To avoid rounding errors, add a small epsilon.
        # NOTE: Tensor.round does not cast to int,
        # so it actually leads to rounding down.
        # e.g. tensor(67.0000).int() = 66
        perc_masked_after_this_step * total_to_sample + 0.1
    ).int()
    num_to_sample = still_masked - num_tokens_masked_after_this_step

    if config.strategy == "entropy":
        track_entropy: torch.Tensor = getattr(entropy, track_to_sample).to(
            device
        )  # (B, L) or (B, L, D)

        if track_to_sample == "function":
            track_entropy = track_entropy.sum(-1)  # (B, L, D) -> (B, L)

        track_entropy = track_entropy.masked_fill(
            ~sampling_mask, torch.finfo(track_entropy.dtype).max
        )
        _, indices = track_entropy.topk(num_to_sample, dim=-1, largest=False)
        is_top_k = torch.zeros((B, L), dtype=torch.bool, device=device).scatter(
            1, indices, True
        )
        where_to_sample = sampling_mask & is_top_k
    elif config.strategy == "random":
        # Skip B since we know there is only 1 prompt here.
        _, masked_indices = sampling_mask.nonzero(as_tuple=True)
        # Random shuffle the masked indices then select the first num_to_sample.
        rnd_indices = masked_indices[torch.randperm(len(masked_indices))][
            :num_to_sample
        ]
        rnd_mask = torch.zeros_like(sampling_mask)
        rnd_mask[:, rnd_indices] = True
        where_to_sample = sampling_mask & rnd_mask

    if track_to_sample == "function":
        where_to_sample = where_to_sample.unsqueeze(-1).expand(
            B, L, tokenizers.function.depth
        )  # (B, L) -> (B, L, D)

    return where_to_sample


def _get_non_special_tokens(
    protein: ESMProteinTensor, tokenizers: TokenizerCollectionProtocol
) -> int:
    if protein.sequence is None:
        # There is no sequence to infer the number of tokens to decode.
        # So we assume the entire sequence minus bos and eos are for decoding.
        return len(protein) - 2

    mask = torch.ones_like(protein.sequence)
    for special_token in tokenizers.sequence.special_token_ids:
        if special_token == tokenizers.sequence.mask_token_id:
            continue  # MASK tokens need to be sampled.
        mask[protein.sequence == special_token] = 0

    return int(torch.sum(mask).item())


def _get_annealed_temperature(step: int, num_steps: int, initial_temperature: float):
    step_ratio = step / max(1, (num_steps - 1))
    return max(initial_temperature - step_ratio, 0.001) ** 2


def iterative_sampling_tokens(
    client: ESM3InferenceClient,
    input_tokens: list[ESMProteinTensor],
    configs: list[GenerationConfig],
    tokenizers: TokenizerCollectionProtocol,
) -> Sequence[ESMProteinTensor | ESMProteinError]:
    devices = set([t.device for t in input_tokens])
    if len(devices) > 1:
        raise AttributeError(f"Input tokens on multiple devices {devices}")

    sampled_tokens = [attr.evolve(tokens) for tokens in input_tokens]

    # Clear structure tokens if user would like to condition only on coordinates.
    for tokens, config in zip(sampled_tokens, configs):
        if config.condition_on_coordinates_only and tokens.coordinates is not None:
            tokens.structure = None

    # Total sequence lengths.
    sequence_lengths = [len(tokens) for tokens in sampled_tokens]
    # Figure out the number of tokens to be sampled for each prompt.
    total_to_sample = []
    for protein, config in zip(sampled_tokens, configs):
        track = config.track

        if getattr(protein, track) is None:
            # We need to sample the entire track.
            num_sampling_steps = _get_non_special_tokens(protein, tokenizers)
        else:
            masked = _get_masked_positions(
                track, getattr(protein, track), getattr(tokenizers, track).mask_token_id
            )
            num_sampling_steps = torch.sum(masked).item()

        total_to_sample.append(num_sampling_steps)

        # Users might over-specify the number of sampling steps for a given prompt
        # TODO: Give a warning about mismatched num_steps and number of masks.
        if (num_sampling_steps > 0) and (num_sampling_steps < config.num_steps):
            config.num_steps = int(num_sampling_steps)

    # Different prompts may ask for different number of decoding steps.
    # For now, we simply run the max number of steps.
    # TODO: return completed proteins as soon as they are finished sampling.
    max_num_steps = max([config.num_steps for config in configs])

    # Now stack the list to make a single batched ESMProteinTensor.
    batched_tokens = _stack_protein_tensors(
        sampled_tokens, sequence_lengths, tokenizers, devices.pop()
    )

    # Remember sampled prompts that has somehow errored out.
    errors: dict[int, ESMProteinError] = {}

    # Decode
    disable_tqdm = bool(os.environ.get("DISABLE_ITERATIVE_SAMPLING_TQDM", False))
    for t in tqdm(range(max_num_steps), disable=disable_tqdm):
        forward_out = _batch_forward(client, batched_tokens)

        # Sample each prompt individually, since their configuration may
        # be very different.
        # TODO: downstream utils work with batch dimsension.
        # Group by sampling configurations and sample those prompts together.
        for i, config in enumerate(configs):  # B
            if i in errors:
                # This prompts has errored out in previous steps.
                # Skip.
                continue

            if config.track in ["coordinates", "residue_annotations"]:
                errors[i] = ESMProteinError(
                    error_code=500,
                    error_msg=f"Iterative sampling {config.track} is not supported.",
                )
                continue

            if t >= config.num_steps:
                # Done sampling for this row.
                continue

            per_prompt_cur_sampled = _BatchedESMProteinTensor.from_protein_tensor(
                batched_tokens.slice(i)
            )
            per_prompt_forward_out: LogitsOutput = _slice_tensor_dataclass(
                forward_out, i, keep_dim=True
            )
            # Trim logits to proper sequence length for this prompt.
            per_prompt_forward_out = _trim_sequence_tensor_dataclass(
                per_prompt_forward_out,
                # Note(jungong) : we can not smiply use sequence_lenths[i] here,
                # what we want is for the sequence length of the logits to match
                # that of the prompt, which may or may not be padded, depending on
                # whether the padding was done locally with the open source model
                # (where per_prompt_cur_sampled is already padded) or by
                # BatchedESM3ModelRunner (where per_prompt_cur_sampled is not padded).
                len(per_prompt_cur_sampled),
            )

            # Handle temperature annealing, since _sample_per_prompt() doesn't have
            # the concept of decoding steps.
            if config.temperature_annealing:
                temperature = _get_annealed_temperature(
                    t, config.num_steps, config.temperature
                )
            else:
                temperature = config.temperature

            track_sample_config = SamplingTrackConfig()
            track_sample_config.invalid_ids = config.invalid_ids
            track_sample_config.temperature = temperature
            track_sample_config.top_p = config.top_p
            sampling_config = SamplingConfig(**{config.track: track_sample_config})  # type: ignore

            # Sampling has to be done per-prompt, since sampling configs
            # are likely be different for different prompts.
            per_prompt_forward_and_sample_output = _sample_per_prompt(
                per_prompt_cur_sampled,
                per_prompt_forward_out,
                sampling_config,
                tokenizers,
                decode_sasa_tokens=False,
            )

            # All positions sampled after _sample_per_prompt() above.
            # (B, L) & (B, L, D)
            per_prompt_new_sampled = per_prompt_forward_and_sample_output.protein_tensor

            # Find the positions we should sample this round.
            assert per_prompt_forward_and_sample_output.entropy is not None
            try:
                where_to_sample = _get_iterative_sampling_mask_for_prompt_and_step(
                    per_prompt_cur_sampled,
                    torch.tensor(sequence_lengths[i]),
                    torch.tensor(total_to_sample[i]),
                    t,
                    per_prompt_forward_and_sample_output.entropy,
                    config,
                    tokenizers,
                )
            except ValueError as e:
                errors[i] = ESMProteinError(error_code=500, error_msg=str(e))
                continue

            where_to_sample.to(input_tokens[0].device)

            old_track_samples = getattr(per_prompt_cur_sampled, config.track)
            new_track_samples = getattr(per_prompt_new_sampled, config.track)

            # Iterative sampling by picking the tokens sampled this round
            # from new_track_samples to old_track_samples.
            new_track_samples = torch.where(
                where_to_sample, new_track_samples, old_track_samples
            )

            # Update the corresponding row with new data.
            getattr(batched_tokens, config.track)[i, ...] = new_track_samples[0]

    # Un-pack to a list of single ProteinTypes.
    output_tokens = [
        batched_tokens.slice(i, sequence_len=sequence_lengths[i])
        if i not in errors
        else errors[i]
        for i in range(len(input_tokens))
    ]

    # Do not update tracks that were not sampled (e.g. keep None instead of masks)
    for inputs, outputs, config in zip(input_tokens, output_tokens, configs):
        if isinstance(outputs, ESMProteinError):
            continue

        # First restore coordinates field.
        # We know coordinates can never be iteratively sampled.
        setattr(outputs, "coordinates", getattr(inputs, "coordinates"))
        # Maybe restore all the other fields.
        for f in attr.fields(SamplingConfig):
            if "embedding" in f.name or f.name == "return_hidden_states":
                continue
            if f.name != config.track:
                setattr(outputs, f.name, getattr(inputs, f.name))

    return output_tokens


def _batch_forward(client: ESM3InferenceClient, protein: _BatchedESMProteinTensor):
    # Forward pass
    return client.logits(
        protein,
        LogitsConfig(
            sequence=True,
            structure=True,
            secondary_structure=True,
            sasa=True,
            function=True,
            residue_annotations=True,
            return_embeddings=True,
        ),
    )


def _sample_per_prompt(
    protein: _BatchedESMProteinTensor,
    logits_output: LogitsOutput,
    sampling_config: SamplingConfig,
    tokenizers: TokenizerCollectionProtocol,
    decode_sasa_tokens: bool = True,
    mask_logits_of_invalid_ids: bool = True,
) -> ForwardAndSampleOutput:
    assert logits_output.logits is not None

    def maybe_clone(x: torch.Tensor | None) -> torch.Tensor | None:
        return x.clone() if x is not None else None

    # Sampling
    tokens_dir = {}
    track_sampling_metadata_dir: dict[str, dict | None] = {}
    integer_sampling_tracks = ["sequence", "structure", "secondary_structure"]
    if not decode_sasa_tokens:
        integer_sampling_tracks.append("sasa")

    for track in integer_sampling_tracks:
        config = getattr(sampling_config, track)
        if config is None:
            tokens_dir[track] = maybe_clone(getattr(protein, track))
            continue
        tokenizer = getattr(tokenizers, track)
        valid_ids = (
            set(tokenizer.all_token_ids)
            - set(tokenizer.special_token_ids)
            - set(config.invalid_ids)
        )
        sampling_metadata = _sample_track(
            logits=getattr(logits_output.logits, track),
            tokens=getattr(protein, track),
            sampling_track_config=config,
            mask_idx=getattr(tokenizers, track).mask_token_id,
            valid_ids=list(valid_ids),
            mask_logits_of_invalid_ids=mask_logits_of_invalid_ids,
        )
        tokens_dir[track] = sampling_metadata.pop("sampled_tokens")  # (L,)
        track_sampling_metadata_dir[track] = sampling_metadata

    # Sample SASA seperately (if needed)
    if decode_sasa_tokens:
        config = getattr(sampling_config, "sasa")
        track_sampling_metadata_dir["sasa"] = None

        if config is None:
            tokens_dir["sasa"] = maybe_clone(getattr(protein, "sasa"))
        else:
            if config.topk_logprobs > 0:
                warn("For SASA sampling, 'topk_logprobs' is expected to be 0.")

            assert logits_output.logits.sasa is not None
            assert protein.sasa is not None

            valid_ids = (
                set(tokenizers.sasa.all_token_ids)
                - set(tokenizers.sasa.special_token_ids)
                - set(config.invalid_ids)
            )
            sasa_logits = logits_output.logits.sasa
            sasa_value = sample_sasa_logits(
                sasa_logits,
                protein.sasa,
                sampling_track_config=config,
                mask_idx=tokenizers.sasa.mask_token_id,
                valid_ids=list(valid_ids),
                mask_logits_of_invalid_ids=mask_logits_of_invalid_ids,
            )
            tokens_dir["sasa"] = sasa_value

            probs = sasa_logits.softmax(dim=-1)
            # Note(tjia): sasa_logits can have -inf because of invalid ids, so
            # probs * sasa_logits.log_softmax(-1) is nan. We need to set
            # those positions to 0 to get the correct entropy value
            entropy = -(torch.nan_to_num(probs * sasa_logits.log_softmax(-1))).sum(-1)

            track_sampling_metadata_dir["sasa"] = {"entropy": entropy}

    # Sample function and residue annotations separately
    config = getattr(sampling_config, "function")
    function_logits = getattr(logits_output.logits, "function")
    if config is None or function_logits is None:
        tokens_dir["function"] = maybe_clone(getattr(protein, "function"))
        tokens_dir["residue_annotations"] = maybe_clone(
            getattr(protein, "residue_annotations")
        )
    else:
        if config.invalid_ids is not None and len(config.invalid_ids) > 0:
            warn("For function sampling, invalid_ids sampling config is not supported.")

        sampling_metadata = _sample_function_track(
            tokenizers.function,
            tokens=getattr(protein, "function"),
            logits=function_logits,
            sampling_track_config=config,
        )
        tokens_dir["function"] = sampling_metadata.pop("sampled_tokens")  # (L, D)
        track_sampling_metadata_dir["function"] = sampling_metadata

        sampled_tokens, _ = sample_residue_annotation_logits(
            logits=logits_output.residue_annotation_logits  # type: ignore
        )
        tokens_dir["residue_annotations"] = sampled_tokens  # (L, MAX_R)

    # Format output
    forward_and_sample_output_dir = {}
    forward_and_sample_output_dir["protein_tensor"] = ESMProteinTensor(
        **tokens_dir  # ty:ignore[invalid-argument-type]
    )
    for property in [
        "entropy",
        "prob",
        "logprob",
        "top_prob",
        "topk_logprob",
        "topk_tokens",
    ]:
        is_all_none = True
        forward_track_data_dir = {}
        for track in track_sampling_metadata_dir.keys():
            values = track_sampling_metadata_dir[track]
            if values is not None and values.get(property, None) is not None:
                forward_track_data_dir[track] = values.get(property, None)
                is_all_none = False
        if not is_all_none:
            forward_and_sample_output_dir[property] = ForwardTrackData(
                **forward_track_data_dir
            )
        else:
            forward_and_sample_output_dir[property] = None

    per_res_embed = (
        logits_output.embeddings
        if sampling_config.return_per_residue_embeddings
        else None
    )
    mean_embedding = (
        # [B, L, D] -> [B, D]
        logits_output.embeddings.mean(dim=1)  # type: ignore
        if sampling_config.return_mean_embedding
        else None
    )

    return ForwardAndSampleOutput(
        per_residue_embedding=per_res_embed,
        mean_embedding=mean_embedding,
        **forward_and_sample_output_dir,  # ty:ignore[invalid-argument-type]
    )


def _sample_track(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    sampling_track_config: SamplingTrackConfig,
    mask_idx: int,
    valid_ids: list[int],
    mask_logits_of_invalid_ids: bool = True,
) -> dict[str, torch.Tensor]:
    """Works with inputs that have batch dimension."""
    # Sample in all positions
    temperature = sampling_track_config.temperature
    # We have to trim the logits and sampled tokens at potentially padded slots
    # since the logits may be computed with a longer padded batch, while tokens
    # are the original input sequence.
    sampled_tokens = sample_logits(
        logits,
        temperature=temperature,
        valid_ids=valid_ids,
        top_p=sampling_track_config.top_p,
        mask_logits_of_invalid_ids=mask_logits_of_invalid_ids,
    )
    log_probs = logits.log_softmax(-1)
    sampling_mask = get_sampling_mask(tokens, sampling_track_config, mask_idx)
    sampled_tokens = torch.where(sampling_mask, sampled_tokens, tokens)

    return _compute_track_metadata(
        sampled_tokens,
        log_probs,
        sampling_mask,
        top_k=sampling_track_config.topk_logprobs,
    )


def _sample_function_track(
    function_tokenizer: InterProQuantizedTokenizer,
    tokens: torch.Tensor,
    logits: torch.Tensor,
    sampling_track_config: SamplingTrackConfig,
) -> dict[str, torch.Tensor]:
    """Works with inputs that have batch dimension."""
    # Do not sample at BOS and EOS tokens
    sampling_mask = torch.ones_like(tokens, dtype=torch.bool)[..., 0]  # (B, L)
    sampling_mask[..., 0] = False
    sampling_mask[..., -1] = False

    sampled_tokens, logprobs = sample_function_logits(
        logits,
        function_tokenizer,
        top_p=sampling_track_config.top_p,
        temperature=sampling_track_config.temperature,
    )
    if sampling_track_config.only_sample_masked_tokens:
        is_mask = torch.all(
            tokens == function_tokenizer.mask_token_id, dim=-1
        )  # (B, L)
        sampling_mask = sampling_mask & is_mask

    sampled_tokens = torch.where(
        sampling_mask[..., None].expand_as(sampled_tokens), sampled_tokens, tokens
    )  # (B, L, D)

    # Set logprobs for non-sampled tokens to 0
    logprobs_null = torch.full_like(logprobs, -torch.inf)  # (B, L, D, V)
    logprobs_null = torch.scatter(
        logprobs_null, -1, tokens[..., None], torch.zeros_like(logprobs_null)[..., [0]]
    )
    logprobs = torch.where(
        sampling_mask[..., None, None].expand_as(logprobs), logprobs, logprobs_null
    )  # (B, L, D, V)

    function_metadata = _compute_track_metadata(
        sampled_tokens,
        logprobs,
        sampling_mask,
        top_k=sampling_track_config.topk_logprobs,
    )
    # Consider the entropy of the joint distribution of all function tokens at each position
    function_metadata["entropy"] = function_metadata["entropy"].sum(
        -1
    )  # (B, L, D) -> (B, L)
    return function_metadata


def _compute_track_metadata(
    sampled_tokens: torch.Tensor,
    log_probs: torch.Tensor,
    sampling_mask: torch.Tensor,
    top_k: int,
) -> dict:
    """Works with inputs that have batch dimension."""
    probs = torch.exp(log_probs)  # (B, L)
    entropy = torch.distributions.Categorical(logits=log_probs).entropy()  # (B, L)

    # Only compute probabilities for sampled tokens
    sampled_logprob = torch.zeros_like(sampled_tokens, dtype=log_probs.dtype)  # (B, L)

    if sampled_tokens.dim() > sampling_mask.dim():
        assert sampled_tokens.dim() == 3  # (B, L, D)
        assert sampling_mask.dim() == 2  # (B, L)
        sampling_mask = sampling_mask[..., None].expand_as(sampled_tokens)

    sampled_tokens_valid = sampled_tokens[sampling_mask]
    sampled_log_probs_valid = log_probs[sampling_mask, sampled_tokens_valid]
    sampled_logprob[sampling_mask] = sampled_log_probs_valid

    # Calculate extra metadata
    sampled_prob = torch.exp(sampled_logprob)
    top_prob = torch.max(probs, dim=-1).values
    topk_logprobs, topk_tokens = torch.topk(log_probs, top_k, dim=-1)
    topk_logprobs = None if top_k == 0 else topk_logprobs
    topk_tokens = None if top_k == 0 else topk_tokens

    return {
        "entropy": entropy,
        "sampled_tokens": sampled_tokens,
        "prob": sampled_prob,
        "logprob": sampled_logprob,
        "top_prob": top_prob,
        "topk_logprob": topk_logprobs,
        "topk_tokens": topk_tokens,
    }
