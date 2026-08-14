"""Shared HuggingFace Hub loading for the opensource model packages.

Both ESMC and ESMFold2 publish plain safetensors checkpoints next to a
``config.json``, so the resolve/read/load sequence is identical. What differs is
each family's key policy, which subclasses control through
:meth:`HubPreTrainedModel._adapt_checkpoint_keys` and
:attr:`HubPreTrainedModel._keys_to_ignore_on_load_unexpected`.
"""

import json
import os
import re
from pathlib import Path
from typing import ClassVar, Self

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors.torch import load_file

CONFIG_NAME = "config.json"
_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SAFETENSORS_SINGLE = "model.safetensors"


def drop_te_extra_state(
    module: nn.Module, state_dict: dict, prefix: str, local_metadata: dict
) -> None:
    """Strip Transformer Engine's non-tensor ``_extra_state`` entries; TE returns
    these as ``io.BytesIO`` objects the checkpoint writer cannot dtype-infer."""
    for key in list(state_dict):
        if key.endswith("_extra_state"):
            del state_dict[key]


def read_safetensors_dir(directory: str | os.PathLike) -> dict[str, torch.Tensor]:
    """Read a (possibly sharded) safetensors checkpoint into a single dict."""
    directory = Path(directory)
    index = directory / _SAFETENSORS_INDEX
    if index.exists():
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        state_dict: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            state_dict.update(load_file(str(directory / shard)))
        return state_dict
    single = directory / _SAFETENSORS_SINGLE
    if single.exists():
        return load_file(str(single))
    raise FileNotFoundError(
        f"No safetensors checkpoint found in {directory} "
        f"(looked for {_SAFETENSORS_INDEX} and {_SAFETENSORS_SINGLE})."
    )


def resolve_model_dir(
    pretrained_model_name_or_path: str | os.PathLike,
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    token: str | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> str:
    """Local dir -> return as-is; hub id -> ``snapshot_download`` it."""
    path = Path(pretrained_model_name_or_path)
    if path.is_dir() and (path / CONFIG_NAME).exists():
        return str(path)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=str(pretrained_model_name_or_path),
        revision=revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
        token=token,
        allow_patterns=["*.json", "*.safetensors"],
        local_files_only=local_files_only,
        force_download=force_download,
    )


class HubPreTrainedModel(nn.Module):
    """Base class handling weight initialisation and Hub loading."""

    config_class: ClassVar[type]

    # Checkpoint keys with no counterpart in the module tree that are expected
    # anyway, as regexes. Anything unexpected and unlisted is an error.
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = []

    # Parameters a checkpoint is expected not to carry, as regexes - a freshly
    # added task head, say. Anything else the checkpoint misses is an error.
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = []

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _init_weights(self, module: nn.Module, device: torch.device | str = "cpu"):
        """Initialise a freshly materialised module. Overridden per family."""

    def post_init(self) -> None:
        """Mirror ``PreTrainedModel.post_init``: run per-module initialisation.

        Families that do not override :meth:`_init_weights` get a no-op, which
        is what HF did for these models too - their weights come entirely from
        the checkpoint.
        """
        self.apply(self._init_weights)

    def _materialize_uninitialized(self, device: torch.device | str = "cpu") -> None:
        """Materialize and initialize parameters/buffers still on the meta
        device after a partial checkpoint load (e.g. a freshly-added head or
        non-persistent RoPE buffers)."""
        for module in self.modules():
            direct = list(module._parameters.values()) + list(module._buffers.values())
            if any(t is not None and t.is_meta for t in direct):
                module.to_empty(device=device, recurse=False)
                self._init_weights(module, device=device)

    @classmethod
    def _adapt_checkpoint_keys(
        cls, raw: dict[str, torch.Tensor], model_keys: set[str]
    ) -> dict[str, torch.Tensor]:
        """Align published-checkpoint keys onto the target module tree.

        The default keeps every key untouched, so nothing is dropped silently.
        """
        return dict(raw)

    @classmethod
    def _normalize_checkpoint_layout(
        cls, raw: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Translate a checkpoint out of the published tensor layout.

        Runs before any key accounting, so a family whose in-memory parameters
        are packed differently from its published ones (ESMC fuses q/k/v and
        gate/up for Transformer Engine) is still held to the same
        nothing-dropped guarantee. The default is the identity.
        """
        return raw

    @classmethod
    def _load_pretrained(
        cls,
        local_dir: str | os.PathLike,
        config,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        strict_keys: bool = True,
    ) -> Self:
        """Instantiate ``cls(config)`` and load the checkpoint in ``local_dir``.

        With ``strict_keys`` the load refuses to discard anything: checkpoint keys
        that find no home raise, and so does any parameter the checkpoint failed
        to cover. Families list the keys they expect to have no counterpart in
        :attr:`_keys_to_ignore_on_load_unexpected`.
        """
        with init_empty_weights():
            model = cls(config)

        raw = cls._normalize_checkpoint_layout(read_safetensors_dir(local_dir))
        adapted = cls._adapt_checkpoint_keys(raw, set(model.state_dict().keys()))
        incompatible = model.load_state_dict(adapted, strict=False, assign=True)

        ignore = [re.compile(p) for p in cls._keys_to_ignore_on_load_unexpected]
        allow_missing = [re.compile(p) for p in cls._keys_to_ignore_on_load_missing]
        keep = [k for k in raw if not any(p.search(k) for p in ignore)]
        unexpected = [
            k
            for k in incompatible.unexpected_keys
            if not any(p.search(k) for p in ignore)
        ]
        # Counted by how many checkpoint entries reached the model, not by name:
        # key adaptation renames as well as drops, and comparing names counted
        # every rename as a loss. Entries that landed are the adapted ones minus
        # those the model rejected, so `unexpected` adds to the loss.
        dropped = len(keep) - (len(adapted) - len(unexpected))
        # The failure that actually corrupts a model: a parameter the checkpoint
        # never covered, left holding whatever `to_empty` found in memory.
        missing = [
            k
            for k in incompatible.missing_keys
            if not any(p.search(k) for p in allow_missing)
        ]

        if strict_keys and (unexpected or dropped > 0 or missing):
            raise RuntimeError(
                f"{cls.__name__}.from_pretrained refused to load the checkpoint "
                f"in {local_dir} silently.\n"
                f"  unexpected keys ({len(unexpected)}): {unexpected[:20]}\n"
                f"  missing keys ({len(missing)}): {missing[:20]}\n"
                f"  checkpoint entries that reached nothing: {dropped}"
            )

        model._materialize_uninitialized(device="cpu")
        model.to(device)
        if dtype is not None:
            model.to(dtype)
        model.eval()
        return model
