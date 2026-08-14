"""ESMC sparse autoencoder (SAE) model.

* :class:`EsmcSaeModel` - the published container, one repo per
  ``(backbone, codebook_dim, k)`` group. Each backbone layer ships as a
  ``layer_{i}.safetensors`` shard; ``from_pretrained`` downloads the whole
  snapshot but loads no weights - callers materialize the layers they need via
  :meth:`initialize_layers`. Single-layer repos auto-load so a bare
  ``forward(x)`` works.
* :class:`EsmcSaeLayer` - the ``nn.Module`` that holds the weights for one
  ``(backbone, codebook_dim, k, layer)`` SAE. Obtained via
  ``model.layers["<idx>"]``.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

CONFIG_NAME = "config.json"


@dataclass
class EsmcSaeParams:
    """Parameters for one backbone layer's SAE inside :class:`EsmcSaeModel`."""

    d_model: int = 2560
    codebook_dim: int = 65536
    k: int = 64
    layer: int = 0


@dataclass
class EsmcSaeConfig:
    """Configuration for :class:`EsmcSaeModel`.

    A container holds one SAE per backbone layer for a fixed
    ``(model, codebook_dim, k)`` group. All SAEs in a container share
    ``d_model``, ``codebook_dim`` and ``k``; they differ only in the backbone
    layer they were trained on. ``available_layers`` lists the backbone-layer
    indices the repo ships; each entry ``i`` is stored on disk as
    ``layer_{i}.safetensors``.

    Parameters
    ----------
    d_model : int
        Dimensionality of the ESMC hidden states fed into the SAEs.
    codebook_dim : int
        Number of sparse features in each SAE's codebook.
    k : int
        Top-k sparsity per SAE.
    available_layers : list[int]
        Which backbone-layer indices the repo ships.
    """

    model_type = "esmc_sae"

    d_model: int = 2560
    codebook_dim: int = 65536
    k: int = 64
    available_layers: list[int] | None = None

    def __post_init__(self) -> None:
        self.available_layers = (
            list(self.available_layers) if self.available_layers is not None else [0]
        )

    @classmethod
    def from_pretrained(cls, directory: str | os.PathLike) -> "EsmcSaeConfig":
        with open(Path(directory) / CONFIG_NAME) as f:
            raw = json.load(f)
        return cls(
            d_model=raw["d_model"],
            codebook_dim=raw["codebook_dim"],
            k=raw["k"],
            available_layers=raw.get("available_layers"),
        )

    def save_pretrained(self, directory: str | os.PathLike) -> None:
        payload = asdict(self)
        payload["model_type"] = self.model_type
        with open(Path(directory) / CONFIG_NAME, "w") as f:
            json.dump(payload, f, indent=2)


@dataclass
class EsmcSaeOutput:
    """Output of :class:`EsmcSaeModel` and :class:`EsmcSaeLayer`."""

    feature_magnitudes: torch.Tensor
    reconstruction_loss: torch.Tensor | None = None

    def to_sparse(self) -> None:
        self.feature_magnitudes = self.feature_magnitudes.to_sparse()


class EsmcSaeLayer(nn.Module):
    """One backbone layer's SAE - building block of :class:`EsmcSaeModel`.

    Obtained via ``model.layers["<layer_idx>"]`` after calling
    ``initialize_layers``.
    """

    idf: torch.Tensor
    max: torch.Tensor

    def __init__(self, params: EsmcSaeParams):
        super().__init__()
        self.params = params

        self.W_enc = nn.Parameter(torch.empty(params.d_model, params.codebook_dim))
        self.W_dec = nn.Parameter(torch.empty(params.codebook_dim, params.d_model))
        self.b_dec = nn.Parameter(torch.zeros(params.d_model))
        # Per-feature normalization stats. Trained alongside the SAE for some
        # variants; leaving these as ones makes the ``features / max * idf``
        # scaling in the backbone a no-op for variants that don't ship them.
        self.register_buffer("idf", torch.ones(params.codebook_dim))
        self.register_buffer("max", torch.ones(params.codebook_dim))

    @property
    def layer(self) -> int:
        """Backbone-layer index this SAE is trained against."""
        return self.params.layer

    def forward(self, x: torch.Tensor, **_kwargs: object) -> EsmcSaeOutput:
        del _kwargs
        x = self._zscore_normalize_representation(x)

        x_with_pre_encoder_bias = x - self.b_dec
        preactivations = F.relu(x_with_pre_encoder_bias @ self.W_enc)

        topk = torch.topk(preactivations, self.params.k, dim=-1)
        feature_magnitudes = torch.zeros_like(preactivations).scatter(
            -1, topk.indices, topk.values
        )

        reconstructed = feature_magnitudes @ self.W_dec + self.b_dec
        reconstruction_loss = (reconstructed - x).pow(2).mean(dim=-1)

        return EsmcSaeOutput(
            feature_magnitudes=feature_magnitudes,
            reconstruction_loss=reconstruction_loss,
        )

    def get_sae_output(
        self, layer_states: torch.Tensor, token_mask: torch.Tensor
    ) -> EsmcSaeOutput:
        _, _, v_len = layer_states.shape
        nonpad_states = layer_states[token_mask].view(-1, v_len)
        return self(nonpad_states)

    def _zscore_normalize_representation(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=-1, keepdim=True)
        x = x - x_mean
        x_std = x.std(dim=-1, keepdim=True)
        return x / (x_std + 1e-5)


class EsmcSaeModel(nn.Module):
    """Container holding one SAE per backbone layer, all sharing the same
    ``(d_model, codebook_dim, k)``.

    ``from_pretrained`` downloads the entire repo (every
    ``layer_{i}.safetensors``) into the local cache but does **not** load any
    weights into memory. Callers materialize the layers they actually need with
    :meth:`initialize_layers`. The full set is available on disk after the
    first call, so subsequent layer switches read from the local cache without
    re-downloading.

    Examples
    --------
    >>> model = EsmcSaeModel.from_pretrained(
    ...     "biohub/esmc-6b-2024-12-sae-k64-codebook16384"
    ... )
    >>> model.initialize_layers([60])  # into memory
    >>> out = model(layer_states, layer=60)  # forward through layer 60
    >>> model.initialize_layers([45])  # add layer 45 (cached locally)
    >>> model.release_layer(60)  # free layer 60
    """

    _device_marker: torch.Tensor

    def __init__(self, config: EsmcSaeConfig):
        super().__init__()
        self.config = config
        # Layers are populated lazily by ``initialize_layers``; the container
        # starts empty so ``from_pretrained`` doesn't materialize hundreds of
        # GB of unused parameters.
        self.layers = nn.ModuleDict()
        # Zero-element buffer that rides along with ``.to(device/dtype)``.
        # ``initialize_layers`` reads its current device/dtype so SAEs added
        # after ``model.to("cuda")`` land on CUDA without re-passing ``device=``.
        self.register_buffer("_device_marker", torch.empty(0), persistent=False)
        self._snapshot_dir: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *,
        revision: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        token: str | None = None,
        allow_patterns: list[str] | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "EsmcSaeModel":
        """Download (or reuse cached) the full repo and return the model.

        By default no weights are read into memory and the caller must invoke
        :meth:`initialize_layers` before running :meth:`forward`. The single
        exception is a repo that ships exactly one layer: that layer is
        auto-loaded (honoring ``device`` / ``dtype``) so the bare
        ``forward(x)`` call just works.
        """
        local_dir = _resolve_snapshot_dir(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=allow_patterns,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        config = EsmcSaeConfig.from_pretrained(local_dir)
        model = cls(config)
        model._snapshot_dir = str(local_dir)
        if device is not None:
            model.to(device)
        if dtype is not None:
            model.to(dtype)
        assert config.available_layers is not None
        if len(config.available_layers) == 1:
            model.initialize_layers(list(config.available_layers))
        return model

    def initialize_layers(
        self,
        layers: list[int],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Load the requested layers from the local snapshot into memory.

        Layers already present in :attr:`layers` are skipped - calling
        ``initialize_layers([23])`` twice is idempotent. ``device`` / ``dtype``
        default to wherever the model itself lives (via the ``_device_marker``
        buffer that moves with ``.to(...)``).
        """
        assert self._snapshot_dir is not None, (
            "EsmcSaeModel has no snapshot directory - call from_pretrained "
            "first, or set _snapshot_dir manually."
        )
        if device is None:
            device = self._device_marker.device
        if dtype is None:
            dtype = self._device_marker.dtype
        snapshot_dir = Path(self._snapshot_dir)
        assert self.config.available_layers is not None
        available = set(self.config.available_layers)
        for layer_idx in layers:
            key = str(layer_idx)
            if key in self.layers:
                continue
            if layer_idx not in available:
                raise KeyError(
                    f"Layer {layer_idx} is not in this repo. "
                    f"available_layers={sorted(available)}"
                )
            shard = snapshot_dir / f"layer_{layer_idx}.safetensors"
            if not shard.exists():
                raise FileNotFoundError(
                    f"Missing layer file {shard} - config lists layer "
                    f"{layer_idx} as available but the shard is not on disk."
                )
            params = EsmcSaeParams(
                d_model=self.config.d_model,
                codebook_dim=self.config.codebook_dim,
                k=self.config.k,
                layer=layer_idx,
            )
            # Build on the meta device so we don't allocate weights that
            # ``load_state_dict`` would immediately overwrite.
            with torch.device("meta"):
                layer = EsmcSaeLayer(params)
            layer.to_empty(device=device)
            layer.load_state_dict(load_file(str(shard)))
            layer.to(dtype=dtype)
            self.layers[key] = layer

    def release_layer(self, layer: int) -> None:
        """Drop the named layer from memory. No-op if not loaded."""
        key = str(layer)
        if key in self.layers:
            del self.layers[key]

    def loaded_layers(self) -> list[int]:
        """Sorted list of layer indices currently materialized in memory."""
        return sorted(int(k) for k in self.layers.keys())

    def forward(
        self, x: torch.Tensor, layer: int | None = None, **kwargs: object
    ) -> EsmcSaeOutput:
        if layer is None:
            if len(self.layers) == 1:
                ((_only_key, only_layer),) = self.layers.items()
                return only_layer(x, **kwargs)
            if len(self.layers) == 0:
                raise RuntimeError(
                    "No layers loaded - call initialize_layers([...]) first. "
                    f"available_layers={self.config.available_layers}"
                )
            raise RuntimeError(
                "Multiple layers are loaded - please select one via "
                f"forward(x, layer=<idx>). Loaded layers: {self.loaded_layers()}"
            )
        key = str(layer)
        if key not in self.layers:
            raise KeyError(
                f"Layer {layer} is not loaded. Call initialize_layers([{layer}]) "
                f"first. Loaded layers: {self.loaded_layers()}"
            )
        return self.layers[key](x, **kwargs)

    def save_pretrained(self, save_directory: str | os.PathLike) -> None:
        """Write ``config.json`` plus one ``layer_{i}.safetensors`` per loaded
        layer.

        Only layers currently in :attr:`layers` are written.
        ``available_layers`` in the saved config is synced to what's actually
        on disk so a ``release_layer`` + ``save_pretrained`` round-trip never
        advertises a layer whose shard is missing.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config.available_layers = self.loaded_layers()
        self.config.save_pretrained(str(save_directory))
        for key, layer in self.layers.items():
            shard = save_directory / f"layer_{key}.safetensors"
            save_file(
                {
                    k: v.detach().cpu().contiguous()
                    for k, v in layer.state_dict().items()
                },
                str(shard),
            )


def _resolve_snapshot_dir(
    pretrained_model_name_or_path: str | os.PathLike,
    *,
    revision: str | None,
    cache_dir: str | os.PathLike | None,
    token: str | None,
    allow_patterns: list[str] | None,
    local_files_only: bool,
    force_download: bool,
) -> str:
    """Local dir -> return as-is; hub id -> ``snapshot_download`` it.

    A directory only counts as "local" if it actually contains ``config.json``,
    so a stale subdir named like a hub id doesn't accidentally shadow the hub
    fetch.
    """
    path = Path(pretrained_model_name_or_path)
    if path.is_dir() and (path / CONFIG_NAME).exists():
        return str(path)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=str(pretrained_model_name_or_path),
        revision=revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
        token=token,
        allow_patterns=allow_patterns,
        local_files_only=local_files_only,
        force_download=force_download,
    )
