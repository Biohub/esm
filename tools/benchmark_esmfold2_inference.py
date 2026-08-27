#!/usr/bin/env python3
"""Measure ESMFold2 inference peak VRAM and wall time for one low-memory mode.

Run one variant per fresh process so CUDA allocator state and model placement do
not leak across measurements. The script intentionally performs no MSA search;
callers that supply precomputed MSA features should prepare them once and reuse
the same feature artifact for every variant. One untimed warm-up is used by
default; pass ``--warmup 0`` when cold-start latency is the quantity of interest.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from esm.models.esmc import kernels as esmc_kernels
from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model
from esm.models.esmfold2 import layers as esmfold2_layers
from esm.models.esmfold2.output import build_molecular_complex_from_features
from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput

VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "esmc_offload": {"offload_esmc_after_lm": True},
    "confidence_chunk_1": {"confidence_sample_chunk_size": 1},
    "combined": {"low_memory_mode": True},
}


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, list[str]]] = []
    name: str | None = None
    sequence: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, sequence))
            name = line[1:].strip() or f"chain_{len(records) + 1}"
            sequence = []
        else:
            if name is None:
                raise ValueError("FASTA sequence appears before its header")
            sequence.append(line.upper())
    if name is not None:
        records.append((name, sequence))
    parsed = [(name, "".join(chunks)) for name, chunks in records]
    if not parsed or any(not sequence for _, sequence in parsed):
        raise ValueError(f"no non-empty FASTA records found in {path}")
    return parsed


def synchronize() -> None:
    torch.cuda.synchronize()


def output_signature(
    output: dict[str, torch.Tensor], asym_id: torch.Tensor
) -> dict[str, float]:
    """Small parity and confidence signature for the benchmark result JSON."""
    signature: dict[str, float] = {}
    for name in ("complex_plddt", "complex_iplddt", "ptm", "iptm"):
        value = output.get(name)
        if value is not None:
            signature[f"{name}_mean"] = float(value.float().mean().cpu())
    plddt = output["plddt"][0].float()
    signature.update(
        plddt_sample0_mean=float(plddt.mean().cpu()),
        plddt_sample0_min=float(plddt.min().cpu()),
        plddt_sample0_median=float(plddt.median().cpu()),
        plddt_sample0_max=float(plddt.max().cpu()),
    )
    for chain_id in asym_id.unique(sorted=True):
        chain_plddt = plddt[asym_id == chain_id]
        signature[f"plddt_sample0_chain_{int(chain_id)}_mean"] = float(
            chain_plddt.mean().cpu()
        )
    signature["coords_l2"] = float(
        output["sample_atom_coords"].float().square().sum().sqrt().cpu()
    )
    signature["distogram_sum"] = float(output["distogram_logits"].float().sum().cpu())
    return signature


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a peak-VRAM benchmark")
    backend_availability = {
        "esmc_xformers": esmc_kernels.XFORMERS_INSTALLED,
        "esmc_flash_attention": esmc_kernels.FLASH_ATTN_INSTALLED,
        "esmc_flash_rotary": esmc_kernels.FLASH_ATTN_ROTARY_INSTALLED,
        "esmfold2_flash_attention": esmfold2_layers.FLASH_ATTN_AVAILABLE,
    }
    print(
        "ESMFold2 benchmark backend availability: "
        + json.dumps(backend_availability, sort_keys=True),
        flush=True,
    )
    records = parse_fasta(args.fasta)
    structure_input = StructurePredictionInput(
        sequences=[
            ProteinInput(id=name.split()[0], sequence=sequence)
            for name, sequence in records
        ]
    )

    model = (
        EsmFold2Model.from_pretrained(args.model, esmc_precision=args.esmc_precision)
        .cuda()
        .eval()
    )
    model.set_chunk_size(None if args.pair_chunk_size == 0 else args.pair_chunk_size)
    builder = ESMFold2InputBuilder()
    features, chain_infos = builder.prepare_input(
        structure_input, seed=args.seed, device=model.device
    )
    forward_kwargs = {
        "num_loops": args.num_loops,
        "num_sampling_steps": args.num_sampling_steps,
        "num_diffusion_samples": args.num_diffusion_samples,
        **VARIANTS[args.variant],
    }

    for warmup_index in range(args.warmup):
        torch.manual_seed(args.seed + warmup_index)
        with torch.inference_mode():
            model(**features, **forward_kwargs)
        synchronize()

    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    synchronize()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(**features, **forward_kwargs)
    synchronize()
    runtime_seconds = time.perf_counter() - start

    if args.structure_output is not None:
        args.structure_output.parent.mkdir(parents=True, exist_ok=True)
        complex_sample = build_molecular_complex_from_features(
            coords=output["sample_atom_coords"][0],
            plddt=output["plddt"][0],
            atom_mask=features["atom_attention_mask"][0],
            ref_element=features["ref_element"][0],
            ref_atom_name_chars=features["ref_atom_name_chars"][0],
            chain_infos=chain_infos,
            complex_id=f"{args.fasta.stem}_sample_0",
        )
        args.structure_output.write_text(complex_sample.to_mmcif())

    device = torch.cuda.current_device()
    return {
        "variant": args.variant,
        "model": args.model,
        "esmc_precision": args.esmc_precision,
        "chain_count": len(records),
        "chain_lengths": [len(sequence) for _, sequence in records],
        "total_tokens": sum(len(sequence) for _, sequence in records),
        "num_loops": args.num_loops,
        "num_sampling_steps": args.num_sampling_steps,
        "num_diffusion_samples": args.num_diffusion_samples,
        "pair_chunk_size": None if args.pair_chunk_size == 0 else args.pair_chunk_size,
        "warmup": args.warmup,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "backend_availability": backend_availability,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocated_after_bytes": torch.cuda.memory_allocated(),
        "runtime_seconds": runtime_seconds,
        "output_signature": output_signature(output, features["asym_id"][0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path)
    parser.add_argument("--model", default="biohub/ESMFold2")
    parser.add_argument("--variant", choices=VARIANTS, default="default")
    parser.add_argument("--esmc-precision", default="bf16")
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--num-diffusion-samples", type=int, default=4)
    parser.add_argument("--pair-chunk-size", type=int, default=64)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="number of untimed warm-up iterations (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--structure-output",
        type=Path,
        help=(
            "optional sample-0 mmCIF output; per-residue pLDDT is stored in "
            "the B-factor column"
        ),
    )
    args = parser.parse_args()

    result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
