"""Benchmark fixed-shape ESM3 logits, including fresh input copies.

Run from a checkout: PYTHONPATH=. python cookbook/local/esm3_cuda_graph.py
Optional compilation/backends are experiments; the JSON reports numerical drift.
Graph outputs alias reusable buffers. Copy outputs before the next replay if
they must be retained. This example is for inference, not training or generation.
"""

import argparse
import copy
import hashlib
import inspect
import json
import statistics
import time
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.constants.esm3 import data_root
from esm.utils.sampling import _BatchedESMProteinTensor

UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
TRACKS = ("sequence", "structure", "secondary_structure", "sasa", "function")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("esm3_cuda_graph.json"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--length",
        type=int,
        default=len(UBIQUITIN),
        help="Residues; nondefault lengths repeat/truncate ubiquitin as a synthetic performance fixture.",
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--outputs", choices=("all", "sequence", "embeddings"), default="all"
    )
    compile_flags = parser.add_mutually_exclusive_group()
    compile_flags.add_argument("--compile-transformer", action="store_true")
    compile_flags.add_argument("--compile-swiglu", action="store_true")
    compile_flags.add_argument(
        "--compile-ffn",
        action="store_true",
        help="Compile FFNs in both baseline and candidate for a matched quantization comparison.",
    )
    parser.add_argument(
        "--attention", choices=("default", "cudnn", "fa4"), default="default"
    )
    parser.add_argument("--mxfp8-ffn", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.length, args.repeats, args.warmup) < 1:
        parser.error("batch size, length, repeats, and warmup must be positive")
    return args


def snapshot(output):
    values = {name: getattr(output.logits, name) for name in TRACKS}
    values.update(
        residue_annotations=output.residue_annotation_logits,
        embeddings=output.embeddings,
    )
    return {
        name: value.detach().cpu().clone()
        for name, value in values.items()
        if value is not None
    }


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    result = {}
    for name, value in actual.items():
        reference = expected[name]
        error = (value - reference).abs()
        result[name] = {
            "finite": bool(torch.isfinite(value).all()),
            "exact": bool(torch.equal(value, reference)),
            "close_rtol_0.01_atol_0.01": bool(
                torch.isclose(value, reference, rtol=1e-2, atol=1e-2).all()
            ),
            "max_absolute_error": error.max().item(),
            "mean_absolute_error": error.mean().item(),
            "relative_rms_error": (
                error.square().mean().sqrt()
                / reference.square().mean().sqrt().clamp_min(1e-12)
            ).item(),
            "argmax_agreement": (value.argmax(-1) == reference.argmax(-1))
            .float()
            .mean()
            .item(),
        }
    return result


def measure(run, args):
    for index in range(args.warmup):
        run(index % 4)
    torch.cuda.synchronize()
    samples = []
    for index in range(args.repeats):
        started = time.perf_counter_ns()
        run(index % 4)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def capture(run):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    return graph, output


@contextmanager
def attention_backend(name):
    if name == "default":
        yield
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if name == "cudnn":
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            yield
        return
    from torch.nn.attention import (
        activate_flash_attention_impl,
        current_flash_attention_impl,
        restore_flash_attention_impl,
    )

    # Requires a compatible flash-attn-4 installation. These fixtures have no
    # padding mask, attention-output request, or training dropout.
    if current_flash_attention_impl() is not None:
        raise RuntimeError(
            "Run this example with the default flash-attention registry."
        )
    activate_flash_attention_impl("FA4")
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            yield
    finally:
        restore_flash_attention_impl()


def strict_options():
    from torch import _inductor

    if "numerics" not in _inductor.list_options():
        raise RuntimeError(
            "Conservative compilation requires an Inductor build exposing the numerics option, as in PyTorch 2.15."
        )
    return {"numerics": "strict", "triton.cudagraphs": False}


def compile_ffns(model, swiglu=False):
    options = strict_options() if swiglu else {"triton.cudagraphs": False}
    for block in model.transformer.blocks:
        ffn = block.ffn
        assert isinstance(ffn, torch.nn.Sequential)
        if swiglu:
            ffn[2].compile(fullgraph=True, dynamic=False, options=options)
        else:
            ffn.compile(fullgraph=True, dynamic=False, options=options)


def optimize(model, args):
    if args.mxfp8_ffn:
        from torchao.prototype.mx_formats.inference_workflow import (
            MXDynamicActivationMXWeightConfig,
        )
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import KernelPreference

        quantize_(
            model,
            MXDynamicActivationMXWeightConfig(
                activation_dtype=torch.float8_e4m3fn,
                weight_dtype=torch.float8_e4m3fn,
                kernel_preference=KernelPreference.AUTO,
            ),
            filter_fn=lambda module, name: (
                isinstance(module, torch.nn.Linear)
                and name.startswith("transformer.")
                and ".ffn." in name
            ),
        )
    if args.compile_transformer:
        from torch import _inductor

        # Passing both mode and options is invalid. Expand the documented mode
        # into options, then preserve eager BF16 rounding between fused ops.
        options = dict(_inductor.list_mode_options()["max-autotune-no-cudagraphs"])
        options.update(strict_options())
        model.transformer.compile(fullgraph=True, dynamic=False, options=options)
    elif args.compile_swiglu:
        compile_ffns(model, swiglu=True)


@torch.inference_mode()
def main():
    args = arguments()
    args.output = args.output.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    torch.cuda.set_device(0)
    report: dict[str, Any] = {
        "baseline": "compiled_ffn_bf16" if args.compile_ffn else "eager_bf16",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scope": "Equal-length sequence-only inputs; four changing inputs; synchronized wall time includes pinned H2D copy; no output D2H, tokenization, model loading, or graph capture in timings.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        baseline_model = ESM3.from_pretrained(
            "esm3-open", device=torch.device("cuda:0")
        ).eval()
        source = Path(inspect.getfile(ESM3))
        report.update(
            esm_source=str(source),
            esm_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            checkpoint_directory=str(Path(data_root("esm3")).resolve()),
            cudnn=torch.backends.cudnn.version(),
        )
        report["packages"] = {}
        for package in (
            "esm",
            "triton",
            "torchao",
            "flash-attn-4",
            "nvidia-cutlass-dsl",
        ):
            try:
                report["packages"][package] = version(package)
            except PackageNotFoundError:
                report["packages"][package] = None
        all_outputs = args.outputs == "all"
        config = LogitsConfig(
            sequence=all_outputs or args.outputs == "sequence",
            structure=all_outputs,
            secondary_structure=all_outputs,
            sasa=all_outputs,
            function=all_outputs,
            residue_annotations=all_outputs,
            return_embeddings=all_outputs or args.outputs == "embeddings",
        )
        sequence = (UBIQUITIN * ((args.length + len(UBIQUITIN) - 1) // len(UBIQUITIN)))[
            : args.length
        ]
        prompts = [sequence] + [
            "".join(
                "_" if i % 7 == offset else residue
                for i, residue in enumerate(sequence)
            )
            for offset in range(3)
        ]
        encoded: list[torch.Tensor] = []
        for prompt in prompts:
            encoded_sequence = baseline_model.encode(
                ESMProtein(sequence=prompt)
            ).sequence
            assert encoded_sequence is not None
            encoded.append(encoded_sequence)
        tokens = [
            torch.stack(
                [
                    encoded[(row + index) % len(encoded)]
                    for row in range(args.batch_size)
                ]
            )
            for index in range(len(encoded))
        ]
        host_inputs = [value.cpu().pin_memory() for value in tokens]
        static_tokens = tokens[0].clone()
        static = _BatchedESMProteinTensor(sequence=static_tokens)
        report["input_shape"] = list(static_tokens.shape)
        report["input_copy_bytes"] = (
            static_tokens.numel() * static_tokens.element_size()
        )
        # Copy before compiling: compiled callables can retain the original
        # module, so deepcopy of an already compiled model is not a valid control.
        candidate = copy.deepcopy(baseline_model) if args.mxfp8_ffn else baseline_model
        if args.compile_ffn:
            compile_ffns(baseline_model)
            started = time.perf_counter()
            baseline_model.logits(static, config)
            torch.cuda.synchronize()
            report["baseline_first_compiled_call_seconds"] = (
                time.perf_counter() - started
            )
        references = [
            snapshot(
                baseline_model.logits(_BatchedESMProteinTensor(sequence=value), config)
            )
            for value in tokens
        ]
        assert any(
            not torch.equal(references[0][key], references[1][key])
            for key in references[0]
        ), "The changing-input fixture did not change the outputs."

        def eager_baseline(index):
            static_tokens.copy_(host_inputs[index], non_blocking=True)
            return baseline_model.logits(static, config)

        report["eager"] = measure(eager_baseline, args)
        baseline_graph, baseline_output = capture(
            lambda: baseline_model.logits(static, config)
        )

        def baseline_replay(index):
            static_tokens.copy_(host_inputs[index], non_blocking=True)
            baseline_graph.replay()
            return baseline_output

        report["baseline_graph_correctness"] = [
            compare(snapshot(baseline_replay(index)), reference)
            for index, reference in enumerate(references)
        ]
        assert all(
            head["exact"]
            for item in report["baseline_graph_correctness"]
            for head in item.values()
        )
        report["graph_before"] = measure(baseline_replay, args)
        # Quantization replaces parameters. Keep the original model alive for
        # the final baseline replay; this option temporarily holds two models.
        optimize(candidate, args)
        if args.compile_ffn and args.mxfp8_ffn:
            compile_ffns(candidate)
        save()
        with attention_backend(args.attention):
            started = time.perf_counter()
            candidate.logits(static, config)
            torch.cuda.synchronize()
            report["first_candidate_call_seconds"] = time.perf_counter() - started
            candidate_references = [
                snapshot(
                    candidate.logits(_BatchedESMProteinTensor(sequence=value), config)
                )
                for value in tokens
            ]
            report["candidate_vs_baseline"] = [
                compare(actual, reference)
                for actual, reference in zip(
                    candidate_references, references, strict=True
                )
            ]
            graph, output = capture(lambda: candidate.logits(static, config))

            def replay(index):
                static_tokens.copy_(host_inputs[index], non_blocking=True)
                graph.replay()
                return output

            report["graph_vs_candidate_eager"] = [
                compare(snapshot(replay(index)), reference)
                for index, reference in enumerate(candidate_references)
            ]
            assert all(
                head["exact"]
                for item in report["graph_vs_candidate_eager"]
                for head in item.values()
            ), "Replay did not match its own eager candidate."
            report["candidate_graph"] = measure(replay, args)
        report["graph_after"] = measure(baseline_replay, args)
        baseline_ms = statistics.mean(
            report[key]["median_ms"] for key in ("graph_before", "graph_after")
        )
        candidate_ms = report["candidate_graph"]["median_ms"]
        report["gain_vs_graph"] = baseline_ms / candidate_ms
        report["gain_vs_eager"] = report["eager"]["median_ms"] / candidate_ms
        report["first_call_amortization_calls"] = (
            report["first_candidate_call_seconds"] * 1000 / (baseline_ms - candidate_ms)
            if (args.compile_transformer or args.compile_swiglu)
            and baseline_ms > candidate_ms
            else None
        )
        compatible = all(
            head["finite"] and head["close_rtol_0.01_atol_0.01"]
            for item in report["candidate_vs_baseline"]
            for head in item.values()
        )
        bitwise_equal = all(
            head["exact"]
            for item in report["candidate_vs_baseline"]
            for head in item.values()
        )
        experimental = (
            args.compile_transformer
            or args.compile_ffn
            or args.mxfp8_ffn
            or args.attention != "default"
        )
        compatible = compatible and (experimental or bitwise_equal)
        report["bitwise_equivalent"] = bitwise_equal
        report["validation"] = (
            "exact"
            if bitwise_equal
            else "within_declared_tolerance"
            if compatible
            else "outside_declared_tolerance"
        )
        report["status"] = "passed" if compatible else "numerical_mismatch"
        save()
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "validation": report["validation"],
                    "latency_ms": {
                        key: report[key]["median_ms"]
                        for key in (
                            "eager",
                            "graph_before",
                            "candidate_graph",
                            "graph_after",
                        )
                    },
                    "gain_vs_graph": report["gain_vs_graph"],
                    "gain_vs_eager": report["gain_vs_eager"],
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        if not compatible:
            raise SystemExit(
                "Candidate changed outputs outside the declared tolerance; inspect the JSON before adopting it. These fixtures do not establish model quality."
            )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
