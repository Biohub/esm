# Fixed-shape ESM3 inference on CUDA

`esm3_cuda_graph.py` benchmarks the public `ESM3.logits()` API with changing,
equal-shaped inputs. Applications own the static input and output buffers, warm
the model before capture, and copy each new input into the captured buffers.
Graph outputs alias reusable storage: consume or clone them before the next
replay. A different shape, set of input tracks, output configuration, or model
configuration requires a separately warmed and captured graph.

The model keeps its usual eager behavior. CUDA Graphs, compilation, alternative
attention, and quantization are explicit application choices. `logits()` already
sets up BF16 autocast on CUDA. Direct `forward()` calls with the open BF16 weights
still require the appropriate autocast context.

## Running the benchmark

Use a clean environment and install the candidate checkout, as described in
[`CONTRIBUTIONS.md`](../../CONTRIBUTIONS.md). From the repository root:

```bash
python -m pip install -e .
PYTHONPATH=. python cookbook/local/esm3_cuda_graph.py --output graph.json
PYTHONPATH=. python cookbook/local/esm3_cuda_graph.py --outputs sequence --output sequence.json
PYTHONPATH=. python cookbook/local/esm3_cuda_graph.py --batch-size 8 --output batch8.json
```

For the versioned B200 experiment, install PyTorch 2.15.0 from its CUDA 13.2
index before installing the checkout:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e .
```

The script reports the resolved checkpoint directory and source hash. To pin the
checkpoint to the revision used here, use the loader's local-data mode:

```bash
esm_source_dir="$PWD"
esm_checkpoint_dir=$(hf download biohub/esm3-sm-open-v1 --revision 47f0545b2b6daf26a93439a3cd610f4f7f3d5478)
cd "$esm_checkpoint_dir"
INFRA_PROVIDER=local PYTHONPATH="$esm_source_dir" python "$esm_source_dir/cookbook/local/esm3_cuda_graph.py" --output /tmp/esm3-graph.json
cd "$esm_source_dir"
```

The optional experiments below are deliberately separate. A numerical mismatch
writes the measurements and per-head error statistics to JSON, then exits
nonzero. A successful capture alone is not a successful numerical comparison.

```bash
# Compiler experiments; neither is a default library setting.
PYTHONPATH=. python cookbook/local/esm3_cuda_graph.py --compile-swiglu --output swiglu.json
PYTHONPATH=. python cookbook/local/esm3_cuda_graph.py --compile-transformer --output compiled.json

# Select an existing SDPA implementation; FA4 is a separate optional package.
python -m pip install 'flash-attn-4[cu13]==4.0.0b39'
PYTHONPATH=. python cookbook/local/esm3_cuda_graph.py --attention cudnn --output cudnn.json
PYTHONPATH=. python cookbook/local/esm3_cuda_graph.py --attention fa4 --output fa4.json
PYTHONPATH=. python cookbook/local/esm3_cuda_graph.py --attention fa4 --length 1024 --output fa4-long.json

# Inference-only FFN quantization, including a matched compiled-FFN comparison.
python -m pip install torchao==0.18.0
PYTHONPATH=. python cookbook/local/esm3_cuda_graph.py --mxfp8-ffn --batch-size 16 --output mx.json
PYTHONPATH=. python cookbook/local/esm3_cuda_graph.py --mxfp8-ffn --compile-ffn --batch-size 16 --output mx-compiled.json
```

The `--compile-ffn` comparison compiles both the BF16 control and the MXFP8
candidate. Its reference is compiled BF16; it does not establish equivalence of
that reference to the original eager model. Optional packages do not become ESM
dependencies, and their compatibility was checked only on the recorded stack.

## Measurement scope

The measurements below used one NVIDIA B200, driver 595.112.07, PyTorch
2.15.0+cu132, and ESM3 Open checkpoint revision
`47f0545b2b6daf26a93439a3cd610f4f7f3d5478`. The primary fixture is 76-residue
ubiquitin, encoded to 78 tokens. Four same-shaped inputs change the sequence
mask; all six heads and embeddings are requested unless specified otherwise.

Latency includes a fresh pinned-host input copy, graph replay, and GPU completion.
It excludes tokenization, host allocation/pinning, compilation, capture, and
output copies to CPU. Each request is synchronized individually. Throughput at
larger batch sizes is a separate measurement. These are inference measurements;
they do not validate training, model quality, or a B200-versus-Hopper comparison.

Controls run before and after each candidate. Final attention/quantization runs
used 20 warmups and 100 samples per path, with profiling after all timed controls.
Profiling between timing phases perturbed later timings in preliminary runs;
those measurements are excluded from the comparisons below.

## Graph capture and selective outputs

The capture fix replaces the CPU-backed scalar construction in `logits()` with
`torch.ones((), device=input.device)`. On the original full-model validation,
stock eager controls took 32.33–32.91 ms, while graph replay including the
624-byte pinned-host input copy took 6.269 ms. All returned tensors matched eager
execution exactly across changing inputs. The input-copy increment was about
7.6 microseconds, approximately 0.12% of graph-call latency.

`LogitsConfig` now also avoids FP32 conversions for outputs the caller did not
request. Sequence-only and embeddings-only calls remove six GPU conversion
copies. The raw `forward()` and output-head hooks still receive complete
`ESMOutput` tensors; this change does not skip head computation. Full-model
outputs remained bitwise identical for batch 1 and 8 ubiquitin and a synthetic
512-residue fixture. The measured graph-latency difference was only about
0–0.2%, and whole-call peak allocation did not decrease in the batch-1 ubiquitin
measurement. Do not interpret this
small copy reduction as a substantial Blackwell speedup.

## Work inside the graph

Nsight Systems 2026.1.1, using `--cuda-graph-trace=node`, recorded 2,386 kernels
per replay for the all-output batch-1 fixture. The following categories are
heuristics from kernel names and sum GPU kernel durations; they are not
uninstrumented wall-time measurements.

| Kernel category | Share of summed kernel time |
|---|---:|
| Matrix multiplication | 22.0% |
| Attention | 3.7% |
| Layer normalization | 14.4% |
| Dtype conversion | 16.8% |
| Copies and concatenation | 14.9% |
| Other elementwise operations | 28.2% |

This profile motivates fusing work inside the graph. Graph replay removes CPU
launch overhead but does not fuse the captured kernels. Attention alone has
limited scope to improve this short-sequence workload. Nsight Compute on one
representative SM100 GEMM also showed less than one wave per SM; that observation
is specific to the sampled kernel, rather than a whole-model utilization score.

## Compiler fusion: accepted and rejected configurations

Compiling only each block's SwiGLU activation with
`options={"numerics": "strict", "triton.cudagraphs": False}` reduced batch-1
graph latency from 6.251/6.251 ms controls to 6.024 ms, a 3.6% reduction. Every
returned tensor was bitwise identical to the original BF16 model across all
four inputs, and repeated graph replay matched the candidate's eager outputs.
This leaves normalization, attention, and GEMMs unchanged. The generated kernel
preserved the BF16 rounding between SiLU and multiplication rather than keeping
the intermediate in higher precision. The option is exposed as
`--compile-swiglu`; its first-call compilation cost is recorded separately.
These results describe the tested fixtures, not universal numerical equivalence.

The final public recipe independently reproduced the result and checked two
additional shapes, all with bitwise-identical outputs:

| Workload | Graph control before / after | SwiGLU fusion | Lower latency |
|---|---:|---:|---:|
| Batch 1, 76 residues | 6.290 / 6.290 ms | 6.061 ms | 3.6% |
| Batch 8, 76 residues | 9.231 / 9.231 ms | 8.708 ms | 5.7% |
| Batch 1, 512 synthetic residues | 11.085 / 11.085 ms | 10.629 ms | 4.1% |

The first compiled call took approximately 2.3 seconds per shape in this run.
Using the measured incremental savings, that corresponds to roughly 10,000,
4,400, and 5,100 calls to amortize, respectively. Each batch-8 call processes
eight proteins. First-call costs depend on the compiler cache and environment.

Nsight Systems verified 2,386 to 2,338 kernels per public-API replay: 48 native
SiLU kernels and 48 native multiply kernels became 48 fused kernels, with all
other kernel counts unchanged.

Wider compilation was faster but failed the predefined output comparison:

| Compiled region/settings | Graph latency | Comparison to original BF16 |
|---|---:|---|
| Transformer, max autotune | 2.489 ms | Failed `rtol=atol=1e-2` |
| Transformer, strict numerics | 2.566 ms | Failed the same gate |
| Transformer, strict + ATen GEMMs + cuDNN | 2.395 ms | Failed the same gate |
| FFNs only, strict numerics | 5.693 ms | Failed the same gate |
| SwiGLU only, strict numerics | 6.024 ms | Bitwise identical on all fixtures |

The original graph controls for these comparisons were about 6.25–6.44 ms.
The faster rejected configurations are not enabled by default or presented as
equivalent replacements. Compiled graph replay matched each configuration's own
eager outputs: the discrepancy arose from compilation, not graph capture.

Nsight confirmed what the experimental whole-transformer compilation removed:
the matched raw-forward wrapper went from 2,385 to 628 kernels per replay;
conversion kernels fell from 727 to 40, and copy/concatenation kernels from 392
to 15. Matmul and attention counts remained 213 and 48. This explains the
potential fusion gain without treating failed numerical checks as successes.
The wrapper preallocated the confidence scalar, explaining its one-kernel
difference from the public API's 2,386-kernel trace.

Compiling the complete raw forward with `fullgraph=True` also encountered the
tokenizer's Rust `token_to_id` lookup. Compiling numerical submodules avoids
that boundary without caching mutable tokenizer state or bypassing the public
model's hooks.

## Attention and MXFP8 experiments

The default PyTorch dispatcher already selected cuDNN attention on this stack.
The FA4 experiment used `flash-attn-4[cu13]==4.0.0b39` through PyTorch's SDPA
registry, preserving ESM3's Q/K normalization and rotary implementation. It
verified the actual SM100 CuTe kernel. The experiment used equal-length inputs,
no arbitrary attention mask, and no requested attention weights.

| Candidate | Batch | Graph control before / after | Candidate |
|---|---:|---:|---:|
| Forced cuDNN | 1 | 6.387 / 6.386 ms | 6.389 ms |
| FA4 | 1 | 6.387 / 6.388 ms | 6.668 ms |
| FA4 | 16 | 13.380 / 13.380 ms | 13.590 ms |
| MXFP8 FFNs, without compilation | 1 | 6.250 / 6.251 ms | 11.362 ms |
| MXFP8 FFNs, without compilation | 16 | 13.316 / 13.310 ms | 22.058 ms |
| MXFP8 FFNs, both paths compiled | 1 | 5.679 / 5.679 ms | 6.083 ms |
| MXFP8 FFNs, both paths compiled | 16 | 11.947 / 11.934 ms | 11.951 ms |

FA4 also took 22.654 ms versus 22.340/22.336 ms controls on a batch-1 synthetic
fixture made by repeating ubiquitin to 1,024 residues (1,026 encoded tokens).
This longer input checks a different computational regime, not protein quality.

MXFP8 used TorchAO 0.18.0 and transformed only the 96 transformer FFN linear
layers. The observed scaled-matmul kernels confirm that quantization was active.
Activation quantization added enough work to erase the faster matmul opportunity
here. Compiling both the BF16 and MXFP8 FFNs removed much of that overhead, but
MXFP8 still did not improve these workloads. The compiled BF16 control is a
separate configuration, not proof of equivalence to the original eager model.

Alternative attention and quantization changed numerical outputs. Each captured
configuration matched its own eager execution, but that does not establish
equivalence to the original BF16 model or downstream biological quality. For
example, the compiled-MXFP8 batch-16 experiment had a minimum sequence argmax
agreement of 99.68% and structure argmax agreement of 86.86% across the four
inputs. These experiments are not enabled by default and are not recommendations
to deploy quantized ESM3.

## Sources

- [PyTorch 2.15 compilation options](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)
- [PyTorch 2.15 FA4 registration](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/nn/attention/_fa4.py)
- [FA4 implementation and installation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/README.md)
- [TorchAO MXFP8 inference workflow](https://github.com/pytorch/ao/blob/v0.18.0/torchao/prototype/mx_formats/inference_workflow.py)
- [Blackwell quantization study and shape constraints](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/)
- [Nsight Systems profiling guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
