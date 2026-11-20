# LightningMoBA

[Paper](https://arxiv.org/abs/2611.26268) | [Main Results](#main-results) | [Installation](#2-installation) | [Usage](#usage) | [Benchmarks](#routing-sweep-table-4-protocol) | [Citation](#citation)

LightningMoBA co-designs dynamic routing, KV-centric execution, and runtime pipelining for long-context sparse attention. It replaces fixed per-query Top-K retention with threshold-based selection, groups selected queries by KV block with fanout-aware work partitioning, and overlaps data movement with Tensor Core computation on SM90 GPUs.

![LightningMoBA overview: dynamic threshold routing, KV-centric execution, and runtime pipelining](assets/lightning_moba_overview.png)

*Figure 1 from the paper. Dynamic routing produces variable-cardinality sparse metadata; KV-centric execution splits high-fanout columns into bounded work items; runtime pipelining reuses resident KV tiles while overlapping query gathers and attention computation.*

## Main Results

LightningMoBA is evaluated against FlashMoBA at matched average retention budgets. The main evaluation uses the full 6,500-example RULER suite, across six context lengths (4K–128K), with TP=8 and batch size 1.

### End-to-end inference

At the Top-16 reference budget, LightningMoBA achieves 1.20–1.60× end-to-end speedup across five checkpoints, while the overall mean RULER score increases from 74.42 to 75.28.

| Model | RULER score (FlashMoBA → Ours) | End-to-end speedup |
|---|---:|---:|
| Qwen3-4B | 72.43 → 75.53 | 1.53× |
| GQA-30B-A3B | 71.87 → 71.01 | 1.58× |
| Qwen3-235B-A22B | 82.18 → 83.08 | 1.60× |
| Llama-3.1-8B | 73.99 → 75.47 | 1.42× |
| Qwen3-32B-YaRN4 | 71.65 → 71.29 | 1.20× |

Scores and end-to-end speedups are computed from the six-length averages for each model. End-to-end timing measures the complete RULER generation loop, not single-request latency; both methods use CUDA Graph dense decoding. GQA-30B-A3B is an internal checkpoint.

### Kernel and routing efficiency

- 2.16–2.56× faster complete attention device path at Top-32, including routing, sparse-index/worklist construction, attention, and output merging.
- 6.99–11.25× faster routing + CSC construction at Top-64, measured separately from attention.

These are controlled GPU measurements on captured Qwen3-4B and GQA-30B-A3B tensors, not end-to-end model speedups. Full per-length results, ablations, and additional comparisons are reported in the paper.

The package also provides standard Top-K sparse attention and a fine-grained Block-8 interface; see [Usage](#usage) for supported execution paths and constraints.

## Setup

### 1. Environment

**Tested hardware:** NVIDIA SM90 GPUs.

Build from source in a Linux environment with:

- An NVIDIA GPU and a CUDA-enabled PyTorch installation.
- A CUDA Toolkit compatible with the installed PyTorch build, including `nvcc`, and a C++17 compiler.
- Python 3.9 or newer, using a Python version supported by your PyTorch and Triton releases.

The default KV-centric pipeline requires an SM90 GPU and CUDA 12.3 or newer, compatible with your PyTorch installation. Both the default pipeline and the block8 CUDA backend require BF16 inputs with head dimension 128. Other entry points have different constraints; see [Usage](#usage).

Check the active environment before building:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
nvcc --version
```

### 2. Installation

Clone the repository:

```bash
git clone https://github.com/Tencent/LightningMoBA.git
cd LightningMoBA
```

With CUDA-enabled PyTorch installed in your active environment, install the build dependencies and compile the package from the repository root:

```bash
python -m pip install setuptools wheel ninja packaging psutil
MAX_JOBS=2 python -m pip install --no-build-isolation -e '.[test]'
```

The `[test]` extra includes the test dependencies. Use the same Python environment for installation, examples, and tests.

The package is imported as `lightning_moba`; its compiled extension is `lightning_moba_cuda`. The shared extension includes block8 CUDA attention, so there is no separate block8 library to compile at runtime.

`MAX_JOBS` limits build parallelism. The default CUDA targets are `80;90a`; `LIGHTNING_MOBA_CUDA_ARCHS` overrides this list and `NVCC_THREADS` controls compiler threads. For an SM90-only build, set `LIGHTNING_MOBA_CUDA_ARCHS='90a'` before installation. Rebuild the extension when changing PyTorch, CUDA, or the C++ ABI.

Check the installed modules:

```bash
python -c "import lightning_moba, lightning_moba_cuda; print(lightning_moba.__version__); print(lightning_moba.__file__); print(lightning_moba_cuda.__file__)"
```

## Usage

**Choose the entry point for your task:**

- **Paper's main method:** start with [Dynamic threshold routing (block 128)](#2-dynamic-threshold-routing-block-128). `lightning_moba_varlen_func` uses scalar `alpha` and the SM90 KV-centric pipeline.
- **Fixed-budget comparison or backward support:** use the [Fixed Top-K baseline path](#1-fixed-top-k-baseline-path), exposed as `lightning_moba_topk_varlen_func` with `moba_topk`.
- **Fine-grained selection:** use the separate [block8 interface](#3-dynamic-threshold-block8-attention); its settings and performance are not interchangeable with the paper's block128 experiments.

Inputs use token-major layout: Q has shape `[total_q, query_heads, head_dim]`, and K/V have shape `[total_k, kv_heads, head_dim]`. Q/K/V must use compatible floating-point dtypes on the same CUDA device, and `query_heads` must be divisible by `kv_heads`. Packed variable-length inputs use contiguous CUDA `int32` cumulative-length tensors.

### 1. Fixed Top-K baseline path

`lightning_moba_topk_varlen_func` preserves the fixed Top-K selector and query-centric attention, including backward support. This is the **baseline**, not the paper's dynamic routing and KV-centric pipeline. Its query grouping still depends on the Top-K budget.

```python
import torch
from lightning_moba import lightning_moba_topk_varlen_func

seq_len = 2048
q = torch.randn(seq_len, 4, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(seq_len, 1, 128, device="cuda", dtype=q.dtype)
v = torch.randn_like(k)
cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

out = lightning_moba_topk_varlen_func(
    q, k, v, cu, cu, seq_len, seq_len,
    moba_chunk_size=128, moba_topk=4, causal=True,
)
print(out.shape)
```

### 2. Dynamic threshold routing (block 128)

`lightning_moba_varlen_func` remains the **default method entry point**, requiring keyword argument `alpha` instead of `moba_topk`. It performs pooling → max-relative threshold selection → fused CSC/Q-CSR construction → SM90 KV-centric producer/consumer pipeline → output merging. The following equivalent, explicit sequence exposes each stage through the public package API:

```python
import torch
from lightning_moba import (
    flash_dynamic_mean_pool,
    flash_cuda_dynamic_relative_threshold_compact_varlen_func,
    flash_cuda_compact_csc_qcsr_from_bitmask,
    lightning_moba_cuda_kv_centric_func,
)

seq_len = 512
q = torch.randn(seq_len, 4, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(seq_len, 1, 128, device="cuda", dtype=q.dtype)
v = torch.randn_like(k)
cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

km, cu_km, _ = flash_dynamic_mean_pool(k, cu, seq_len, 128)
pattern = flash_cuda_dynamic_relative_threshold_compact_varlen_func(
    q, km, cu, cu, cu_km, seq_len, seq_len, alpha=0.9, chunk_size=128,
    sink=1, window=2, causal=True, min_blocks_per_row=0, build_csc=False,
)
pattern, q_offsets, q_csc = flash_cuda_compact_csc_qcsr_from_bitmask(
    pattern, cu, cu_km, seq_len, seq_len, 128,
)
out = lightning_moba_cuda_kv_centric_func(
    q, k, v, cu, cu, seq_len, seq_len,
    pattern.col_offsets, pattern.col_nnz, pattern.csc_indices,
    chunk_size=128, backend="sm90", q_offsets=q_offsets, q_csc=q_csc,
)
print(out.shape)
print("Mean retained blocks:", pattern.row_nnz.float().mean().item())
```

The last call runs both KV-centric attention and variable-NNZ output merging. `backend="sm90"` explicitly selects the pipeline, bypassing the legacy wrapper's streaming dispatch. Do not pass `topk` or fixed-width `q2col` metadata for a dynamic pattern: row cardinality can exceed any nominal Top-K reference budget. The Q-CSR tensors must come from the same pattern and inputs.

For the combined call, use `lightning_moba_varlen_func(q, k, v, cu, cu, seq_len, seq_len, alpha=0.9)`. Neither this entry point nor the explicit `backend="sm90"` path requires a `LIGHTNING_MOBA_KVCENTRIC_SM90_K1` environment switch. Unsupported inputs or a missing pipeline extension cause an error, not a baseline fallback.

Both paths require SM90, BF16, head dimension 128, and 128-token blocks. They are causal and inference-only, with Q/K/V `requires_grad=False`. `alpha` is a finite scalar in `(0, 1]`; non-anchor blocks satisfy `scaled_score >= scaled_row_max + log(alpha)`. `sink=1` and `window=2` count **128-token blocks**. The example threshold is illustrative, not calibrated to Top-16. No fixed per-row budget or minimum-budget padding is imposed.

For packed sequences, provide matching cumulative lengths and align each nonempty Q sequence with the corresponding K suffix. `build_csc=False` returns selection metadata only, so it must be followed by CSC construction before attention. Valid CSC entries are delimited by `col_offsets` and `col_nnz`; `row_nnz` counts blocks, not tokens. Compact CSC construction synchronizes with the host, making the complete pipeline eager-only. Partial-output storage scales with selected pairs rather than a fixed memory limit.

Run the complete prefill example with an independent numerical reference:

```bash
python examples/selector_attention.py --seq-len 512 --alpha 0.9 --check
```

See [selector_attention.py](examples/selector_attention.py) for the executable version. The fine-grained block8 interface below is a separate path, not the block128 main experiment.

### 3. Dynamic-threshold block8 attention

`lightning_moba_block8_func` selects 8-token key blocks using a scalar relative threshold, then computes causal attention over the selected tokens. It is an inference-only interface and requires Q/K/V with `requires_grad=False`.

The following example uses the SM90 CUDA backend:

```python
import torch
from lightning_moba import lightning_moba_block8_func

seq_len = 512
q = torch.randn(seq_len, 4, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(seq_len, 1, 128, device="cuda", dtype=q.dtype)
v = torch.randn_like(k)

out, counts = lightning_moba_block8_func(
    q, k, v,
    alpha=0.9,
    backend="cuda",
    tail_backend="sdpa",
    return_counts=True,
)
print(out.shape)
print("Mean visible tokens:", counts.float().mean().item())
```

The selection and execution rules are:

- `alpha` is one finite scalar in `(0, 1]`, shared across heads within a call. Non-anchor blocks are selected by comparing their pooled scores against the eligible row maximum plus `log(alpha)`. Larger values make selection more restrictive; `alpha=1` is not dense attention, and `alpha=0` is invalid.
- `sink` and `window` count logical **8-token blocks**, not execution tiles. Defaults are `sink=1` and `window=2`, retaining the first block and the current/preceding blocks, subject to the final causal mask.
- There is no fixed token budget or per-row cap. Retained tokens vary with the input. `counts` reports visible token counts with shape `[query_tokens, query_heads]`, including dense-tail tokens.
- Q positions are interpreted as a suffix of the K sequence. Queries beyond the 256-token-aligned K prefix use dense causal suffix attention over the full K/V sequence. Inputs shorter than 256 tokens use this dense path throughout.
- `backend="cuda"` requires SM90, BF16, and head dimension 128. `backend="triton"` accepts FP16/BF16 and head dimensions 32, 64, 128, or 256, with the default sink/window settings.
- `tail_backend="sdpa"` uses PyTorch SDPA and does not require vLLM. The default `"auto"` can use vLLM's FA3 implementation on SM90 when available; `"flash"` explicitly requires that optional backend.

For packed inputs, use `lightning_moba_block8_varlen_func`. It dispatches sequences independently and synchronizes cumulative-length metadata to the CPU; it is not a fused multi-sequence kernel or a CUDA-graph-safe interface.

See the [block8 example](examples/block8_attention.py) and [block8 interface](lightning_moba/block8/interface.py) for more details.

### 4. Public routing and lower-level interfaces

The package exposes both the main method's building blocks and explicit baseline interfaces:

| Entry point | Purpose |
|---|---|
| `lightning_moba_topk_varlen_func` | Explicit Top-K/query-centric baseline, including backward support. |
| `flash_topk_varlen_func` | Build a Top-K CSC pattern; returns column offsets, column counts, and row indices by default. |
| `lightning_moba_attn_varlen_func` | Run query-centric baseline attention over a supplied CSC pattern, with autograd support. |
| `flash_dynamic_mean_pool` | Pool K into 128-token blocks for the BF16/D128 routing path. |
| `flash_cuda_dynamic_relative_threshold_compact_varlen_func` | Max-relative scalar-alpha routing with selection-only or compact CSC output. |
| `flash_cuda_compact_csc_qcsr_from_bitmask` | Materialize compact CSC and variable-length reverse Q-CSR together. |
| `flash_cuda_dynamic_relative_threshold_padded_varlen_func` | Full relative routing with fixed-allocation CSC for CUDA Graph measurements. |
| `lightning_moba_cuda_kv_centric_func` | `backend="sm90"`: variable-cardinality CSC/Q-CSR to pipeline and merge; `backend="legacy"`: explicit Top-K-capacity wrapper. |
| `lightning_moba_streaming_kv_centric_func` | Explicit streaming prefill with bounded partial-output storage; returns output and head-major LSE. |

When routing and baseline attention are called separately, keep their block sizes consistent: `lg_block_n` must match the routing `moba_chunk_size`. The routing function takes `moba_topk` before `moba_chunk_size`, unlike the combined Top-K baseline.

The lower-level KV-centric interfaces are inference-only. Eligible calls to the legacy `lightning_moba_cuda_kv_centric_func` use streaming by default; `LIGHTNING_MOBA_STREAM_CHUNK_EDGES` controls this dispatch and defaults to `2000000`. These switches do not affect the new `lightning_moba_varlen_func`. The explicit streaming backend is eager-only and requires a positive capacity; it is a separate non-pipelined implementation, not the default SM90 pipeline. Its CSC and query-index metadata still scale with the total number of selected associations.

See the [standard interfaces](lightning_moba/lightning_moba_interface.py), [KV-centric implementation](lightning_moba/triton_kv_centric.py), and [streaming implementation](lightning_moba/streaming_kv_centric.py) for signatures and input constraints.

## Tests and Benchmarks

Install the test extra as shown above. On an SM90 GPU, run the default-pipeline, public-interface, block8, and Top-K regression suites with:

```bash
make test PYTHON=python
```

The default-pipeline tests verify direct SM90 dispatch and compare outputs against an independent threshold-attention reference, including packed batches, suffix queries, GQA, high-fanout columns, and more than 64 retained blocks. Run just this new GPU regression with `make test-default PYTHON=python`.

`make test` is not the entire test suite. `make test-cpu PYTHON=python` runs non-CUDA public-interface and dispatch-contract checks; the public-interface tests still require an importable package and compiled extension. Mock dispatch checks alone do not establish GPU numerical correctness.

Additional GPU suites cover standard attention forward/backward and streaming KV-centric inference:

```bash
python -m pytest -q tests/test_moba_attn.py
python -m pytest -q tests/test_streaming_kvcentric.py
```

Hardware-specific cases and optional integrations have their own prerequisites. The optional external backend checks are not part of a standalone package installation.

### Routing sweep (Table 4 protocol)

[benchmarks/bench_routing_sweep.py](benchmarks/bench_routing_sweep.py) compares fixed Top-K routing and block128 max-relative threshold routing at `k = 16, 32, 64`. It calibrates alpha independently for each input/length/budget against the **actual valid CSC pair count**, targeting at most **0.05%** relative mismatch. Calibration, mean pooling, metadata validation, and attention are outside the timed region. Each result reports the median of 50 GPU measurements after 10 warmups, plus alpha, counts, matching error, input identity, hardware, and timing scope.

Two timing modes are deliberately separate:

- `--mode graph-native`: CUDA Graph replay of native Top-K CSC construction versus the padded relative selector. Both include scoring, selection, and CSC. This measures the unsorted CSC-builder scope consumed by KV-centric execution; it **does not include the additional segmented sort in the public Top-K/query-centric interface**. The relative side uses fixed-allocation storage rather than the compact default attention path.
- `--mode eager-public`: CUDA-event timing of the public Top-K router including segmented sort, versus the complete compact relative router. This measures the current public router paths, **not the paper's CUDA Graph timing**. Do not mix these modes when comparing speedups.

Start with a small synthetic smoke run; these outputs are not paper results:

```bash
python benchmarks/bench_routing_sweep.py \
    --lengths 4096 --topks 16 32 64 --mode graph-native \
    --output routing_smoke.json
```

For captured inputs, provide a trusted local `torch.save` dictionary containing BF16 `q` and `k`, loaded with `weights_only=True`. The default layout is `[tokens, heads, 128]`; use `--layout head-major` for `[heads, tokens, 128]`. Set `--model-label` to identify the checkpoint. For example, after placing a capture at `captures/qwen3_4b_128k.pt`:

```bash
python benchmarks/bench_routing_sweep.py \
    --capture captures/qwen3_4b_128k.pt --layout token-major \
    --model-label Qwen3-4B --lengths 131072 --topks 16 32 64 \
    --mode graph-native --output routing_qwen3_4b_128k.json
```

The paper sweeps **16K, 32K, 64K, and 128K**, using captured Qwen3-4B and GQA-30B-A3B tensors with **32 Q heads, 8 KV heads, D128**, replayed on a single rank. Repeat the command with each corresponding capture; slicing a longer capture is supported but explicitly flagged and is not equivalent to recapturing the model at a shorter context. Model captures, including the internal GQA checkpoint's tensors, are not bundled with this repository.

The script preserves threshold ties and never truncates a pattern to force a match. An unreachable budget is reported as `unmatched`, without a speedup. Insufficient memory or failed graph capture is reported as a failure, not silently retried with another timing mode. Padded CSC alone needs 16 GiB for 128K tokens and 32 Q heads; the script estimates workspace needs before running. Existing output files are not overwritten.

This provides an executable routing-sweep entry point, not a claim that the paper's numbers have been rerun on the current checkout. The cumulative execution, ownership, and pipeline ablation tables still require their separate controlled harnesses; the Top-K scripts below do not reproduce those tables.

### Baseline benchmarks

Other performance scripts are provided in [benchmarks](benchmarks):

- [Top-K/query-centric baseline](benchmarks/bench_lightning_moba.py), including routing and backward; this does not benchmark the default dynamic pipeline.
- [Baseline attention-only timing](benchmarks/bench_lightning_moba_attn.py), with routing outside the timed region.
- [Top-K baseline memory measurements](benchmarks/bench_lightning_moba_memory.py).
- Component benchmarks for [Top-K](benchmarks/bench_flash_topk.py), [mean pooling](benchmarks/bench_triton_mean_pool.py), and [sorting](benchmarks/bench_varlen_sort.py).

Benchmarks may require additional packages, including FlashAttention, NumPy, and pandas. Inspect each script's dependencies and workload sizes before running it. End-to-end and attention-only timings measure different work; compare results using the same inputs, selection settings, and hardware.

## Acknowledgements

LightningMoBA builds on FlashMoBA and [FlashAttention](https://github.com/Dao-AILab/flash-attention), and uses [CUTLASS](https://github.com/NVIDIA/cutlass), [PyTorch](https://github.com/pytorch/pytorch), and [Triton](https://github.com/triton-lang/triton). We thank their authors and maintainers. Upstream copyright notices and third-party licenses are retained in the source tree.

## Citation

If you use LightningMoBA in your research, please cite our [paper](https://arxiv.org/abs/2611.26268):

```bibtex
@misc{shi2026lightningmoba,
  title         = {Accelerating Mixture of Block Attention via Dynamic Routing, {KV}-Centric Execution and Runtime Pipelining},
  author        = {Yingqi Shi and Junru Lu and Yun Liu and Di Yin and Xing Sun},
  year          = {2026},
  eprint        = {2611.26268},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2611.26268}
}
```

## License

Copyright (C) 2026 Tencent. LightningMoBA is licensed under the BSD-3-Clause [LICENSE](LICENSE).
