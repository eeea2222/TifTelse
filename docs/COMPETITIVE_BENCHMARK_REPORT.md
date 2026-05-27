# Competitive Benchmark Report

This report compares TIF/TELSE against the strongest local baselines currently
available in the benchmark suite:

- `torch_eager_ref`: direct PyTorch reference using `torch.lerp`.
- `torch_compile_ref`: the same reference under `torch.compile(fullgraph=True)`.
- `tif_auto`: `fused_gated_residual(..., backend="auto")`.
- `tif_torch`: `fused_gated_residual(..., backend="torch")`.
- `tif_triton_forced`: `fused_gated_residual(..., backend="triton")` where supported.

The raw benchmark logs and JSON artifacts for this run are stored locally under:

```text
test_results/competitive_20260527_165021/
```

They are intentionally not committed as package data because the raw logs are
large and machine-specific.

## Test Matrix

The competitive run covered:

- CPU `float32`.
- CUDA `float32`, `float16`, and `bfloat16`.
- Forward and backward/autograd measurements.
- Activations: `silu`, `swish`, `relu`, `gelu`, and `custom_relu_tanh`.
- Gates: `sigmoid`, `hard_sigmoid`, and selected `clamp01` cases.
- Layouts: contiguous, strided-last, transpose, and broadcast.
- Shapes from `standard` and `hard` presets, including:
  - `[4096]`
  - `[65536]`
  - `[16, 128, 256]`
  - `[16, 256, 512]`
  - `[8, 512, 1024]` in the Triton-focused sweep

Overall measured scope:

```text
total_rows=5394
measured=4967
failures=1
```

The one observed failure was in the `torch_compile_ref` baseline, not in
`tif_auto` or `tif_torch`:

```text
backend=torch_compile_ref
dtype=bfloat16
layout=broadcast
shape=[16, 256, 512]
autograd=True
error=grad mismatch input=0 max_diff=0.25
```

## Speed vs Eager PyTorch

Speedup is reported relative to `torch_eager_ref` for the same workload.

```text
torch_compile_ref:  n=1211 wins=608 win_rate=50.2% median=1.005 avg=1.148 best=18.084
tif_torch:          n=1212 wins=506 win_rate=41.7% median=0.971 avg=1.064 best=2.956
tif_auto:           n=1212 wins=455 win_rate=37.5% median=0.950 avg=1.034 best=3.890
tif_triton_forced:  n=120  wins=18  win_rate=15.0% median=0.622 avg=0.738 best=1.978
```

Best observed TIF result:

```text
backend=tif_auto
selected_backend=torch_composition
device=cuda
dtype=float32
shape=[65536]
activation=gelu
gate=hard_sigmoid
layout=contiguous
autograd=True
speedup_vs_eager=3.890
```

Best observed forced Triton result:

```text
backend=tif_triton_forced
device=cuda
dtype=float32
shape=[8, 512, 1024]
activation=silu
gate=sigmoid
layout=contiguous
autograd=False
speedup_vs_eager=1.978
```

## Fastest Backend Counts

For each workload/autograd pair, the benchmark selected the lowest-latency row.

```text
torch_compile_ref: fastest_count=467 avg_speedup_when_fastest=1.601
torch_eager_ref:   fastest_count=368 avg_speedup_when_fastest=1.000
tif_torch:         fastest_count=269 avg_speedup_when_fastest=1.443
tif_auto:          fastest_count=108 avg_speedup_when_fastest=1.513
```

When `tif_auto` is compared against the best available torch competitor
(`torch_eager_ref` or `torch_compile_ref`) for the same workload:

```text
n=1212
wins=285
win_rate=23.5%
median_ratio=0.838
avg_ratio=0.877
```

This means TIF/TELSE is useful, but it should not claim universal speed wins
over `torch.compile`.

## CUDA Memory

Average peak CUDA memory across measured CUDA rows:

```text
torch_compile_ref:  23.60 MB
tif_auto:           29.43 MB
tif_torch:          30.07 MB
tif_triton_forced:  30.02 MB
```

Triton-eligible forward rows can reduce memory for specific fused cases, but
the total average is workload-dependent and does not beat `torch.compile` across
the full suite.

## Advantages

- Correctness for `tif_auto` and `tif_torch` held across the measured matrix.
- Registered custom ops are kept on the PyTorch path instead of being routed to
  a fixed Triton kernel with incompatible semantics.
- Unsupported Triton cases skip or fall back for clear reasons such as custom
  ops, non-contiguous tensors, broadcast shapes, unsupported gate/activation
  pairs, or CPU tensors.
- TIF/TELSE wins against eager PyTorch in many workloads, especially selected
  backward/fallback cases.
- `debug=True` gives concrete backend explanations, which makes performance
  behavior auditable.

## Disadvantages

- `torch.compile` is a strong competitor and is often fastest on large CUDA
  forward workloads.
- Forced Triton is not broadly faster; it won only 15.0% of measured forced
  Triton rows in this run.
- Triton backward is frequently slower than PyTorch or Inductor-backed paths.
- `backend="auto"` is intentionally conservative and does not maximize speed in
  every workload.
- Performance is sensitive to dtype, shape, layout, autograd, and local GPU.

## Verdict

TIF/TELSE is correctness-safe and useful as an explicit tensor conditional API
with conservative backend routing. It provides real speed wins against eager
PyTorch in selected workloads, but it should be positioned honestly:

- Use TIF/TELSE for explicit tensor value selection, soft gating, and safe fused
  routing.
- Prefer `torch.compile` when it can own and optimize the full graph.
- Treat Triton as a narrow optimization path, not a universal acceleration
  backend.
