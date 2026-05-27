# TIF / TELSE

TIF means Tensorized If. TELSE means Tensorized Else.

This package is an experimental but serious first milestone for mathematical conditional execution in AI training. It is not a replacement for Python control flow and it does not claim that every `if/else` can or should be tensorized.

Python `if/else` controls programs. TIF/TELSE selects or blends tensor values.

## What is implemented

- `tif(condition, then_value, telse(else_value))`
- `soft_tif(gate, then_value, else_value)`
- `straight_through_tif(gate, then_value, else_value)` as an explicit experimental primitive
- `detect_capabilities()` for runtime feature detection
- `available_activations()` / `available_gates()`
- `register_activation(...)` / `register_gate(...)`
- `compose_ops(...)` for simple PyTorch op pipelines
- `fused_gated_residual(...)` for the training-relevant pattern:

```python
y = residual + sigmoid(logits) * activation(a) + (1 - sigmoid(logits)) * b
```

Built-in activations include `identity`/`none`, `relu`, `gelu`, `silu`/`swish`, `mish`, `elu`, `selu`, `leaky_relu`, `hardtanh`, `hardswish`, `hardsigmoid`/`hard_sigmoid`, `softplus`, `sigmoid`, and `tanh`.

Built-in gates include `sigmoid`, `identity`/`none`, `clamp01`, and `hard_sigmoid`.

Users may also pass PyTorch callables:

```python
import torch
from tif_telse import compose_ops, fused_gated_residual

custom_activation = compose_ops("relu", torch.tanh)
y = fused_gated_residual(logits, a, b, residual, activation=custom_activation)
```

Registered user ops are treated as custom ops for backend routing. Built-in names and aliases cannot be overwritten, so the narrow Triton kernel is only used for the known built-in `silu`/`swish` activation plus built-in `sigmoid` gate.

Parameterized built-ins use kwargs:

```python
y = fused_gated_residual(
    logits,
    a,
    b,
    residual,
    activation="leaky_relu",
    activation_kwargs={"negative_slope": 0.2},
)
```

The general `tif` API stays conservative and PyTorch-native by default. The fused primitive uses dynamic backend routing:

- CPU, non-contiguous tensors, unsupported dtypes, shape mismatches, missing Triton, or missing CUDA use PyTorch composition.
- `torch.compile`/Inductor contexts use PyTorch composition so Inductor can own fusion.
- Large contiguous CUDA tensors can use the Triton fused kernel only for forward/inference calls in the known-fast `sigmoid` gate + `silu`/`swish` activation pattern when the local heuristic says it is likely to win.
- Autograd calls use PyTorch composition by default because the measured Triton backward path is usually weaker than PyTorch/Inductor.
- `backend="triton"` can force Triton for explicit profiling and raises a useful error if unsupported.
- `debug=True` returns a backend report with the selected backend, reason, resolved activation/gate, whether either is custom, and detected capabilities.

The router is deliberately empirical. If the custom path is not known to be faster or safe, it falls back to PyTorch. Custom callables are regular PyTorch tensor functions; `torch.compile` compatibility depends on whether TorchDynamo can trace the callable.

## Semantics

Equivalent value-selection cases:

```python
y = tif(mask, a, telse(b))
```

For boolean PyTorch tensor masks, this is equivalent to:

```python
y = torch.where(mask, a, b)
```

For soft gates:

```python
y = soft_tif(gate, a, b)
```

is:

```python
y = b + gate * (a - b)
```

Scalar Python booleans preserve lazy branch semantics. If branches are callables, only the selected callable is evaluated.

## What TIF/TELSE is not

TIF/TELSE is not equivalent to Python control flow when branches have side effects, perform I/O, mutate state, raise different exceptions, rely on only one branch existing, or do work beyond value selection. Tensor branches are value expressions; both branch values generally need to exist unless a specialized fused primitive says otherwise.

CUDA tensor conditions are never converted to Python booleans. The library does not call `.item()`, `.cpu()`, `.numpy()`, or detach tensors to decide a branch.

## Benchmarks

Run:

```bash
python benchmarks/bench_fused_gated_residual.py --device cuda
```

The benchmark reports latency, speedup ratios, activation, backend, dtype, shape, compile status, Triton status, autograd status, and pass/fail. By default it covers `silu`, `swish`, `relu`, `gelu`, `mish`, and one composed custom activation. A backend is only performance-worthy if it is correct and faster than at least one strong baseline for the measured workload.

For a broader stress matrix, run:

```bash
python benchmarks/hard_benchmark.py --preset hard --device all --include-forced-triton --record-skips
```

The hard benchmark covers many activations, gates, layouts, dtypes, CPU/CUDA devices, eager PyTorch, `torch.compile`, TIF auto, TIF torch, optional forced Triton, forward latency, backward latency, correctness checks, and CUDA peak memory.

No speedup is claimed by this README. Run the benchmark on the target machine and use the report.

Both benchmarks support `--json` for machine-readable output; `hard_benchmark.py` also supports `--output-json PATH`. See [docs/USAGE_AND_DESIGN.md](docs/USAGE_AND_DESIGN.md#json-export) for the row schema.

## Examples

Runnable scripts in [examples/](examples/):

| File | Shows |
|---|---|
| [`01_basic_tif.py`](examples/01_basic_tif.py) | Scalar bool (lazy), torch bool mask, numpy bool mask, ambiguous-float TypeError. |
| [`02_soft_tif.py`](examples/02_soft_tif.py) | `soft_tif` differentiable gating and `straight_through_tif` STE. |
| [`03_custom_activation.py`](examples/03_custom_activation.py) | `register_activation` with aliases, used through the fused primitive. |
| [`04_compose_ops.py`](examples/04_compose_ops.py) | `compose_ops` pipeline mixing registry names and inline callables. |
| [`05_fused_gated_residual.py`](examples/05_fused_gated_residual.py) | The headline AI-training pattern on CPU and CUDA with a `BackendReport`. |

```bash
python examples/05_fused_gated_residual.py
```

## Public API

Everything below is importable from `tif_telse`. Anything not on this list is private and may change without notice.

### Value selection
- `tif(condition, then_value, telse(else_value), *, mode="auto", strict=True, debug=False)` — unified value-selection entry point.
- `telse(value)` — wrapper marker for the else branch.
- `soft_tif(gate, then_value, else_value, *, clamp=False)` — differentiable lerp gating.
- `straight_through_tif(gate, then_value, else_value, *, threshold=0.5)` — hard-forward, soft-backward gating.

### Fused training primitive
- `fused_gated_residual(logits_or_gate, a, b, residual=None, *, activation="silu", gate="sigmoid", activation_kwargs=None, gate_kwargs=None, backend="auto", debug=False)`

### Registry
- `register_activation(name, fn, *, aliases=(), overwrite=False)`
- `register_gate(name, fn, *, aliases=(), overwrite=False)`
- `available_activations() -> tuple[str, ...]`
- `available_gates() -> tuple[str, ...]`
- `compose_ops(*ops) -> Callable[[Tensor], Tensor]`
- `resolve_op(op, kwargs=None) -> ResolvedOp` — for advanced custom backends.

### Capability detection
- `detect_capabilities(torch_mod=None, *, device=None) -> RuntimeCapabilities`
- `RuntimeCapabilities` — frozen dataclass with `torch_available`, `numpy_available`, `triton_available`, `triton_op_available`, `torch_compile_available`, `compile_active`, `cuda_available`, `device_type`.

### Reports
- `BackendDecision` — base report shape (`backend`, `triton_used`, `reason`, `capabilities`).
- `BackendReport` — extended with `activation`, `gate`, `custom_activation`, `custom_gate`. Returned by `fused_gated_residual(..., debug=True)`.

## Further reading

See [docs/USAGE_AND_DESIGN.md](docs/USAGE_AND_DESIGN.md) for the full design rationale, common patterns, backend philosophy, performance notes, benchmark honesty section, mental-model diagram, and FAQ.
