# TIF/TELSE: Mathematical If/Else for AI Training

> **if/else controls the program. TIF/TELSE controls the tensor flow.**

TIF/TELSE is a **tensor-conditional execution layer** for AI training. It is not a cosmetic replacement for Python `if/else`. It is the math you actually want when "branching" needs to live on the GPU, inside autograd, and inside a `torch.compile` graph.

```
Python if/else  ->  control flow for programs
TIF / TELSE     ->  mathematical conditional flow for tensors
```

Branching is control flow. Gating is math. This library is about the math.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [Quick Start](#quick-start)
- [The core idea](#the-core-idea)
- [Features](#features)
- [Built-in registries](#built-in-registries)
- [Common patterns](#common-patterns)
- [Backend philosophy](#backend-philosophy)
- [When to use TIF/TELSE](#when-to-use-tiftelse)
- [When NOT to use it](#when-not-to-use-it)
- [Advantages](#advantages)
- [Disadvantages](#disadvantages)
- [Performance notes](#performance-notes)
- [Benchmark honesty](#benchmark-honesty)
- [Mental model](#mental-model)
- [Design philosophy](#design-philosophy)
- [FAQ](#faq)

---

## Why this exists

Modern AI training is full of branches that *look* like `if/else` but cannot be expressed that way without paying a cost:

- Routing tokens between experts.
- Switching activations conditionally per channel.
- Gating residual connections.
- Applying different loss penalties in different regions of a batch.
- Masking padding tokens, ignored labels, dropped paths.

Writing these as Python `if/else` over CUDA tensor data forces:

1. A device→host sync (`bool(tensor)`, `.item()`, `.cpu()`).
2. A graph break (`torch.compile` cannot trace through it).
3. Dead autograd (gradients vanish through control flow).
4. Lost throughput.

TIF/TELSE replaces "if/else on tensors" with **value selection and gating**, the operations that are:

- **GPU-native** — stays on device.
- **Autograd-friendly** — gradients flow.
- **Inductor-friendly** — `torch.compile` can fuse it.
- **Backend-routed** — picks PyTorch, `torch.compile`, or Triton based on what is actually fastest for the workload.

---

## Quick Start

```bash
pip install -e .                # base install
pip install -e .[triton]        # optional Triton fast paths
pip install -e .[test]          # for tests
```

```python
import torch
from tif_telse import tif, telse, soft_tif, fused_gated_residual

# 1) Scalar Python bool — true lazy branching.
y = tif(True, "yes", telse("no"))            # -> "yes"  (callable branches are lazy)

# 2) Boolean tensor mask — torch.where-equivalent, autograd-safe.
mask = torch.tensor([True, False, True])
y = tif(mask, torch.ones(3), telse(torch.zeros(3)))

# 3) Soft differentiable gating.
gate = torch.tensor([0.2, 0.5, 0.9], requires_grad=True)
y = soft_tif(gate, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.0, 0.0, 0.0]))

# 4) Fused AI-training pattern: residual + sigmoid-gated activation.
logits   = torch.randn(64, 512, device="cuda")
a        = torch.randn(64, 512, device="cuda")
b        = torch.randn(64, 512, device="cuda")
residual = torch.randn(64, 512, device="cuda")
y = fused_gated_residual(logits, a, b, residual, activation="silu", gate="sigmoid")
```

---

## The core idea

A Python conditional:

```python
if c:
    y = a
else:
    y = b
```

Becomes one of three mathematical forms, depending on what `c` is:

### Hard value selection

```python
y = where(c, a, b)
```

Boolean mask. Picks per-element. No blending. Gradients flow through whichever side was selected.

### Mask algebra

```python
y = c * a + (1 - c) * b
```

`c` is numeric (often 0/1 or a hard mask cast to float). Algebraically identical to `where` for 0/1 inputs but expressed as arithmetic — sometimes preferred for fusion or for differentiable hard gating tricks.

### Soft differentiable gating

```python
m = sigmoid(gate_logits)
y = m * a + (1 - m) * b          # equivalently: torch.lerp(b, a, m)
```

`gate` is a real-valued tensor. Now `y` is differentiable with respect to the gate **and** both branches. This is what powers learned routing.

TIF/TELSE gives you one API across all three and refuses to silently confuse them.

---

## Features

| Primitive | Purpose |
|---|---|
| `tif(condition, then, telse(else_))` | Unified value-selection entry point. Dispatches on `condition` type. |
| `telse(value)` | Marker wrapper to make the else branch syntactically explicit. |
| `soft_tif(gate, a, b, *, clamp=False)` | Differentiable lerp gating between `a` and `b`. |
| `straight_through_tif(gate, a, b, *, threshold=0.5)` | Hard-forward, soft-backward (STE) gating. |
| `fused_gated_residual(...)` | Training pattern `residual + lerp(b, act(a), gate(logits))`, backend-routed. |
| `detect_capabilities()` | Reports torch/numpy/triton availability, `torch.compile` state, CUDA. |
| `register_activation(name, fn, *, aliases=(), overwrite=False)` | Add a user activation to the registry. |
| `register_gate(name, fn, *, aliases=(), overwrite=False)` | Add a user gate to the registry. |
| `available_activations()` / `available_gates()` | Inspect registries. |
| `compose_ops(*ops)` | Build a single callable from a pipeline of registered ops or callables. |
| `debug=True` | Returns a `BackendReport` explaining what was selected and why. |

### `tif` dispatch table

| `condition` type | Behavior | Mode required |
|---|---|---|
| Python `bool` | True lazy branching; only the selected branch's callable runs. | — |
| Python numeric scalar (not `bool`) | Raises unless `mode="scalar"` is set. Explicit truthiness opt-in. | `"scalar"` |
| `numpy.ndarray` of `bool` | `np.where(c, a, b)`. | auto |
| `numpy.ndarray` of float | Ambiguous by default. Must opt into `c*a + (1-c)*b` blending or nonzero masking. | `"soft"` or `"hard_mask"` |
| `torch.Tensor` of `bool` | `torch.where`, broadcasts, preserves dtype/device/grad. | auto |
| `torch.Tensor` of float | Ambiguous by default. Must opt into `"soft"` or `"hard_mask"`. | explicit |

> **Why the ambiguity check?** Float tensor conditions could mean "soft gate" or "non-zero mask". Different math, different gradients. The library refuses to guess.

---

## Built-in registries

**Activations:** `identity`/`none`, `relu`, `gelu`, `silu`/`swish`, `mish`, `elu`, `selu`, `leaky_relu`, `hardtanh`, `hardswish`, `hardsigmoid`/`hard_sigmoid`, `softplus`, `sigmoid`, `tanh`.

**Gates:** `sigmoid`, `identity`/`none`, `clamp01`, `hard_sigmoid`.

Names are case-insensitive, and hyphens normalize to underscores. Common aliases such as `swish`, `none`, `hardsigmoid`, and `hard_sigmoid` are registered explicitly. Parameterized activations take kwargs:

```python
fused_gated_residual(
    logits, a, b, residual,
    activation="leaky_relu",
    activation_kwargs={"negative_slope": 0.2},
)
```

Custom callables work too — `torch.compile` compatibility depends on whether Dynamo can trace them.

Registered user ops are reported as custom ops and are routed through PyTorch composition unless a future backend explicitly supports them. Built-in names and aliases are reserved; pass `overwrite=True` only to replace a previous user-registered name.

---

## Common patterns

### 1) Basic scalar replacement (lazy)

```python
y = tif(use_cache, lambda: load_from_cache(), telse(lambda: compute()))
```

Only the chosen callable runs. This is regular Python control flow — `tif` just makes the value-selection intent explicit.

### 2) Tensor mask selection

```python
mask = labels != -100                                   # ignore index
loss = tif(mask, per_token_loss, telse(torch.zeros_like(per_token_loss)))
```

Equivalent to `torch.where(mask, per_token_loss, 0.)` — but composes cleanly with the rest of the library.

### 3) Soft differentiable routing

```python
gate_logits = router(x)                                 # [B, T, 1]
gate        = torch.sigmoid(gate_logits)
y           = soft_tif(gate, expert_a(x), expert_b(x))  # blended, differentiable
```

Both branches are computed (this is the price of tensor conditionals). The gradient flows through `gate`, `expert_a`, and `expert_b`.

### 4) Conditional residual block (the headline pattern)

```python
y = fused_gated_residual(
    logits, a, b, residual,
    activation="silu",
    gate="sigmoid",
    backend="auto",
)
# Mathematically: residual + sigmoid(logits) * silu(a) + (1 - sigmoid(logits)) * b
```

The router decides per-call whether to use PyTorch composition, let `torch.compile`/Inductor own it, or dispatch to the Triton fused kernel.

### 5) Activation/gate registry by name

```python
y = fused_gated_residual(logits, a, b, residual, activation="gelu", gate="hard_sigmoid")
```

### 6) User-registered activation

```python
from tif_telse import register_activation

def square(x):
    return x * x

register_activation("square", square, aliases=("sq",))
y = fused_gated_residual(logits, a, b, residual, activation="sq")
```

### 7) Composed operation pipeline

```python
from tif_telse import compose_ops
import torch

custom = compose_ops("relu", torch.tanh, lambda x: x * 0.5)
y = fused_gated_residual(logits, a, b, residual, activation=custom)
```

### 8) MoE-like soft expert blending

```python
gate    = torch.sigmoid(router(x))                          # [B, T, 1]
y_expert = soft_tif(gate, expert_a(x), expert_b(x))
```

For top-k hard routing, use `straight_through_tif` to get hard-forward / soft-backward behavior.

### 9) Conditional loss shaping

```python
heavy = (target_class == rare_class).float()
weight = soft_tif(heavy, torch.tensor(5.0), torch.tensor(1.0))
loss   = (weight * per_sample_loss).mean()
```

### 10) Debugging backend choice

```python
y, report = fused_gated_residual(logits, a, b, residual, activation="silu", debug=True)
print(report.backend, report.triton_used, report.reason)
# e.g. "triton_fused_gate_silu_residual True 'auto selected Triton fused kernel for ...'"
```

`debug=True` carries observability cost. In production keep it off — the non-debug path skips capability detection and report construction entirely.

---

## Backend philosophy

> **Triton is an accelerator, not a religion. The fastest correct backend wins.**

The router applies these rules, in order:

1. **CPU, non-contiguous tensors, unsupported dtypes, shape mismatches, missing Triton, missing CUDA, or custom callables** → PyTorch composition.
2. **Inside `torch.compile`/Inductor** → PyTorch composition. Let Inductor own fusion. Inserting a Triton custom op here often breaks fusion downstream.
3. **`backend="triton"` forced** → Triton if eligible, else a useful `RuntimeError` explaining why it isn't.
4. **`backend="auto"` on large contiguous CUDA tensors with the known-fast pattern** (`activation="silu"`, `gate="sigmoid"`) → Triton fused kernel.
5. **Mid-size or backward-heavy autograd cases where measured Triton backward is weaker** → PyTorch composition.
6. **Anything else** → PyTorch composition.

### Hard invariants

- **Never** move CUDA tensors to CPU implicitly.
- **Never** call `.item()`, `.cpu()`, `.numpy()`, `bool(tensor)`, or detach silently to decide a branch.
- **Always** preserve dtype, device, broadcasting, and autograd.
- **Never** route arbitrary user callables to Triton.

---

## When to use TIF/TELSE

- Element-wise tensor masking and selection.
- Activation blending or activation switching mid-network.
- Residual blending (the `fused_gated_residual` pattern).
- Differentiable gates and learned routers.
- Soft MoE / soft expert mixing.
- Training-time conditional math (per-sample loss weighting, curriculum gating).
- Reducing graph breaks in `torch.compile` regions.
- Replacing CUDA-side `if/else` that currently forces a sync.
- Workloads where a fused CUDA path is provably faster.
- Anywhere you want conservative heuristic routing instead of a hand-picked backend.

## When NOT to use it

- File I/O, network requests, subprocesses.
- Anything that mutates state outside the tensor world.
- Branches where one side raises exceptions and the other doesn't.
- Branches where only one side is *safe to execute* (the other dereferences invalid memory, hits a guard, etc.). Tensor conditionals compute both sides.
- Program-level control flow ("if not authenticated, redirect").
- Arbitrary Python logic that fundamentally cannot be tensorized.

> **Rule of thumb:** if the two branches have side effects, you want `if/else`. If they produce values, you might want TIF/TELSE.

---

## Advantages

- **GPU-friendly conditional math** — no device→host syncs.
- **Autograd-compatible soft conditionals** — `soft_tif` and `straight_through_tif`.
- **`torch.compile`-friendly composition** — the PyTorch path is fully traceable; the auto router yields to Inductor when Inductor is the right tool.
- **Registry + callable flexibility** — named ops for stability, callables for experimentation.
- **Cleaner training code** — value selection instead of ad-hoc `where` chains.
- **Conservative heuristic routing** — the Triton path is used only for a narrow measured pattern.
- **Triton acceleration for known-fast fused paths** — narrow, opt-in-by-shape, falls back honestly.
- **Honest fallback to PyTorch/Inductor** — the library does not pretend Triton is always faster.

## Disadvantages

- **Not a universal `if/else` replacement.** Tensor conditionals are value selection.
- **Both branches usually compute.** This is the price of staying on the GPU.
- **Triton is not always faster.** Launch overhead, autograd path, and dtype all change the answer.
- **`torch.compile`/Inductor can beat Triton** for many shapes. The router knows this.
- **Custom callables depend on Dynamo traceability** for `torch.compile` paths.
- **Soft gates are not exact hard `if/else`.** They are smooth approximations; choose the right primitive (`tif` vs `soft_tif` vs `straight_through_tif`).
- **Performance depends on shape, dtype, layout, device, and autograd mode.** There is no single "fastest backend".

---

## Performance notes

- Non-debug calls skip `RuntimeCapabilities` construction entirely. `debug=True` is for diagnostics, not the hot path.
- Static capability flags (Triton importable, `torch.compiler` present, `torch.library.triton_op` present) are cached at import — they cannot change at runtime.
- The Triton fused kernel uses `n_elements` as a runtime arg (not `tl.constexpr`) so a single compiled kernel handles every tensor size. Only `BLOCK_SIZE` is `constexpr`, with autotune over `{256, 512, 1024}`.
- Auto routing currently steers to PyTorch composition under `torch.compile`, on small tensors (`< 1M` elements), and for mid-size low-precision or large fp32 autograd cases where Triton backward measured weaker.
- The fast Triton path is intentionally narrow: `silu`/`swish` activation + `sigmoid` gate + tensor residual + contiguous CUDA + matching shapes + matching dtype in `{fp16, bf16, fp32}`.

---

## Benchmark honesty

This is the section most performance libraries skip. We won't.

- **Compare against strong baselines.** Eager PyTorch with `torch.lerp` is a strong baseline. Add `torch.compile` to that and it gets stronger.
- **Old algebra baselines (`c*a + (1-c)*b`) are weak.** Beating those is not a real win. The library benches against `torch.lerp(b, act(a), gate(logits))` and the compiled version.
- **The report tells the truth.** When `debug=True`, the `BackendReport` says which backend won and why — including the cases where PyTorch beat Triton.
- **No "always faster" claim.** TIF/TELSE uses conservative benchmark-informed heuristics, not backend loyalty.
- **Run the benchmark on your hardware.**

```bash
python benchmarks/bench_fused_gated_residual.py --device cuda
python benchmarks/hard_benchmark.py --preset hard --device all \
       --include-forced-triton --record-skips
```

The hard benchmark sweeps activations × gates × layouts × dtypes × shapes × devices × autograd, checks correctness against a `torch.lerp`-based reference, and reports peak memory on CUDA.

### JSON export

Both benchmarks emit machine-readable JSON for CI dashboards and post-hoc analysis:

```bash
# Print JSON to stdout
python benchmarks/bench_fused_gated_residual.py --device cuda --json > bench.json
python benchmarks/hard_benchmark.py --preset hard --json > hard.json

# Or write directly to a file (hard benchmark only)
python benchmarks/hard_benchmark.py --preset standard --output-json hard.json
```

Each row carries the fields needed to compare apples to apples: `backend`, `selected_backend`, `triton`, `compiled`, `autograd`, `activation`, `gate`, `layout`, `device`, `dtype`, `shape`, `latency_ms`, `speedup_vs_eager`, `peak_memory_mb`, `correct`, `reason`. The final lines printed by each script (`PERFORMANCE_CONTRACT_PASS`, `HARD_BENCHMARK_PASS`, etc.) make CI gating easy.

### How to read a benchmark row

A row only counts as a "win" if **all three** are true: `correct=True`, `latency_ms < eager_baseline`, and the comparison is against a strong baseline (`torch_eager_ref` with `torch.lerp`, not the naive `c*a + (1-c)*b` algebra). Rows with `triton=True` but `speedup_vs_eager < 1.0` are honest losses — they go in the report unchanged.

---

## Mental model

```
                user-facing call
                       |
                       v
            +----------------------+
            |  resolve_activation  |   (registry lookup OR custom callable)
            |  resolve_gate        |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |   capability checks  |   (cached static flags + dynamic compile state)
            +----------+-----------+
                       |
                       v
            +----------------------+
            |    backend router    |   PyTorch eager   <- default + Inductor regions
            |                      |   torch.compile   <- yielded to Inductor implicitly
            |                      |   Triton kernel   <- narrow, known-fast, contiguous CUDA
            +----------+-----------+
                       |
                       v
              output tensor
       (preserves dtype, device, autograd,
        broadcasting, and CUDA placement)
```

---

## Design philosophy

- **Value selection, not side effects.** Tensor branches are expressions, not statements.
- **No silent device transfers, ever.** If you want to leave the GPU, do it yourself.
- **No silent ambiguity.** Float tensor conditions, numeric scalar conditions, and bool conditions are different kinds of object. The API forces you to be explicit when it matters.
- **Triton is an accelerator, not a religion.** Use it where it's actually faster.
- **The fastest correct backend wins.** Always.
- **The report does not lie.** `debug=True` shows the real selection and the real reason — even when the reason is "PyTorch is faster here".
- **Narrow > wide for fast paths.** The Triton kernel handles one pattern very well. Everything else uses PyTorch.

---

## FAQ

**Is this a `torch.where` wrapper?**
No. For boolean tensor masks, `tif` is `torch.where`. But TIF/TELSE is a layered API: hard selection, mask algebra, soft gating, straight-through estimators, and fused training patterns under one roof — with backend routing and a registry.

**Will Triton be faster on my model?**
Maybe. Run the benchmark. The auto router will decline Triton when it shouldn't win. You can force `backend="triton"` to measure, and force `backend="torch"` to compare.

**Does it work with `torch.compile(fullgraph=True)`?**
Yes for the PyTorch composition path — and the auto router specifically yields to Inductor when it detects an active compile context, so Inductor can own the fusion.

**Are both branches always computed?**
For tensor conditions, yes. That is the cost of staying on the GPU. For Python `bool` conditions with callable branches, only the selected callable runs.

**What about top-k MoE routing?**
Use `straight_through_tif` for binary hard routing with soft gradients. For full top-k MoE you typically want a dedicated MoE library; TIF/TELSE handles the two-expert / binary case cleanly and the differentiable gating math underneath.

**Does it call `.cpu()` or `.item()` on my tensors?**
Never. The library will refuse to guess across devices and will not move data to host to make a branch decision.

**Why is `n_elements` not `tl.constexpr` in the kernel?**
Because making it a compile-time constant forces a recompile per tensor size. Keeping it a runtime arg means one compiled kernel serves all sizes.

**How do I add my own activation?**

```python
from tif_telse import register_activation

def my_act(x):
    return x.tanh() * x.sigmoid()

register_activation("my_act", my_act, aliases=("mine",))
```

Then use `activation="my_act"` anywhere.

---

## A closing line

> Python `if/else` is how programs make decisions.
> TIF/TELSE is how **tensors** make decisions — on the GPU, through autograd, inside fused kernels, without lying about which backend won.
