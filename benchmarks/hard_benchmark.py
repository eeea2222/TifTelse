from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tif_telse import compose_ops, fused_gated_residual


TensorFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class OpCase:
    name: str
    arg: str | TensorFn
    fn: TensorFn
    kwargs: dict[str, object] | None = None


@dataclass(frozen=True)
class Workload:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: str
    activation: OpCase
    gate: OpCase


@dataclass
class BenchRow:
    workload_id: int
    backend: str
    selected_backend: str
    reason: str
    activation: str
    gate: str
    layout: str
    device: str
    dtype: str
    shape: list[int]
    compiled: bool
    triton: bool
    autograd: bool
    latency_ms: float | None
    speedup_vs_eager: float | None
    peak_memory_mb: float | None
    correct: bool
    skipped: bool
    error: str


def activation_cases() -> dict[str, OpCase]:
    custom = compose_ops("relu", torch.tanh)
    return {
        "identity": OpCase("identity", "identity", lambda x: x),
        "relu": OpCase("relu", "relu", F.relu),
        "gelu": OpCase("gelu", "gelu", F.gelu),
        "silu": OpCase("silu", "silu", F.silu),
        "swish": OpCase("swish", "swish", F.silu),
        "mish": OpCase("mish", "mish", F.mish),
        "elu": OpCase("elu", "elu", F.elu),
        "selu": OpCase("selu", "selu", F.selu),
        "leaky_relu": OpCase(
            "leaky_relu",
            "leaky_relu",
            lambda x: F.leaky_relu(x, negative_slope=0.2),
            {"negative_slope": 0.2},
        ),
        "hardtanh": OpCase(
            "hardtanh",
            "hardtanh",
            lambda x: F.hardtanh(x, min_val=-0.25, max_val=0.75),
            {"min_val": -0.25, "max_val": 0.75},
        ),
        "hardswish": OpCase("hardswish", "hardswish", F.hardswish),
        "softplus": OpCase("softplus", "softplus", F.softplus),
        "tanh": OpCase("tanh", "tanh", torch.tanh),
        "custom_relu_tanh": OpCase("custom_relu_tanh", custom, lambda x: torch.tanh(F.relu(x))),
    }


def gate_cases() -> dict[str, OpCase]:
    return {
        "sigmoid": OpCase("sigmoid", "sigmoid", torch.sigmoid),
        "hard_sigmoid": OpCase("hard_sigmoid", "hard_sigmoid", F.hardsigmoid),
        "clamp01": OpCase("clamp01", "clamp01", lambda x: x.clamp(0, 1)),
        "identity": OpCase("identity", "identity", lambda x: x),
    }


def parse_names(raw: str, available: dict[str, OpCase]) -> list[OpCase]:
    if raw == "all":
        return list(available.values())
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"Unknown names {unknown}. Available: {', '.join(sorted(available))}")
    return [available[name] for name in names]


def shapes_for_preset(preset: str) -> list[tuple[int, ...]]:
    if preset == "smoke":
        return [(4096,), (16, 128, 256)]
    if preset == "standard":
        return [(4096,), (65536,), (16, 128, 256), (16, 256, 512)]
    if preset == "hard":
        return [(1024,), (65536,), (262144,), (16, 128, 256), (16, 256, 512), (8, 512, 1024)]
    raise SystemExit("preset must be smoke, standard, or hard")


def devices_for_arg(raw: str) -> list[torch.device]:
    if raw == "auto":
        return [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
    if raw == "all":
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        return devices
    return [torch.device(raw)]


def dtype_names_for_device(device: torch.device, requested: str) -> list[str]:
    if requested != "auto":
        return [name.strip() for name in requested.split(",") if name.strip()]
    if device.type == "cuda":
        return ["float32", "float16", "bfloat16"]
    return ["float32"]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in (torch.float16, torch.bfloat16):
        return 4e-2, 4e-2
    return 2e-4, 2e-4


def rand_for_gate(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, gate: OpCase) -> torch.Tensor:
    if gate.name == "identity":
        return torch.rand(shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


def make_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    *,
    gate: OpCase | None,
    layout: str,
) -> torch.Tensor:
    if layout == "strided_last" and shape[-1] > 1:
        base_shape = (*shape[:-1], shape[-1] * 2)
        base = rand_for_gate(base_shape, dtype, device, gate) if gate else torch.randn(base_shape, dtype=dtype, device=device)
        return base[..., ::2]
    if layout == "transpose" and len(shape) >= 2:
        base_shape = (*shape[:-2], shape[-1], shape[-2])
        base = rand_for_gate(base_shape, dtype, device, gate) if gate else torch.randn(base_shape, dtype=dtype, device=device)
        return base.transpose(-1, -2)
    return rand_for_gate(shape, dtype, device, gate) if gate else torch.randn(shape, dtype=dtype, device=device)


def broadcast_shapes(shape: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(shape) == 1:
        return (1,), shape, shape, (1,)
    gate_shape = (*shape[:-1], 1)
    tail_shape = (*([1] * (len(shape) - 1)), shape[-1])
    return gate_shape, shape, tail_shape, gate_shape


def make_inputs(workload: Workload) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if workload.layout == "broadcast":
        logits_shape, a_shape, b_shape, residual_shape = broadcast_shapes(workload.shape)
        logits = rand_for_gate(logits_shape, workload.dtype, workload.device, workload.gate)
        a = torch.randn(a_shape, dtype=workload.dtype, device=workload.device)
        b = torch.randn(b_shape, dtype=workload.dtype, device=workload.device)
        residual = torch.randn(residual_shape, dtype=workload.dtype, device=workload.device)
        return logits, a, b, residual

    logits = make_tensor(workload.shape, workload.dtype, workload.device, gate=workload.gate, layout=workload.layout)
    a = make_tensor(workload.shape, workload.dtype, workload.device, gate=None, layout=workload.layout)
    b = make_tensor(workload.shape, workload.dtype, workload.device, gate=None, layout=workload.layout)
    residual = make_tensor(workload.shape, workload.dtype, workload.device, gate=None, layout=workload.layout)
    return logits, a, b, residual


def reference(workload: Workload, logits, a, b, residual):
    return residual + torch.lerp(b, workload.activation.fn(a), workload.gate.fn(logits))


def tif_fn(workload: Workload, backend: str):
    def fn(logits, a, b, residual):
        return fused_gated_residual(
            logits,
            a,
            b,
            residual,
            activation=workload.activation.arg,
            activation_kwargs=workload.activation.kwargs,
            gate=workload.gate.arg,
            gate_kwargs=workload.gate.kwargs,
            backend=backend,
        )

    return fn


def compiled_or_none(fn):
    try:
        if hasattr(torch.compiler, "reset"):
            torch.compiler.reset()
        return torch.compile(fn, fullgraph=True)
    except Exception:
        return None


def clone_for_grad(inputs: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    return [x.detach().clone().requires_grad_(True) for x in inputs]


def forward_correct(fn, ref_fn, inputs: tuple[torch.Tensor, ...], dtype: torch.dtype) -> tuple[bool, str]:
    try:
        actual = fn(*inputs)
        expected = ref_fn(*inputs)
        atol, rtol = tolerance(dtype)
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            max_diff = (actual - expected).abs().max().detach()
            return False, f"forward mismatch max_diff={float(max_diff)}"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def backward_correct(fn, ref_fn, inputs: tuple[torch.Tensor, ...], dtype: torch.dtype) -> tuple[bool, str]:
    try:
        actual_inputs = clone_for_grad(inputs)
        expected_inputs = clone_for_grad(inputs)
        fn(*actual_inputs).sum().backward()
        ref_fn(*expected_inputs).sum().backward()
        atol, rtol = tolerance(dtype)
        for idx, (actual, expected) in enumerate(zip(actual_inputs, expected_inputs)):
            if actual.grad is None or expected.grad is None:
                return False, f"missing grad for input {idx}"
            if not torch.allclose(actual.grad, expected.grad, atol=atol, rtol=rtol):
                max_diff = (actual.grad - expected.grad).abs().max().detach()
                return False, f"grad mismatch input={idx} max_diff={float(max_diff)}"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def measure(
    fn,
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
    *,
    autograd: bool,
    iters: int,
    warmup: int,
) -> tuple[float, float | None]:
    for _ in range(warmup):
        if autograd:
            run_inputs = clone_for_grad(inputs)
            fn(*run_inputs).sum().backward()
        else:
            fn(*inputs)
    sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(iters):
        run_inputs = clone_for_grad(inputs) if autograd else inputs
        start = time.perf_counter()
        out = fn(*run_inputs)
        if autograd:
            out.sum().backward()
        sync(device)
        times.append((time.perf_counter() - start) * 1000)

    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return statistics.median(times), peak_mb


def row_for_skip(workload_id: int, workload: Workload, backend: str, autograd: bool, error: str) -> BenchRow:
    return BenchRow(
        workload_id=workload_id,
        backend=backend,
        selected_backend="skipped",
        reason=error,
        activation=workload.activation.name,
        gate=workload.gate.name,
        layout=workload.layout,
        device=workload.device.type,
        dtype=str(workload.dtype).replace("torch.", ""),
        shape=list(workload.shape),
        compiled=backend.startswith("compile"),
        triton=False,
        autograd=autograd,
        latency_ms=None,
        speedup_vs_eager=None,
        peak_memory_mb=None,
        correct=False,
        skipped=True,
        error=error,
    )


def measure_row(
    workload_id: int,
    workload: Workload,
    backend_name: str,
    fn,
    ref_fn,
    inputs: tuple[torch.Tensor, ...],
    *,
    baseline_ms: float,
    autograd: bool,
    iters: int,
    warmup: int,
    selected_backend: str = "",
    reason: str = "",
    triton: bool = False,
) -> BenchRow:
    if backend_name in {"tif_auto", "tif_torch", "tif_triton_forced"}:
        backend = backend_name.removeprefix("tif_")
        if backend == "triton_forced":
            backend = "triton"
        probe_inputs = tuple(x.detach().clone().requires_grad_(autograd) for x in inputs) if autograd else inputs
        try:
            _, report = fused_gated_residual(
                *probe_inputs,
                activation=workload.activation.arg,
                activation_kwargs=workload.activation.kwargs,
                gate=workload.gate.arg,
                gate_kwargs=workload.gate.kwargs,
                backend=backend,
                debug=True,
            )
            selected_backend = report.backend
            reason = report.reason
            triton = report.triton_used
        except Exception as exc:
            selected_backend = selected_backend or "unknown"
            reason = reason or f"{type(exc).__name__}: {exc}"

    correct, error = backward_correct(fn, ref_fn, inputs, workload.dtype) if autograd else forward_correct(fn, ref_fn, inputs, workload.dtype)
    latency_ms = None
    peak_mb = None
    if correct:
        latency_ms, peak_mb = measure(fn, inputs, workload.device, autograd=autograd, iters=iters, warmup=warmup)
    return BenchRow(
        workload_id=workload_id,
        backend=backend_name,
        selected_backend=selected_backend or backend_name,
        reason=reason,
        activation=workload.activation.name,
        gate=workload.gate.name,
        layout=workload.layout,
        device=workload.device.type,
        dtype=str(workload.dtype).replace("torch.", ""),
        shape=list(workload.shape),
        compiled=backend_name.startswith("torch_compile"),
        triton=triton,
        autograd=autograd,
        latency_ms=latency_ms,
        speedup_vs_eager=(baseline_ms / latency_ms) if latency_ms and latency_ms > 0 else None,
        peak_memory_mb=peak_mb,
        correct=correct,
        skipped=False,
        error=error,
    )


def iter_workloads(args) -> list[Workload]:
    acts = parse_names(args.activations, activation_cases())
    gates = parse_names(args.gates, gate_cases())
    layouts = [name.strip() for name in args.layouts.split(",") if name.strip()]
    workloads = []
    for device in devices_for_arg(args.device):
        for dtype_name in dtype_names_for_device(device, args.dtypes):
            dtype = getattr(torch, dtype_name)
            for shape in shapes_for_preset(args.preset):
                for layout in layouts:
                    if layout == "transpose" and len(shape) < 2:
                        continue
                    for activation in acts:
                        for gate in gates:
                            workloads.append(Workload(shape, dtype, device, layout, activation, gate))
                            if args.max_workloads and len(workloads) >= args.max_workloads:
                                return workloads
    return workloads


def main() -> int:
    parser = argparse.ArgumentParser(description="Hard benchmark matrix for TIF/TELSE fused conditional training ops.")
    parser.add_argument("--preset", choices=("smoke", "standard", "hard"), default="standard")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or all")
    parser.add_argument("--dtypes", default="auto", help="auto or comma-separated dtype names")
    parser.add_argument("--activations", default="silu,swish,relu,gelu,mish,leaky_relu,hardswish,softplus,custom_relu_tanh")
    parser.add_argument("--gates", default="sigmoid,hard_sigmoid,clamp01")
    parser.add_argument("--layouts", default="contiguous,strided_last,transpose,broadcast")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--max-workloads", type=int, default=0)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--include-forced-triton", action="store_true")
    parser.add_argument("--record-skips", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--fail-on-correctness", action="store_true")
    args = parser.parse_args()

    rows: list[BenchRow] = []
    workloads = iter_workloads(args)

    for workload_id, workload in enumerate(workloads, start=1):
        inputs = make_inputs(workload)

        def ref_fn(logits, a, b, residual):
            return reference(workload, logits, a, b, residual)

        eager_ms, eager_peak = measure(
            ref_fn,
            inputs,
            workload.device,
            autograd=False,
            iters=args.iters,
            warmup=args.warmup,
        )
        rows.append(
            BenchRow(
                workload_id=workload_id,
                backend="torch_eager_ref",
                selected_backend="torch_eager_ref",
                reason="baseline",
                activation=workload.activation.name,
                gate=workload.gate.name,
                layout=workload.layout,
                device=workload.device.type,
                dtype=str(workload.dtype).replace("torch.", ""),
                shape=list(workload.shape),
                compiled=False,
                triton=False,
                autograd=False,
                latency_ms=eager_ms,
                speedup_vs_eager=1.0,
                peak_memory_mb=eager_peak,
                correct=True,
                skipped=False,
                error="",
            )
        )

        candidates: list[tuple[str, object, bool, str, bool]] = []
        if not args.no_compile:
            compiled_ref = compiled_or_none(ref_fn)
            if compiled_ref is not None:
                candidates.append(("torch_compile_ref", compiled_ref, True, "torch_compile_ref", False))
            elif args.record_skips:
                rows.append(row_for_skip(workload_id, workload, "torch_compile_ref", False, "torch.compile unavailable or failed"))

        for backend in ("auto", "torch"):
            fn = tif_fn(workload, backend)
            try:
                out, report = fused_gated_residual(
                    *inputs,
                    activation=workload.activation.arg,
                    activation_kwargs=workload.activation.kwargs,
                    gate=workload.gate.arg,
                    gate_kwargs=workload.gate.kwargs,
                    backend=backend,
                    debug=True,
                )
                del out
                candidates.append((f"tif_{backend}", fn, False, report.backend, report.triton_used))
            except Exception as exc:
                if args.record_skips:
                    rows.append(row_for_skip(workload_id, workload, f"tif_{backend}", False, f"{type(exc).__name__}: {exc}"))

        if args.include_forced_triton:
            fn = tif_fn(workload, "triton")
            try:
                out, report = fused_gated_residual(
                    *inputs,
                    activation=workload.activation.arg,
                    activation_kwargs=workload.activation.kwargs,
                    gate=workload.gate.arg,
                    gate_kwargs=workload.gate.kwargs,
                    backend="triton",
                    debug=True,
                )
                del out
                candidates.append(("tif_triton_forced", fn, False, report.backend, report.triton_used))
            except Exception as exc:
                if args.record_skips:
                    rows.append(row_for_skip(workload_id, workload, "tif_triton_forced", False, f"{type(exc).__name__}: {exc}"))

        for backend_name, fn, _compiled, selected_backend, triton_used in candidates:
            rows.append(
                measure_row(
                    workload_id,
                    workload,
                    backend_name,
                    fn,
                    ref_fn,
                    inputs,
                    baseline_ms=eager_ms,
                    autograd=False,
                    iters=args.iters,
                    warmup=args.warmup,
                    selected_backend=selected_backend,
                    triton=triton_used,
                )
            )

        if not args.no_backward:
            eager_bw_ms, eager_bw_peak = measure(
                ref_fn,
                inputs,
                workload.device,
                autograd=True,
                iters=max(3, math.ceil(args.iters / 3)),
                warmup=max(2, math.ceil(args.warmup / 2)),
            )
            rows.append(
                BenchRow(
                    workload_id=workload_id,
                    backend="torch_eager_ref",
                    selected_backend="torch_eager_ref",
                    reason="baseline backward",
                    activation=workload.activation.name,
                    gate=workload.gate.name,
                    layout=workload.layout,
                    device=workload.device.type,
                    dtype=str(workload.dtype).replace("torch.", ""),
                    shape=list(workload.shape),
                    compiled=False,
                    triton=False,
                    autograd=True,
                    latency_ms=eager_bw_ms,
                    speedup_vs_eager=1.0,
                    peak_memory_mb=eager_bw_peak,
                    correct=True,
                    skipped=False,
                    error="",
                )
            )
            for backend_name, fn, _compiled, selected_backend, triton_used in candidates:
                rows.append(
                    measure_row(
                        workload_id,
                        workload,
                        backend_name,
                        fn,
                        ref_fn,
                        inputs,
                        baseline_ms=eager_bw_ms,
                        autograd=True,
                        iters=max(3, math.ceil(args.iters / 3)),
                        warmup=max(2, math.ceil(args.warmup / 2)),
                        selected_backend=selected_backend,
                        triton=triton_used,
                    )
                )

    failures = [row for row in rows if not row.skipped and not row.correct]
    measured = [row for row in rows if not row.skipped and row.latency_ms is not None]
    winners = [
        row
        for row in measured
        if row.backend != "torch_eager_ref" and row.correct and row.speedup_vs_eager is not None and row.speedup_vs_eager > 1.0
    ]
    best = max(winners, key=lambda row: row.speedup_vs_eager or 0, default=None)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print(
            "id backend selected activation gate layout device dtype shape autograd "
            "latency_ms speedup peak_mb correct triton skipped reason"
        )
        for row in rows:
            latency = "NA" if row.latency_ms is None else f"{row.latency_ms:.4f}"
            speedup = "NA" if row.speedup_vs_eager is None else f"{row.speedup_vs_eager:.3f}"
            peak = "NA" if row.peak_memory_mb is None else f"{row.peak_memory_mb:.1f}"
            reason = row.reason or row.error
            print(
                f"{row.workload_id} {row.backend} {row.selected_backend} {row.activation} {row.gate} "
                f"{row.layout} {row.device} {row.dtype} {row.shape} {row.autograd} "
                f"{latency} {speedup} {peak} {row.correct} {row.triton} {row.skipped} {reason}"
            )
        print(
            f"SUMMARY workloads={len(workloads)} rows={len(rows)} measured={len(measured)} "
            f"failures={len(failures)} winners={len(winners)}"
        )
        if best is not None:
            print(
                f"BEST speedup={best.speedup_vs_eager:.3f} backend={best.backend} "
                f"selected={best.selected_backend} activation={best.activation} gate={best.gate} "
                f"layout={best.layout} dtype={best.dtype} shape={best.shape} autograd={best.autograd}"
            )
        if failures:
            print("CORRECTNESS_FAIL")
        elif winners:
            print("HARD_BENCHMARK_PASS")
        else:
            print("HARD_BENCHMARK_NO_SPEEDUP")

    return 1 if failures and args.fail_on_correctness else 0


if __name__ == "__main__":
    raise SystemExit(main())
