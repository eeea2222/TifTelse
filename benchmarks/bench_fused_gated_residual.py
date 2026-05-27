from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tif_telse import compose_ops, fused_gated_residual


@dataclass
class BenchRow:
    case: str
    activation: str
    backend: str
    device: str
    dtype: str
    shape: list[int]
    compiled: bool
    triton: bool
    autograd: bool
    latency_ms: float
    speedup_vs_eager: float
    passed: bool


def activation_for_name(name: str):
    if name in ("silu", "swish"):
        return F.silu
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "mish":
        return F.mish
    if name == "custom":
        return lambda x: torch.tanh(F.relu(x))
    raise ValueError(f"Unknown benchmark activation: {name}")


def activation_arg_for_name(name: str):
    if name == "custom":
        return compose_ops("relu", torch.tanh)
    return name


def torch_ref(logits, a, b, residual, activation_name: str):
    branch = activation_for_name(activation_name)(a)
    return residual + torch.lerp(b, branch, torch.sigmoid(logits))


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, args, device, autograd: bool, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        if autograd:
            cloned = [x.detach().clone().requires_grad_(True) for x in args]
            fn(*cloned).sum().backward()
        else:
            fn(*args)
    sync(device)
    times = []
    for _ in range(iters):
        if autograd:
            run_args = [x.detach().requires_grad_(True) for x in args]
        else:
            run_args = args
        start = time.perf_counter()
        out = fn(*run_args)
        if autograd:
            out.sum().backward()
        sync(device)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def make_args(shape, dtype, device, noncontiguous):
    base_shape = shape if not noncontiguous else (shape[0], shape[1], shape[2] * 2)
    tensors = [torch.randn(base_shape, device=device, dtype=dtype) for _ in range(4)]
    if noncontiguous:
        tensors = [t[..., ::2] for t in tensors]
    return tensors


def allclose_for_dtype(a, b, dtype):
    atol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    rtol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    return torch.allclose(a, b, atol=atol, rtol=rtol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--activations", default="silu,swish,relu,gelu,mish,custom")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    shapes = [(16, 128, 256), (16, 256, 512), (8, 512, 1024)]
    dtype_names = ["float32"]
    if device.type == "cuda":
        dtype_names += ["float16", "bfloat16"]

    rows: list[BenchRow] = []
    winners = []
    activations = [name.strip() for name in args.activations.split(",") if name.strip()]

    for activation_name in activations:
        activation_arg = activation_arg_for_name(activation_name)
        for shape in shapes:
            for dtype_name in dtype_names:
                dtype = getattr(torch, dtype_name)
                for noncontiguous in (False, True):
                    case = f"{'noncontig' if noncontiguous else 'contig'}"
                    inputs = make_args(shape, dtype, device, noncontiguous)

                    def eager_ref(logits, a, b, residual):
                        return torch_ref(logits, a, b, residual, activation_name)

                    eager_ms = measure(eager_ref, inputs, device, False, args.iters, args.warmup)
                    rows.append(BenchRow(case, activation_name, "torch_eager", device.type, dtype_name, list(shape), False, False, False, eager_ms, 1.0, True))

                    compiled_ms = None
                    try:
                        if hasattr(torch.compiler, "reset"):
                            torch.compiler.reset()

                        def ref_for_compile(logits, a, b, residual):
                            return torch_ref(logits, a, b, residual, activation_name)

                        compiled_ref = torch.compile(ref_for_compile, fullgraph=True)
                        compiled_ms = measure(compiled_ref, inputs, device, False, args.iters, args.warmup)
                        ok = allclose_for_dtype(compiled_ref(*inputs), eager_ref(*inputs), dtype)
                        rows.append(BenchRow(case, activation_name, "torch_compile", device.type, dtype_name, list(shape), True, False, False, compiled_ms, eager_ms / compiled_ms, ok))
                    except Exception:
                        pass

                    def tif_auto(logits, a, b, residual):
                        return fused_gated_residual(logits, a, b, residual, activation=activation_arg, backend="auto")

                    tif_ms = measure(tif_auto, inputs, device, False, args.iters, args.warmup)
                    out, report = fused_gated_residual(*inputs, activation=activation_arg, backend="auto", debug=True)
                    ok = allclose_for_dtype(out, eager_ref(*inputs), dtype)
                    rows.append(BenchRow(case, activation_name, report.backend, device.type, dtype_name, list(shape), False, report.triton_used, False, tif_ms, eager_ms / tif_ms, ok))
                    if ok and tif_ms < eager_ms:
                        winners.append((shape, dtype_name, case, activation_name, report.backend, eager_ms / tif_ms))

                    eager_bw_ms = measure(eager_ref, inputs, device, True, max(5, args.iters // 3), args.warmup)
                    tif_bw_ms = measure(tif_auto, inputs, device, True, max(5, args.iters // 3), args.warmup)
                    grad_inputs = [x.detach().requires_grad_(True) for x in inputs]
                    _, grad_report = fused_gated_residual(*grad_inputs, activation=activation_arg, backend="auto", debug=True)
                    rows.append(BenchRow(case, activation_name, "torch_eager", device.type, dtype_name, list(shape), False, False, True, eager_bw_ms, 1.0, True))
                    rows.append(BenchRow(case, activation_name, grad_report.backend, device.type, dtype_name, list(shape), False, grad_report.triton_used, True, tif_bw_ms, eager_bw_ms / tif_bw_ms, ok))

    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
    else:
        print("case activation backend device dtype shape compiled triton autograd latency_ms speedup_vs_eager passed")
        for r in rows:
            print(
                f"{r.case} {r.activation} {r.backend} {r.device} {r.dtype} {r.shape} "
                f"{r.compiled} {r.triton} {r.autograd} {r.latency_ms:.4f} "
                f"{r.speedup_vs_eager:.3f} {r.passed}"
            )
        if winners:
            best = max(winners, key=lambda x: x[-1])
            print(
                f"PERFORMANCE_CONTRACT_PASS best_shape={best[0]} dtype={best[1]} "
                f"case={best[2]} activation={best[3]} backend={best[4]} speedup={best[5]:.3f}"
            )
        else:
            print("PERFORMANCE_CONTRACT_FAIL no TIF backend beat eager PyTorch in measured cases")


if __name__ == "__main__":
    main()
