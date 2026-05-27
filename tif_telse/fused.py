from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

from .backends import (
    BackendDecision,
    RuntimeCapabilities,
    _TRITON_AVAILABLE,
    _TRITON_OP_AVAILABLE,
    detect_capabilities,
    torch_compile_active,
)
from .ops import ResolvedOp, apply_op, resolve_activation, resolve_gate

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BackendReport(BackendDecision):
    backend: str
    triton_used: bool
    reason: str
    capabilities: RuntimeCapabilities
    activation: str
    gate: str
    custom_activation: bool
    custom_gate: bool


def _torch_gated_residual(
    logits_or_gate: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor | None,
    *,
    activation: ResolvedOp,
    gate: ResolvedOp,
) -> torch.Tensor:
    m = apply_op(gate, logits_or_gate)
    branch = apply_op(activation, a)
    out = torch.lerp(b, branch, m)
    return out if residual is None else residual + out


def _triton_kernel_eligibility(
    logits: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor | None,
    activation: ResolvedOp,
    gate: ResolvedOp,
) -> tuple[bool, str]:
    if not _TRITON_AVAILABLE:
        return False, "triton is not importable"
    if not _TRITON_OP_AVAILABLE:
        return False, "torch.library Triton integration is unavailable"
    if _triton_gate_silu_residual is None:
        return False, "triton fused op was not registered"
    if residual is None:
        return False, "triton path requires a tensor residual for the milestone kernel"
    tensors = (logits, a, b, residual)
    if not all(t.is_cuda for t in tensors):
        return False, "triton path is CUDA-only"
    if not all(t.shape == logits.shape for t in tensors):
        return False, "triton path requires identical shapes"
    if not all(t.is_contiguous() for t in tensors):
        return False, "triton path requires contiguous tensors"
    logits_dtype = logits.dtype
    if not all(t.dtype == logits_dtype for t in (a, b, residual)):
        return False, "triton path requires matching dtypes"
    if logits_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False, "triton path supports fp16, bf16, and fp32"
    if activation.custom or gate.custom:
        return False, "triton path requires registry-known activation and gate"
    if activation.name != "silu" or gate.name != "sigmoid":
        return False, "triton milestone kernel supports activation='silu' and gate='sigmoid'"
    return True, "eligible"


def _build_report(
    backend: str,
    triton_used: bool,
    reason: str,
    device: object,
    activation: ResolvedOp,
    gate: ResolvedOp,
) -> BackendReport:
    return BackendReport(
        backend=backend,
        triton_used=triton_used,
        reason=reason,
        capabilities=detect_capabilities(torch, device=device),
        activation=activation.name,
        gate=gate.name,
        custom_activation=activation.custom,
        custom_gate=gate.custom,
    )


def _select_fused_backend(
    logits: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor | None,
    *,
    activation: ResolvedOp,
    gate: ResolvedOp,
    requested: str,
    debug: bool,
) -> tuple[bool, BackendReport | None]:
    def report(backend: str, triton_used: bool, reason: str) -> BackendReport | None:
        if not debug:
            return None
        return _build_report(backend, triton_used, reason, logits.device, activation, gate)

    if requested == "torch":
        return False, report("torch_composition", False, "forced torch backend")

    eligible, reason = _triton_kernel_eligibility(logits, a, b, residual, activation, gate)
    if requested == "triton":
        if not eligible:
            if logits.device.type != "cuda":
                raise RuntimeError(
                    "Triton backend is CUDA-only; use backend='torch' or backend='auto' for CPU."
                )
            raise RuntimeError(f"Triton backend requested but unavailable: {reason}")
        return True, report(
            "triton_fused_gate_silu_residual",
            True,
            "forced triton backend",
        )

    if requested != "auto":
        raise ValueError("backend must be 'auto', 'torch', or 'triton'.")

    if torch_compile_active(torch):
        return False, report(
            "torch_composition",
            False,
            "auto selected torch because torch.compile/Inductor can fuse the composition",
        )

    if not eligible:
        return False, report("torch_composition", False, reason)

    n_elements = logits.numel()
    if n_elements < 1_000_000:
        return False, report(
            "torch_composition",
            False,
            "auto selected torch for small tensors; Triton launch overhead wins less often here",
        )

    if torch.is_grad_enabled():
        needs_backward = any(t.requires_grad for t in (logits, a, b, residual) if t is not None)
        if needs_backward:
            if logits.dtype in (torch.float16, torch.bfloat16) and n_elements < 4_000_000:
                return False, report(
                    "torch_composition",
                    False,
                    "auto selected torch for mid-size low-precision autograd; measured Triton backward is weaker here",
                )
            if logits.dtype == torch.float32 and n_elements >= 4_000_000:
                return False, report(
                    "torch_composition",
                    False,
                    "auto selected torch for large fp32 autograd; measured Triton backward is weaker here",
                )

    return True, report(
        "triton_fused_gate_silu_residual",
        True,
        "auto selected Triton fused kernel for eligible large contiguous CUDA tensors",
    )


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _fused_gate_silu_residual_kernel(
        logits_ptr,
        a_ptr,
        b_ptr,
        residual_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.sigmoid(logits)
        silu = a * tl.sigmoid(a)
        out = residual + b + gate * (silu - b)
        tl.store(out_ptr + offsets, out, mask=mask)


    @torch.library.triton_op("tif_telse::fused_gate_silu_residual", mutates_args={})
    def _triton_gate_silu_residual(
        logits: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(a)
        n_elements = a.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        torch.library.wrap_triton(_fused_gate_silu_residual_kernel)[grid](
            logits,
            a,
            b,
            residual,
            out,
            n_elements,
        )
        return out


    def _triton_backward(ctx, grad: torch.Tensor):
        logits, a, b = ctx.saved_tensors
        gate = torch.sigmoid(logits)
        sig_a = torch.sigmoid(a)
        silu = a * sig_a
        grad_logits = grad * (silu - b) * gate * (1 - gate)
        grad_a = grad * gate * (sig_a * (1 + a * (1 - sig_a)))
        grad_b = grad * (1 - gate)
        grad_residual = grad
        return grad_logits, grad_a, grad_b, grad_residual


    def _triton_setup_context(ctx, inputs, output):
        logits, a, b, _residual = inputs
        ctx.save_for_backward(logits, a, b)


    _triton_gate_silu_residual.register_autograd(
        _triton_backward,
        setup_context=_triton_setup_context,
    )

else:
    _triton_gate_silu_residual = None


def fused_gated_residual(
    logits_or_gate: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor | None = None,
    *,
    activation: str | Callable[..., torch.Tensor] = "silu",
    gate: str | Callable[..., torch.Tensor] = "sigmoid",
    activation_kwargs: dict[str, object] | None = None,
    gate_kwargs: dict[str, object] | None = None,
    backend: Literal["auto", "torch", "triton"] = "auto",
    debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, BackendReport]:
    """Fused training conditional primitive.

    The Triton backend is intentionally narrow. Unsupported cases use the
    PyTorch composition so autograd and torch.compile friendliness are retained.
    """

    resolved_activation = resolve_activation(activation, activation_kwargs)
    resolved_gate = resolve_gate(gate, gate_kwargs)

    use_triton, decision = _select_fused_backend(
        logits_or_gate,
        a,
        b,
        residual,
        activation=resolved_activation,
        gate=resolved_gate,
        requested=backend,
        debug=debug,
    )

    if use_triton and _triton_gate_silu_residual is not None:
        out = _triton_gate_silu_residual(logits_or_gate, a, b, residual)  # type: ignore[arg-type]
        return (out, decision) if debug else out

    out = _torch_gated_residual(
        logits_or_gate,
        a,
        b,
        residual,
        activation=resolved_activation,
        gate=resolved_gate,
    )
    return (out, decision) if debug else out
