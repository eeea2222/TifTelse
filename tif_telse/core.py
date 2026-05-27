from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any, Literal

import numpy as np

from .backends import detect_capabilities

try:
    import torch
except Exception:  # pragma: no cover - import-time optionality for docs tooling
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TElse:
    value: Any


def telse(value: Any) -> TElse:
    return TElse(value)


def _unwrap_else(value: Any) -> Any:
    return value.value if isinstance(value, TElse) else value


def _resolve(value: Any) -> Any:
    return value() if callable(value) else value


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _debug_info(backend: str, reason: str, *, device: Any | None = None) -> dict[str, Any]:
    return {
        "backend": backend,
        "reason": reason,
        "capabilities": detect_capabilities(torch, device=device),
    }


def _torch_lerp_endpoint(value: Any, gate: Any, name: str) -> Any:
    if _is_torch_tensor(value):
        return value
    if isinstance(value, Number):
        return torch.as_tensor(value, dtype=gate.dtype, device=gate.device)
    raise TypeError(
        f"soft_tif with a torch tensor gate requires tensor or numeric scalar {name}; "
        "non-scalar arrays are not moved across devices implicitly."
    )


def tif(
    condition: Any,
    then_value: Any,
    else_branch: Any,
    *,
    mode: Literal["auto", "scalar", "hard_mask", "soft"] = "auto",
    strict: bool = True,
    debug: bool = False,
) -> Any:
    """Tensorized conditional value selection.

    Scalar Python booleans are real lazy branches. Tensor conditions stay tensor
    native and never become Python booleans.
    """

    else_value = _unwrap_else(else_branch)

    if isinstance(condition, bool):
        out = _resolve(then_value if condition else else_value)
        report = _debug_info("python_scalar_bool", "Python bool uses true lazy branch semantics")
        return (out, report) if debug else out

    if isinstance(condition, Number) and not isinstance(condition, bool):
        if mode != "scalar" and strict:
            raise TypeError(
                "Numeric scalar conditions are ambiguous. Use mode='scalar' "
                "to opt into Python truthiness explicitly."
            )
        selected = bool(condition)
        out = _resolve(then_value if selected else else_value)
        report = _debug_info("python_scalar_numeric", "Explicit scalar mode uses Python truthiness")
        return (out, report) if debug else out

    if isinstance(condition, np.ndarray):
        then_resolved = _resolve(then_value)
        else_resolved = _resolve(else_value)
        if condition.dtype == np.bool_ or mode in ("hard_mask", "auto"):
            out = np.where(condition, then_resolved, else_resolved)
        elif mode == "soft":
            out = condition * then_resolved + (1 - condition) * else_resolved
        else:
            raise TypeError("NumPy conditions require boolean masks or mode='soft'.")
        report = _debug_info("numpy", "NumPy condition uses NumPy vectorized selection/blending")
        return (out, report) if debug else out

    if _is_torch_tensor(condition):
        if condition.dtype == torch.bool:
            then_resolved = _resolve(then_value)
            else_resolved = _resolve(else_value)
            out = torch.where(condition, then_resolved, else_resolved)
            report = _debug_info("torch_where", "Boolean tensor condition uses torch.where", device=condition.device)
            return (out, report) if debug else out

        if torch.is_floating_point(condition):
            if mode == "soft":
                out = soft_tif(condition, _resolve(then_value), _resolve(else_value))
                report = _debug_info(
                    "torch_soft_lerp",
                    "Floating tensor condition uses differentiable torch.lerp blending",
                    device=condition.device,
                )
                return (out, report) if debug else out
            if mode == "hard_mask":
                then_resolved = _resolve(then_value)
                else_resolved = _resolve(else_value)
                out = torch.where(condition != 0, then_resolved, else_resolved)
                report = _debug_info(
                    "torch_hard_float_mask",
                    "Explicit hard_mask mode uses nonzero tensor mask with torch.where",
                    device=condition.device,
                )
                return (out, report) if debug else out
            raise TypeError(
                "Floating tensor conditions are ambiguous. Use mode='soft' "
                "for differentiable gates or mode='hard_mask' for nonzero masks."
            )

        raise TypeError("Torch tensor conditions must be bool or floating point.")

    raise TypeError(f"Unsupported condition type: {type(condition).__name__}")


def soft_tif(gate: Any, then_value: Any, else_value: Any, *, clamp: bool = False) -> Any:
    if _is_torch_tensor(gate):
        effective_gate = gate.clamp(0, 1) if clamp else gate
        then_endpoint = _torch_lerp_endpoint(then_value, effective_gate, "then_value")
        else_endpoint = _torch_lerp_endpoint(else_value, effective_gate, "else_value")
        return torch.lerp(else_endpoint, then_endpoint, effective_gate)

    if isinstance(gate, np.ndarray):
        effective_gate = np.clip(gate, 0, 1) if clamp else gate
        return else_value + effective_gate * (then_value - else_value)

    raise TypeError("soft_tif requires a PyTorch tensor or NumPy array gate.")


def straight_through_tif(
    gate: Any,
    then_value: Any,
    else_value: Any,
    *,
    threshold: float = 0.5,
) -> Any:
    if not _is_torch_tensor(gate):
        raise TypeError("straight_through_tif is currently implemented for PyTorch tensors only.")
    hard = (gate >= threshold).to(dtype=gate.dtype)
    ste_gate = hard - gate.detach() + gate
    return soft_tif(ste_gate, then_value, else_value)
