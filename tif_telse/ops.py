from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

TensorOp = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class OpSpec:
    name: str
    fn: Callable[..., torch.Tensor]
    builtin: bool = True


@dataclass(frozen=True)
class ResolvedOp:
    name: str
    fn: Callable[..., torch.Tensor]
    kwargs: dict[str, object]
    builtin: bool
    custom: bool


_ACTIVATIONS: dict[str, OpSpec] = {}
_GATES: dict[str, OpSpec] = {}


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("Operation names cannot be empty.")
    return normalized


def _register(registry: dict[str, OpSpec], name: str, fn: Callable[..., torch.Tensor], aliases: tuple[str, ...]) -> None:
    if not callable(fn):
        raise TypeError("Registered operation must be callable.")
    canonical = _normalize_name(name)
    spec = OpSpec(canonical, fn, builtin=True)
    for key in (canonical, *aliases):
        registry[_normalize_name(key)] = spec


def register_activation(name: str, fn: Callable[..., torch.Tensor], *, aliases: tuple[str, ...] = ()) -> None:
    _register(_ACTIVATIONS, name, fn, aliases)


def register_gate(name: str, fn: Callable[..., torch.Tensor], *, aliases: tuple[str, ...] = ()) -> None:
    _register(_GATES, name, fn, aliases)


def available_activations() -> tuple[str, ...]:
    return tuple(sorted(_ACTIVATIONS))


def available_gates() -> tuple[str, ...]:
    return tuple(sorted(_GATES))


def _callable_name(fn: Callable[..., torch.Tensor]) -> str:
    return getattr(fn, "__name__", fn.__class__.__name__)


def _resolve(registry: dict[str, OpSpec], op: str | Callable[..., torch.Tensor], kwargs: dict[str, object] | None) -> ResolvedOp:
    resolved_kwargs = dict(kwargs or {})
    if isinstance(op, str):
        key = _normalize_name(op)
        try:
            spec = registry[key]
        except KeyError as exc:
            available = ", ".join(sorted(registry))
            raise ValueError(f"Unknown operation '{op}'. Available operations: {available}") from exc
        return ResolvedOp(spec.name, spec.fn, resolved_kwargs, builtin=spec.builtin, custom=False)

    if callable(op):
        return ResolvedOp(_callable_name(op), op, resolved_kwargs, builtin=False, custom=True)

    raise TypeError("Operation must be a registered name or callable.")


def resolve_activation(op: str | Callable[..., torch.Tensor], kwargs: dict[str, object] | None = None) -> ResolvedOp:
    return _resolve(_ACTIVATIONS, op, kwargs)


def resolve_gate(op: str | Callable[..., torch.Tensor], kwargs: dict[str, object] | None = None) -> ResolvedOp:
    return _resolve(_GATES, op, kwargs)


def resolve_op(op: str | Callable[..., torch.Tensor], kwargs: dict[str, object] | None = None) -> ResolvedOp:
    if not isinstance(op, str):
        return _resolve(_ACTIVATIONS, op, kwargs)
    try:
        return resolve_activation(op, kwargs)
    except ValueError:
        return resolve_gate(op, kwargs)


def apply_op(op: ResolvedOp, x: torch.Tensor) -> torch.Tensor:
    return op.fn(x, **op.kwargs)


def compose_ops(*ops: str | Callable[..., torch.Tensor]) -> TensorOp:
    resolved = tuple(resolve_op(op) for op in ops)

    def composed(x: torch.Tensor) -> torch.Tensor:
        out = x
        for op in resolved:
            out = apply_op(op, out)
        return out

    names = "_then_".join(op.name for op in resolved) or "identity"
    composed.__name__ = f"compose_{names}"
    return composed


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1)


register_activation("identity", _identity, aliases=("none",))
register_activation("relu", F.relu)
register_activation("gelu", F.gelu)
register_activation("silu", F.silu, aliases=("swish",))
register_activation("mish", F.mish)
register_activation("elu", F.elu)
register_activation("selu", F.selu)
register_activation("leaky_relu", F.leaky_relu, aliases=("lrelu",))
register_activation("hardtanh", F.hardtanh)
register_activation("hardswish", F.hardswish)
register_activation("hardsigmoid", F.hardsigmoid)
register_activation("softplus", F.softplus)
register_activation("sigmoid", torch.sigmoid)
register_activation("tanh", torch.tanh)

register_gate("sigmoid", torch.sigmoid)
register_gate("identity", _identity, aliases=("none",))
register_gate("clamp01", _clamp01, aliases=("clamp_01",))
register_gate("hard_sigmoid", F.hardsigmoid, aliases=("hardsigmoid",))
