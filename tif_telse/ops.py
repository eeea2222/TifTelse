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
_BUILTIN_ACTIVATION_KEYS: set[str] = set()
_BUILTIN_GATE_KEYS: set[str] = set()


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("Operation names cannot be empty.")
    return normalized


def _register(
    registry: dict[str, OpSpec],
    builtin_keys: set[str],
    name: str,
    fn: Callable[..., torch.Tensor],
    aliases: tuple[str, ...],
    *,
    builtin: bool,
    overwrite: bool,
) -> None:
    if not callable(fn):
        raise TypeError("Registered operation must be callable.")
    canonical = _normalize_name(name)
    normalized_keys = tuple(_normalize_name(key) for key in (canonical, *aliases))
    conflicts = [key for key in normalized_keys if key in builtin_keys and not builtin]
    if conflicts:
        joined = ", ".join(sorted(conflicts))
        raise ValueError(f"Cannot override built-in operation name or alias: {joined}")
    if not overwrite:
        existing = [key for key in normalized_keys if key in registry]
        if existing:
            joined = ", ".join(sorted(existing))
            raise ValueError(f"Operation name or alias already registered: {joined}")

    spec = OpSpec(canonical, fn, builtin=builtin)
    for key in normalized_keys:
        registry[key] = spec
        if builtin:
            builtin_keys.add(key)


def _register_builtin_activation(name: str, fn: Callable[..., torch.Tensor], *, aliases: tuple[str, ...] = ()) -> None:
    _register(_ACTIVATIONS, _BUILTIN_ACTIVATION_KEYS, name, fn, aliases, builtin=True, overwrite=True)


def _register_builtin_gate(name: str, fn: Callable[..., torch.Tensor], *, aliases: tuple[str, ...] = ()) -> None:
    _register(_GATES, _BUILTIN_GATE_KEYS, name, fn, aliases, builtin=True, overwrite=True)


def register_activation(
    name: str,
    fn: Callable[..., torch.Tensor],
    *,
    aliases: tuple[str, ...] = (),
    overwrite: bool = False,
) -> None:
    _register(_ACTIVATIONS, _BUILTIN_ACTIVATION_KEYS, name, fn, aliases, builtin=False, overwrite=overwrite)


def register_gate(
    name: str,
    fn: Callable[..., torch.Tensor],
    *,
    aliases: tuple[str, ...] = (),
    overwrite: bool = False,
) -> None:
    _register(_GATES, _BUILTIN_GATE_KEYS, name, fn, aliases, builtin=False, overwrite=overwrite)


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
        return ResolvedOp(spec.name, spec.fn, resolved_kwargs, builtin=spec.builtin, custom=not spec.builtin)

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


_register_builtin_activation("identity", _identity, aliases=("none",))
_register_builtin_activation("relu", F.relu)
_register_builtin_activation("gelu", F.gelu)
_register_builtin_activation("silu", F.silu, aliases=("swish",))
_register_builtin_activation("mish", F.mish)
_register_builtin_activation("elu", F.elu)
_register_builtin_activation("selu", F.selu)
_register_builtin_activation("leaky_relu", F.leaky_relu, aliases=("lrelu",))
_register_builtin_activation("hardtanh", F.hardtanh)
_register_builtin_activation("hardswish", F.hardswish)
_register_builtin_activation("hardsigmoid", F.hardsigmoid, aliases=("hard_sigmoid",))
_register_builtin_activation("softplus", F.softplus)
_register_builtin_activation("sigmoid", torch.sigmoid)
_register_builtin_activation("tanh", torch.tanh)

_register_builtin_gate("sigmoid", torch.sigmoid)
_register_builtin_gate("identity", _identity, aliases=("none",))
_register_builtin_gate("clamp01", _clamp01, aliases=("clamp_01",))
_register_builtin_gate("hard_sigmoid", F.hardsigmoid, aliases=("hardsigmoid",))
