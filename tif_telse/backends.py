from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeCapabilities:
    torch_available: bool
    numpy_available: bool
    triton_available: bool
    triton_op_available: bool
    torch_compile_available: bool
    compile_active: bool
    cuda_available: bool
    device_type: str | None


@dataclass(frozen=True)
class BackendDecision:
    backend: str
    triton_used: bool
    reason: str
    capabilities: RuntimeCapabilities


def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401
    except Exception:
        return False
    return True


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception:
        return False
    return True


_NUMPY_AVAILABLE: bool = _has_numpy()
_TRITON_AVAILABLE: bool = _has_triton()


def _torch_compile_available(torch_mod: Any) -> bool:
    compiler = getattr(torch_mod, "compiler", None)
    return compiler is not None and hasattr(compiler, "is_compiling") and hasattr(torch_mod, "compile")


try:
    import torch as _torch_for_static_caps  # type: ignore[import-not-found]
except Exception:
    _torch_for_static_caps = None  # type: ignore[assignment]

_TORCH_AVAILABLE: bool = _torch_for_static_caps is not None
if _torch_for_static_caps is not None:
    _library = getattr(_torch_for_static_caps, "library", None)
    _TRITON_OP_AVAILABLE: bool = (
        _library is not None
        and hasattr(_library, "triton_op")
        and hasattr(_library, "wrap_triton")
    )
    _TORCH_COMPILE_AVAILABLE: bool = _torch_compile_available(_torch_for_static_caps)
else:
    _TRITON_OP_AVAILABLE = False
    _TORCH_COMPILE_AVAILABLE = False


def torch_compile_active(torch_mod: Any) -> bool:
    compiler = getattr(torch_mod, "compiler", None)
    if compiler is not None:
        for name in ("is_compiling", "is_dynamo_compiling"):
            fn = getattr(compiler, name, None)
            if fn is not None:
                try:
                    if bool(fn()):
                        return True
                except Exception:
                    pass
    dynamo = getattr(torch_mod, "_dynamo", None)
    fn = getattr(dynamo, "is_compiling", None) if dynamo is not None else None
    if fn is not None:
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def detect_capabilities(torch_mod: Any | None = None, *, device: Any | None = None) -> RuntimeCapabilities:
    if torch_mod is None:
        torch_mod = _torch_for_static_caps

    device_type = None
    if device is not None:
        device_type = getattr(device, "type", None)
        if device_type is None:
            device_type = str(device)

    compile_active = False
    cuda_available = False
    if torch_mod is not None:
        compile_active = torch_compile_active(torch_mod)
        if compile_active and device_type is not None:
            cuda_available = device_type == "cuda"
        else:
            cuda = getattr(torch_mod, "cuda", None)
            if cuda is not None:
                try:
                    cuda_available = bool(cuda.is_available())
                except Exception:
                    cuda_available = False

    return RuntimeCapabilities(
        torch_available=_TORCH_AVAILABLE,
        numpy_available=_NUMPY_AVAILABLE,
        triton_available=_TRITON_AVAILABLE,
        triton_op_available=_TRITON_OP_AVAILABLE,
        torch_compile_available=_TORCH_COMPILE_AVAILABLE,
        compile_active=compile_active,
        cuda_available=cuda_available,
        device_type=device_type,
    )
