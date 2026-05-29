from importlib.metadata import PackageNotFoundError, version

from .backends import BackendDecision, RuntimeCapabilities, detect_capabilities
from .core import TElse, soft_tif, straight_through_tif, telse, tif
from .fused import BackendReport, fused_gated_residual
from .ops import (
    ResolvedOp,
    apply_op,
    available_activations,
    available_gates,
    compose_ops,
    register_activation,
    register_gate,
    resolve_activation,
    resolve_gate,
    resolve_op,
)

try:
    __version__ = version("tif-telse")
except PackageNotFoundError:  # pragma: no cover - source checkout without install metadata
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "BackendDecision",
    "BackendReport",
    "ResolvedOp",
    "RuntimeCapabilities",
    "apply_op",
    "available_activations",
    "available_gates",
    "compose_ops",
    "detect_capabilities",
    "TElse",
    "fused_gated_residual",
    "register_activation",
    "register_gate",
    "resolve_activation",
    "resolve_gate",
    "resolve_op",
    "soft_tif",
    "straight_through_tif",
    "telse",
    "tif",
]
