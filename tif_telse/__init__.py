from .backends import BackendDecision, RuntimeCapabilities, detect_capabilities
from .core import TElse, tif, telse, soft_tif, straight_through_tif
from .fused import BackendReport, fused_gated_residual
from .ops import (
    available_activations,
    available_gates,
    compose_ops,
    register_activation,
    register_gate,
    resolve_op,
)

__all__ = [
    "BackendDecision",
    "BackendReport",
    "RuntimeCapabilities",
    "available_activations",
    "available_gates",
    "compose_ops",
    "detect_capabilities",
    "TElse",
    "fused_gated_residual",
    "register_activation",
    "register_gate",
    "resolve_op",
    "soft_tif",
    "straight_through_tif",
    "telse",
    "tif",
]
