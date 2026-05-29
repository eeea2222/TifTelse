"""Register a user-defined activation and use it through the fused primitive."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from tif_telse import (
    available_activations,
    fused_gated_residual,
    register_activation,
)


def swiglu_like(x: torch.Tensor) -> torch.Tensor:
    # Toy example: SiLU scaled by a learnable-free factor.
    return torch.nn.functional.silu(x) * 0.5


def main() -> None:
    register_activation("swiglu_like", swiglu_like, aliases=("sgl",))

    print(f"registered: 'swiglu_like' is now in {'swiglu_like' in available_activations()}")

    logits = torch.randn(8)
    a = torch.randn(8)
    b = torch.randn(8)
    residual = torch.randn(8)

    y, report = fused_gated_residual(
        logits, a, b, residual,
        activation="sgl",          # alias works
        gate="sigmoid",
        debug=True,
    )
    print(f"activation = {report.activation}, backend = {report.backend}")
    print(f"output shape = {y.shape}")


if __name__ == "__main__":
    main()
