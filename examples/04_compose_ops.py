"""Compose registered ops and raw callables into a single activation pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from tif_telse import compose_ops, fused_gated_residual


def main() -> None:
    # Mix registry names and inline callables in one pipeline.
    pipeline = compose_ops("relu", torch.tanh, lambda x: x * 0.5)
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    print(f"pipeline({x.tolist()}) = {pipeline(x).tolist()}")
    print(f"pipeline name = {pipeline.__name__}")

    # Use it as a fused activation.
    logits = torch.randn(16)
    a = torch.randn(16)
    b = torch.randn(16)
    residual = torch.randn(16)

    y, report = fused_gated_residual(
        logits, a, b, residual,
        activation=pipeline,
        debug=True,
    )
    print(f"backend = {report.backend}, custom_activation = {report.custom_activation}")
    print(f"output  = {y.shape}")


if __name__ == "__main__":
    main()
