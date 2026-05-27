"""Fused gated residual: the AI-training pattern. CPU + CUDA + debug report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F

from tif_telse import fused_gated_residual


def reference(logits, a, b, residual):
    return residual + torch.lerp(b, F.silu(a), torch.sigmoid(logits))


def run(device: str) -> None:
    shape = (16, 128, 256)
    dtype = torch.float32
    logits = torch.randn(shape, device=device, dtype=dtype)
    a = torch.randn(shape, device=device, dtype=dtype)
    b = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn(shape, device=device, dtype=dtype)

    y, report = fused_gated_residual(
        logits, a, b, residual,
        activation="silu", gate="sigmoid",
        backend="auto", debug=True,
    )
    ref = reference(logits, a, b, residual)
    ok = torch.allclose(y, ref, atol=1e-4, rtol=1e-4)
    print(f"[{device}] backend = {report.backend}")
    print(f"[{device}] triton  = {report.triton_used}")
    print(f"[{device}] reason  = {report.reason}")
    print(f"[{device}] matches reference = {ok}")


def main() -> None:
    run("cpu")
    if torch.cuda.is_available():
        run("cuda")
    else:
        print("[cuda] skipped (no CUDA device)")


if __name__ == "__main__":
    main()
