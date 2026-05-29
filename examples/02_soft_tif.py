"""Soft, differentiable gating between two tensor branches."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from tif_telse import soft_tif, straight_through_tif


def main() -> None:
    gate = torch.tensor([0.1, 0.5, 0.9], requires_grad=True)
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)

    # Soft blend: y = b + gate * (a - b)
    y = soft_tif(gate, a, b)
    print(f"soft  -> {y.detach().tolist()}")

    y.sum().backward()
    print(f"  grad(gate) = {gate.grad.tolist()}")
    print(f"  grad(a)    = {a.grad.tolist()}")
    print(f"  grad(b)    = {b.grad.tolist()}")

    # Straight-through: hard forward, soft backward.
    gate2 = torch.tensor([0.25, 0.75], requires_grad=True)
    a2 = torch.tensor([2.0, 4.0], requires_grad=True)
    b2 = torch.tensor([10.0, 20.0], requires_grad=True)
    y2 = straight_through_tif(gate2, a2, b2)
    print(f"\nSTE   -> forward {y2.detach().tolist()} (hard 0/1 selection)")
    y2.sum().backward()
    print(f"  grad(gate) = {gate2.grad.tolist()} (soft, flows through)")


if __name__ == "__main__":
    main()
