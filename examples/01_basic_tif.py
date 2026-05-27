"""Basic tif usage: scalar bool (lazy), bool tensor mask, numpy bool mask."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from tif_telse import telse, tif


def main() -> None:
    # 1. Scalar Python bool — only the selected callable runs.
    calls = []
    y = tif(True, lambda: calls.append("then") or "yes",
                  telse(lambda: calls.append("else") or "no"))
    print(f"scalar bool   -> {y!r} (executed: {calls})")

    # 2. Boolean torch tensor mask — equivalent to torch.where, autograd-safe.
    mask = torch.tensor([True, False, True])
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([0.0, 0.0, 0.0])
    print(f"torch mask    -> {tif(mask, a, telse(b)).tolist()}")

    # 3. NumPy bool mask.
    np_mask = np.array([True, False, True])
    print(f"numpy mask    -> {tif(np_mask, np.array([10, 20, 30]), telse(0)).tolist()}")

    # 4. Float tensor condition requires explicit mode (no silent guessing).
    gate = torch.tensor([0.2, 0.8])
    try:
        tif(gate, torch.ones(2), telse(torch.zeros(2)))
    except TypeError as exc:
        print(f"float tensor  -> raised TypeError: {exc}")


if __name__ == "__main__":
    main()
