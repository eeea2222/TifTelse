from pathlib import Path

import numpy as np
import pytest
import torch

import tif_telse


def test_version_is_exposed_and_nonempty():
    assert isinstance(tif_telse.__version__, str)
    assert tif_telse.__version__


def test_py_typed_marker_is_shipped():
    marker = Path(tif_telse.__file__).resolve().parent / "py.typed"
    assert marker.is_file()


def test_public_all_is_importable_and_consistent():
    for name in tif_telse.__all__:
        assert hasattr(tif_telse, name), f"{name} listed in __all__ but not importable"


def test_resolvedop_and_apply_op_are_public():
    resolved = tif_telse.resolve_op("relu")
    assert isinstance(resolved, tif_telse.ResolvedOp)
    x = torch.tensor([-1.0, 2.0])
    assert torch.equal(tif_telse.apply_op(resolved, x), torch.relu(x))


def test_soft_tif_numpy_rejects_torch_endpoint():
    gate = np.array([0.25, 0.75])
    with pytest.raises(TypeError, match="not converted to NumPy implicitly"):
        tif_telse.soft_tif(gate, torch.zeros(2), np.zeros(2))


def test_soft_tif_numpy_clamp_blends_within_unit_interval():
    gate = np.array([-0.5, 0.5, 1.5])
    out = tif_telse.soft_tif(gate, 10.0, 2.0, clamp=True)
    np.testing.assert_allclose(out, [2.0, 6.0, 10.0])
