import numpy as np
import pytest
import torch

from tif_telse import detect_capabilities, soft_tif, straight_through_tif, telse, tif


def test_scalar_bool_lazy_selected_only():
    called = []

    def yes():
        called.append("yes")
        return 3

    def no():
        called.append("no")
        raise AssertionError("unselected branch executed")

    assert tif(True, yes, telse(no)) == 3
    assert called == ["yes"]


def test_scalar_numeric_requires_explicit_mode_by_default():
    with pytest.raises(TypeError, match="Numeric scalar conditions are ambiguous"):
        tif(1, "a", telse("b"))
    assert tif(0, "a", telse("b"), mode="scalar") == "b"


def test_numpy_vectorized_where_and_soft():
    cond = np.array([True, False, True])
    np.testing.assert_array_equal(tif(cond, np.array([1, 2, 3]), telse(0)), [1, 0, 3])
    gate = np.array([0.0, 0.25, 1.0])
    np.testing.assert_allclose(tif(gate, 10.0, telse(2.0), mode="soft"), [2.0, 4.0, 10.0])


def test_torch_bool_mask_broadcast_dtype_and_grad():
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    y = torch.tensor([[10.0, 20.0]], requires_grad=True)
    mask = torch.tensor([[True, False], [False, True]])
    out = tif(mask, x, telse(y))
    expected = torch.where(mask, x, y)
    assert out.dtype == x.dtype
    assert torch.equal(out, expected)
    out.sum().backward()
    assert x.grad is not None
    assert y.grad is not None


def test_debug_report_includes_backend_reason_and_capabilities():
    out, report = tif(True, "a", telse("b"), debug=True)
    assert out == "a"
    assert report["backend"] == "python_scalar_bool"
    assert "reason" in report
    assert report["capabilities"].torch_available


def test_detect_capabilities_is_available():
    caps = detect_capabilities()
    assert caps.torch_available
    assert caps.numpy_available


def test_float_tensor_requires_explicit_mode():
    gate = torch.tensor([0.2, 0.8])
    with pytest.raises(TypeError, match="Floating tensor conditions are ambiguous"):
        tif(gate, torch.ones(2), telse(torch.zeros(2)))


def test_soft_tif_gradients_flow_through_gate_and_branches():
    gate = torch.tensor([0.25, 0.75], requires_grad=True)
    a = torch.tensor([2.0, 4.0], requires_grad=True)
    b = torch.tensor([10.0, 20.0], requires_grad=True)
    out = soft_tif(gate, a, b)
    out.sum().backward()
    assert torch.allclose(gate.grad, a.detach() - b.detach())
    assert torch.allclose(a.grad, gate.detach())
    assert torch.allclose(b.grad, 1 - gate.detach())


def test_soft_tif_accepts_numeric_scalar_torch_endpoints_without_cpu_transfer():
    gate = torch.tensor([0.25, 0.75])
    out = soft_tif(gate, 10.0, 2.0)
    assert torch.allclose(out, torch.tensor([4.0, 8.0]))
    assert out.device == gate.device


def test_soft_tif_rejects_numpy_endpoint_for_torch_gate():
    gate = torch.tensor([0.25, 0.75])
    with pytest.raises(TypeError, match="not moved across devices implicitly"):
        soft_tif(gate, np.array([1.0, 2.0]), torch.zeros(2))


def test_straight_through_forward_hard_backward_soft():
    gate = torch.tensor([0.25, 0.75], requires_grad=True)
    a = torch.tensor([2.0, 4.0], requires_grad=True)
    b = torch.tensor([10.0, 20.0], requires_grad=True)
    out = straight_through_tif(gate, a, b)
    assert torch.allclose(out.detach(), torch.tensor([10.0, 4.0]))
    out.sum().backward()
    assert torch.allclose(gate.grad, a.detach() - b.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tensor_stays_on_cuda():
    mask = torch.tensor([True, False], device="cuda")
    a = torch.ones(2, device="cuda")
    b = torch.zeros(2, device="cuda")
    out = tif(mask, a, telse(b))
    assert out.is_cuda
    assert torch.equal(out.cpu(), torch.tensor([1.0, 0.0]))


def test_compile_fullgraph_bool_mask_when_available():
    def fn(mask, a, b):
        return tif(mask, a, telse(b))

    compiled = torch.compile(fn, fullgraph=True)
    mask = torch.tensor([True, False, True])
    a = torch.ones(3)
    b = torch.zeros(3)
    assert torch.equal(compiled(mask, a, b), torch.tensor([1.0, 0.0, 1.0]))
