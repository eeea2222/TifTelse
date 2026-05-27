import pytest
import torch
import torch.nn.functional as F

from tif_telse import available_activations, available_gates, compose_ops, fused_gated_residual, register_activation


def reference(logits, a, b, residual, activation_fn=F.silu, gate_fn=torch.sigmoid):
    out = torch.lerp(b, activation_fn(a), gate_fn(logits))
    return residual + out if residual is not None else out


def test_fused_torch_matches_reference_cpu():
    logits = torch.randn(16, dtype=torch.float32, requires_grad=True)
    a = torch.randn(16, dtype=torch.float32, requires_grad=True)
    b = torch.randn(16, dtype=torch.float32, requires_grad=True)
    residual = torch.randn(16, dtype=torch.float32, requires_grad=True)
    out, report = fused_gated_residual(logits, a, b, residual, backend="torch", debug=True)
    assert report.backend == "torch_composition"
    assert torch.allclose(out, reference(logits, a, b, residual))
    out.sum().backward()
    assert all(t.grad is not None for t in (logits, a, b, residual))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_forced_triton_matches_reference_cuda(dtype):
    logits = torch.randn(4096, device="cuda", dtype=dtype, requires_grad=True)
    a = torch.randn(4096, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(4096, device="cuda", dtype=dtype, requires_grad=True)
    residual = torch.randn(4096, device="cuda", dtype=dtype, requires_grad=True)
    out, report = fused_gated_residual(logits, a, b, residual, backend="triton", debug=True)
    ref = reference(logits, a, b, residual)
    assert out.is_cuda
    assert out.dtype == dtype
    assert torch.allclose(out, ref, atol=3e-2, rtol=3e-2)
    assert report.triton_used
    assert report.activation == "silu"
    assert report.gate == "sigmoid"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_forced_triton_backward_matches_torch_close_cuda():
    dtype = torch.float32
    tensors = [torch.randn(2048, device="cuda", dtype=dtype, requires_grad=True) for _ in range(4)]
    ref_tensors = [t.detach().clone().requires_grad_(True) for t in tensors]
    out = fused_gated_residual(*tensors, backend="triton")
    ref = reference(*ref_tensors)
    out.sum().backward()
    ref.sum().backward()
    for actual, expected in zip(tensors, ref_tensors):
        assert torch.allclose(actual.grad, expected.grad, atol=1e-4, rtol=1e-4)


def test_fused_falls_back_for_unsupported_activation_cpu():
    x = torch.randn(8)
    out, report = fused_gated_residual(x, x, x, x, activation="gelu", debug=True)
    assert report.backend == "torch_composition"
    assert not report.triton_used
    assert out.shape == x.shape


def test_cpu_auto_with_known_fast_pattern_selects_torch_composition():
    x = torch.randn(8)
    out, report = fused_gated_residual(x, x, x, x, activation="silu", gate="sigmoid", debug=True)
    assert report.backend == "torch_composition"
    assert not report.triton_used
    assert torch.allclose(out, reference(x, x, x, x))


def test_cpu_forced_triton_raises_clear_cuda_only_error():
    x = torch.randn(8)
    with pytest.raises(RuntimeError, match=r"CUDA-only.*backend='torch'.*backend='auto'"):
        fused_gated_residual(x, x, x, x, activation="silu", backend="triton")


def test_available_ops_include_aliases():
    activations = available_activations()
    gates = available_gates()
    assert "silu" in activations
    assert "swish" in activations
    assert "none" in activations
    assert "identity" in activations
    assert "clamp01" in gates
    assert "hard_sigmoid" in gates


@pytest.mark.parametrize(
    ("activation", "activation_fn", "kwargs"),
    [
        ("identity", lambda x: x, None),
        ("none", lambda x: x, None),
        ("relu", F.relu, None),
        ("gelu", F.gelu, None),
        ("silu", F.silu, None),
        ("swish", F.silu, None),
        ("mish", F.mish, None),
        ("elu", F.elu, None),
        ("selu", F.selu, None),
        ("leaky_relu", F.leaky_relu, {"negative_slope": 0.2}),
        ("hardtanh", F.hardtanh, {"min_val": -0.25, "max_val": 0.5}),
        ("hardswish", F.hardswish, None),
        ("hardsigmoid", F.hardsigmoid, None),
        ("softplus", F.softplus, None),
        ("sigmoid", torch.sigmoid, None),
        ("tanh", torch.tanh, None),
    ],
)
def test_builtin_activations_match_torch(activation, activation_fn, kwargs):
    logits = torch.randn(32)
    a = torch.randn(32)
    b = torch.randn(32)
    residual = torch.randn(32)
    out = fused_gated_residual(
        logits,
        a,
        b,
        residual,
        activation=activation,
        activation_kwargs=kwargs,
        backend="torch",
    )
    if kwargs is None:
        expected_activation = activation_fn
    else:
        expected_activation = lambda x: activation_fn(x, **kwargs)
    assert torch.allclose(out, reference(logits, a, b, residual, expected_activation))


@pytest.mark.parametrize(
    ("gate", "gate_fn"),
    [
        ("sigmoid", torch.sigmoid),
        ("identity", lambda x: x),
        ("none", lambda x: x),
        ("clamp01", lambda x: x.clamp(0, 1)),
        ("hard_sigmoid", F.hardsigmoid),
        ("hardsigmoid", F.hardsigmoid),
    ],
)
def test_builtin_gates_match_torch(gate, gate_fn):
    logits = torch.linspace(-2, 2, 32)
    a = torch.randn(32)
    b = torch.randn(32)
    residual = torch.randn(32)
    out = fused_gated_residual(logits, a, b, residual, activation="relu", gate=gate, backend="torch")
    assert torch.allclose(out, reference(logits, a, b, residual, F.relu, gate_fn))


def test_torch_composition_preserves_broadcasting():
    logits = torch.randn(2, 1)
    a = torch.randn(2, 3)
    b = torch.randn(1, 3)
    residual = torch.randn(2, 1)
    out, report = fused_gated_residual(logits, a, b, residual, activation="relu", debug=True)
    expected = reference(logits, a, b, residual, F.relu)
    assert out.shape == (2, 3)
    assert torch.allclose(out, expected)
    assert report.backend == "torch_composition"


def test_custom_activation_callable_and_composed_ops():
    logits = torch.randn(16)
    a = torch.randn(16)
    b = torch.randn(16)
    residual = torch.randn(16)
    custom = compose_ops("relu", torch.tanh)
    out, report = fused_gated_residual(logits, a, b, residual, activation=custom, backend="auto", debug=True)
    expected = reference(logits, a, b, residual, lambda x: torch.tanh(F.relu(x)))
    assert torch.allclose(out, expected)
    assert report.backend == "torch_composition"
    assert report.custom_activation


def test_compose_ops_can_use_gate_registry_ops_too():
    op = compose_ops("relu", "clamp01")
    x = torch.tensor([-1.0, 0.25, 2.0])
    assert torch.allclose(op(x), torch.tensor([0.0, 0.25, 1.0]))


def test_user_registered_activation():
    def square(x):
        return x * x

    register_activation("square_test", square, aliases=("sq_test",))
    logits = torch.randn(8)
    a = torch.randn(8)
    b = torch.randn(8)
    residual = torch.randn(8)
    out = fused_gated_residual(logits, a, b, residual, activation="sq_test")
    assert torch.allclose(out, reference(logits, a, b, residual, square))


def test_unknown_activation_error_lists_available_ops():
    x = torch.randn(4)
    with pytest.raises(ValueError, match="Unknown operation"):
        fused_gated_residual(x, x, x, x, activation="definitely_not_real")


def test_fused_debug_report_has_capabilities_and_reason_cpu():
    x = torch.randn(8)
    _, report = fused_gated_residual(x, x, x, x, activation="swish", debug=True)
    assert report.backend == "torch_composition"
    assert report.reason
    assert report.activation == "silu"
    assert report.gate == "sigmoid"
    assert not report.custom_activation
    assert report.capabilities.torch_available
    assert report.capabilities.device_type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compile_fullgraph_fused_torch_path_cuda():
    def fn(logits, a, b, residual):
        return fused_gated_residual(logits, a, b, residual, backend="torch")

    compiled = torch.compile(fn, fullgraph=True)
    tensors = [torch.randn(128, device="cuda") for _ in range(4)]
    assert torch.allclose(compiled(*tensors), reference(*tensors), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compile_fullgraph_fused_auto_path_cuda():
    def fn(logits, a, b, residual):
        return fused_gated_residual(logits, a, b, residual, backend="auto")

    compiled = torch.compile(fn, fullgraph=True)
    tensors = [torch.randn(128, device="cuda") for _ in range(4)]
    assert torch.allclose(compiled(*tensors), reference(*tensors), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compile_fullgraph_custom_callable_cuda():
    def custom(x):
        return F.relu(x) + 0.1 * x

    def fn(logits, a, b, residual):
        return fused_gated_residual(logits, a, b, residual, activation=custom, backend="torch")

    compiled = torch.compile(fn, fullgraph=True)
    tensors = [torch.randn(128, device="cuda") for _ in range(4)]
    expected = reference(*tensors, activation_fn=custom)
    assert torch.allclose(compiled(*tensors), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_auto_routes_small_cuda_tensors_to_torch_composition():
    tensors = [torch.randn(1024, device="cuda") for _ in range(4)]
    _, report = fused_gated_residual(*tensors, backend="auto", debug=True)
    assert report.backend == "torch_composition"
    assert not report.triton_used
    assert "small tensors" in report.reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_only_known_fast_pattern_can_force_triton_cuda():
    tensors = [torch.randn(4096, device="cuda") for _ in range(4)]
    with pytest.raises(RuntimeError, match="activation='silu'"):
        fused_gated_residual(*tensors, activation="gelu", backend="triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_swish_alias_can_force_known_fast_triton_cuda():
    tensors = [torch.randn(4096, device="cuda") for _ in range(4)]
    _, report = fused_gated_residual(*tensors, activation="swish", backend="triton", debug=True)
    assert report.triton_used
    assert report.activation == "silu"
