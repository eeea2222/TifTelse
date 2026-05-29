import pytest
import torch
import torch.nn.functional as F

from tif_telse import (
    ResolvedOp,
    apply_op,
    compose_ops,
    register_activation,
    resolve_activation,
    resolve_gate,
    resolve_op,
)


def test_resolve_op_returns_resolvedop_for_builtin_activation():
    resolved = resolve_op("relu")
    assert isinstance(resolved, ResolvedOp)
    assert resolved.name == "relu"
    assert resolved.builtin
    assert not resolved.custom
    x = torch.tensor([-1.0, 2.0])
    assert torch.equal(apply_op(resolved, x), F.relu(x))


def test_resolve_op_falls_back_from_activation_to_gate_registry():
    # clamp01 is only registered as a gate, not an activation.
    resolved = resolve_op("clamp01")
    assert resolved.name == "clamp01"
    x = torch.tensor([-1.0, 0.5, 2.0])
    assert torch.equal(apply_op(resolved, x), x.clamp(0, 1))


def test_resolve_op_with_callable_is_custom():
    def square(x):
        return x * x

    resolved = resolve_op(square)
    assert resolved.custom
    assert not resolved.builtin
    assert resolved.name == "square"
    x = torch.tensor([2.0, 3.0])
    assert torch.equal(apply_op(resolved, x), x * x)


def test_resolve_op_rejects_non_str_non_callable():
    with pytest.raises(TypeError, match="registered name or callable"):
        resolve_op(123)  # type: ignore[arg-type]


def test_resolve_with_kwargs_are_applied():
    resolved = resolve_activation("leaky_relu", {"negative_slope": 0.5})
    x = torch.tensor([-2.0, 2.0])
    assert torch.allclose(apply_op(resolved, x), F.leaky_relu(x, negative_slope=0.5))


def test_unknown_op_error_lists_available_ops():
    with pytest.raises(ValueError, match="Unknown operation 'nope'"):
        resolve_activation("nope")
    with pytest.raises(ValueError, match="Unknown operation 'nope'"):
        resolve_gate("nope")


def test_name_normalization_is_case_and_dash_insensitive():
    assert resolve_activation("ReLU").name == "relu"
    assert resolve_gate("HARD-SIGMOID").name == "hard_sigmoid"


def test_register_activation_rejects_empty_name():
    with pytest.raises(ValueError, match="cannot be empty"):
        register_activation("   ", lambda x: x)


def test_register_activation_rejects_non_callable():
    with pytest.raises(TypeError, match="must be callable"):
        register_activation("not_callable_test", 5)  # type: ignore[arg-type]


def test_compose_ops_empty_is_identity():
    op = compose_ops()
    assert op.__name__ == "compose_identity"
    x = torch.randn(4)
    assert torch.equal(op(x), x)


def test_compose_ops_names_pipeline():
    op = compose_ops("relu", "clamp01")
    assert op.__name__ == "compose_relu_then_clamp01"
