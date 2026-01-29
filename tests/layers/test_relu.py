"""
Tests for GPT-2/layers/relu.py ReLU layer.
"""
import pytest
import torch as t
import torch.nn as nn
import importlib.util
from pathlib import Path


# Load ReLU class from hyphenated directory
_relu_path = Path(__file__).parent.parent.parent / "GPT-2" / "layers" / "relu.py"
_spec = importlib.util.spec_from_file_location("relu", _relu_path)
_relu_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_relu_module)
ReLU = _relu_module.ReLU


def test_relu_forward_positive():
    """ReLU should pass positive values unchanged."""
    layer = ReLU()
    x = t.tensor([1.0, 2.0, 3.0])
    out = layer(x)
    t.testing.assert_close(out, x)


def test_relu_forward_negative():
    """ReLU should zero out negative values."""
    layer = ReLU()
    x = t.tensor([-1.0, -2.0, -3.0])
    out = layer(x)
    expected = t.zeros(3)
    t.testing.assert_close(out, expected)


def test_relu_forward_mixed():
    """ReLU should handle mixed positive and negative values."""
    layer = ReLU()
    x = t.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = layer(x)
    expected = t.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    t.testing.assert_close(out, expected)


def test_relu_forward_zero():
    """ReLU should pass zero as zero."""
    layer = ReLU()
    x = t.tensor([0.0])
    out = layer(x)
    t.testing.assert_close(out, t.tensor([0.0]))


def test_relu_matches_pytorch():
    """Custom ReLU should produce identical results to torch.nn.ReLU."""
    layer = ReLU()
    official = nn.ReLU()
    
    x = t.randn(32, 64)
    t.testing.assert_close(layer(x), official(x))


def test_relu_output_shape_2d():
    """ReLU should preserve shape for 2D tensors."""
    layer = ReLU()
    x = t.randn(16, 32)
    assert layer(x).shape == (16, 32)


def test_relu_output_shape_3d():
    """ReLU should preserve shape for 3D tensors."""
    layer = ReLU()
    x = t.randn(8, 16, 32)
    assert layer(x).shape == (8, 16, 32)


def test_relu_gradients():
    """ReLU should properly propagate gradients."""
    layer = ReLU()
    x = t.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    # Gradient is 1 for positive values, 0 for negative
    expected_grad = t.tensor([[0.0, 1.0], [1.0, 0.0]])
    t.testing.assert_close(x.grad, expected_grad)


def test_relu_no_parameters():
    """ReLU should have no learnable parameters."""
    layer = ReLU()
    params = list(layer.parameters())
    assert len(params) == 0, f"Expected 0 parameters, got {len(params)}"
