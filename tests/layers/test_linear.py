"""
Tests for GPT-2/layers/linear.py Linear layer.
"""
import pytest
import torch as t
import torch.nn as nn
import importlib.util
from pathlib import Path


# Load Linear class from hyphenated directory
_linear_path = Path(__file__).parent.parent.parent / "GPT-2" / "layers" / "linear.py"
_spec = importlib.util.spec_from_file_location("linear", _linear_path)
_linear_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_linear_module)
Linear = _linear_module.Linear


def test_linear_forward_with_bias():
    """Linear should produce identical results to torch.nn.Linear with bias enabled."""
    x = t.rand((10, 512))
    yours = Linear(512, 64, bias=True)
    official = nn.Linear(512, 64, bias=True)
    
    # Copy weights from official to yours (note: nn.Linear uses transposed weights)
    with t.no_grad():
        yours.W.copy_(official.weight.T)
        yours.bias.copy_(official.bias)
    
    actual = yours(x)
    expected = official(x)
    t.testing.assert_close(actual, expected)


def test_linear_forward_no_bias():
    """Linear should produce identical results to torch.nn.Linear without bias."""
    x = t.rand((10, 512))
    yours = Linear(512, 64, bias=False)
    official = nn.Linear(512, 64, bias=False)
    
    # Copy weights from official to yours
    with t.no_grad():
        yours.W.copy_(official.weight.T)
    
    actual = yours(x)
    expected = official(x)
    t.testing.assert_close(actual, expected)


def test_linear_parameters_with_bias():
    """Linear with bias should have correct parameter structure."""
    layer = Linear(2, 3, bias=True)
    
    # Should have weight and bias as parameters
    params = list(layer.parameters())
    assert len(params) == 2, f"Expected 2 parameters (W and bias), got {len(params)}"
    
    # Check shapes
    assert layer.W.shape == (2, 3), f"Expected weight shape (2, 3), got {layer.W.shape}"
    assert layer.bias.shape == (3,), f"Expected bias shape (3,), got {layer.bias.shape}"
    
    # Check stored attributes
    assert layer.in_feats == 2
    assert layer.out_feats == 3


def test_linear_parameters_no_bias():
    """Linear without bias should have only weight parameter."""
    layer = Linear(2, 3, bias=False)
    
    # Should have only weight as parameter
    params = list(layer.parameters())
    assert len(params) == 1, f"Expected 1 parameter (W only), got {len(params)}"
    
    # Bias should be None
    assert layer.bias is None, "Bias should be None when not enabled."
    
    # Check weight shape
    assert layer.W.shape == (2, 3), f"Expected weight shape (2, 3), got {layer.W.shape}"


def test_linear_no_bias_forward():
    """Linear without bias should correctly compute forward pass."""
    x = t.rand((10, 512))
    layer = Linear(512, 64, bias=False)
    
    assert layer.bias is None, "Bias should be None when not enabled."
    assert len(list(layer.parameters())) == 1, "Should have exactly 1 parameter (weight only)"
    
    # Manual computation check
    expected = x @ layer.W
    actual = layer(x)
    t.testing.assert_close(actual, expected)


def test_linear_output_shapes():
    """Linear should produce correct output shapes for various inputs."""
    layer = Linear(64, 32, bias=True)
    
    # 2D input (batch, features)
    x_2d = t.rand((16, 64))
    assert layer(x_2d).shape == (16, 32)
    
    # 3D input (batch, seq, features)
    x_3d = t.rand((16, 10, 64))
    assert layer(x_3d).shape == (16, 10, 32)


def test_linear_gradients():
    """Linear should properly propagate gradients."""
    layer = Linear(10, 5, bias=True)
    x = t.rand((4, 10), requires_grad=True)
    
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Input should have gradients"
    assert layer.W.grad is not None, "Weight should have gradients"
    assert layer.bias.grad is not None, "Bias should have gradients"


def test_linear_zero_input():
    """With zero input, output should equal bias."""
    layer = Linear(5, 3, bias=True)
    x = t.zeros(1, 5)
    out = layer(x)
    t.testing.assert_close(out, layer.bias.unsqueeze(0))