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


class TestLinearInit:
    """Tests for Linear layer initialization."""
    
    def test_creates_weight_parameter(self):
        """Weight matrix should be created with correct shape."""
        layer = Linear(in_feats=10, out_feats=5)
        assert hasattr(layer, 'W')
        assert layer.W.shape == (10, 5)
    
    def test_creates_bias_parameter(self):
        """Bias vector should be created with correct shape."""
        layer = Linear(in_feats=10, out_feats=5)
        assert hasattr(layer, 'bias')
        assert layer.bias.shape == (5,)
    
    def test_weight_requires_grad(self):
        """Weight should require gradients."""
        layer = Linear(in_feats=10, out_feats=5)
        assert layer.W.requires_grad is True
    
    def test_bias_requires_grad_when_enabled(self):
        """Bias should require gradients when bias=True."""
        layer = Linear(in_feats=10, out_feats=5, bias=True)
        assert layer.bias.requires_grad is True
    
    def test_bias_no_grad_when_disabled(self):
        """Bias should not require gradients when bias=False."""
        layer = Linear(in_feats=10, out_feats=5, bias=False)
        assert layer.bias.requires_grad is False
    
    def test_weight_is_nn_parameter(self):
        """Weight should be an nn.Parameter."""
        layer = Linear(in_feats=10, out_feats=5)
        assert isinstance(layer.W, nn.Parameter)
    
    def test_bias_is_nn_parameter(self):
        """Bias should be an nn.Parameter."""
        layer = Linear(in_feats=10, out_feats=5)
        assert isinstance(layer.bias, nn.Parameter)


class TestLinearForward:
    """Tests for Linear layer forward pass."""
    
    def test_output_shape_2d_input(self):
        """Output shape should be correct for 2D input (batch, features)."""
        layer = Linear(in_feats=10, out_feats=5)
        x = t.randn(32, 10)
        out = layer(x)
        assert out.shape == (32, 5)
    
    def test_output_shape_3d_input(self):
        """Output shape should be correct for 3D input (batch, seq, features)."""
        layer = Linear(in_feats=10, out_feats=5)
        x = t.randn(32, 20, 10)
        out = layer(x)
        assert out.shape == (32, 20, 5)
    
    def test_output_shape_single_sample(self):
        """Output shape should be correct for single sample."""
        layer = Linear(in_feats=10, out_feats=5)
        x = t.randn(1, 10)
        out = layer(x)
        assert out.shape == (1, 5)
    
    def test_forward_is_differentiable(self):
        """Forward pass should allow gradient computation."""
        layer = Linear(in_feats=10, out_feats=5)
        x = t.randn(32, 10, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.W.grad is not None
    
    def test_forward_applies_linear_transformation(self):
        """Forward should compute y = xW + b."""
        layer = Linear(in_feats=3, out_feats=2)
        
        # Set known weights and bias for verification
        with t.no_grad():
            layer.W.copy_(t.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
            layer.bias.copy_(t.tensor([0.1, 0.2]))
        
        x = t.tensor([[1.0, 1.0, 1.0]])
        out = layer(x)
        
        # Expected: [1, 1, 1] @ [[1, 2], [3, 4], [5, 6]] + [0.1, 0.2]
        # = [1+3+5, 2+4+6] + [0.1, 0.2] = [9.1, 12.2]
        expected = t.tensor([[9.1, 12.2]])
        assert t.allclose(out, expected, atol=1e-5)


class TestLinearComparison:
    """Tests comparing custom Linear to PyTorch's nn.Linear."""
    
    def test_same_output_as_pytorch_linear(self):
        """Custom Linear should produce same output as nn.Linear with same weights."""
        in_feats, out_feats = 10, 5
        
        custom_layer = Linear(in_feats, out_feats)
        pytorch_layer = nn.Linear(in_feats, out_feats)
        
        # Copy weights from custom to PyTorch (note: nn.Linear uses transposed weights)
        with t.no_grad():
            pytorch_layer.weight.copy_(custom_layer.W.T)
            pytorch_layer.bias.copy_(custom_layer.bias)
        
        x = t.randn(32, in_feats)
        custom_out = custom_layer(x)
        pytorch_out = pytorch_layer(x)
        
        assert t.allclose(custom_out, pytorch_out, atol=1e-5)


class TestLinearEdgeCases:
    """Edge case tests for Linear layer."""
    
    def test_single_input_feature(self):
        """Should work with single input feature."""
        layer = Linear(in_feats=1, out_feats=5)
        x = t.randn(10, 1)
        out = layer(x)
        assert out.shape == (10, 5)
    
    def test_single_output_feature(self):
        """Should work with single output feature."""
        layer = Linear(in_feats=10, out_feats=1)
        x = t.randn(10, 10)
        out = layer(x)
        assert out.shape == (10, 1)
    
    def test_large_dimensions(self):
        """Should work with larger dimensions."""
        layer = Linear(in_feats=512, out_feats=2048)
        x = t.randn(4, 512)
        out = layer(x)
        assert out.shape == (4, 2048)
    
    def test_zero_input(self):
        """Output with zero input should equal bias."""
        layer = Linear(in_feats=5, out_feats=3)
        x = t.zeros(1, 5)
        out = layer(x)
        assert t.allclose(out, layer.bias.unsqueeze(0), atol=1e-5)
