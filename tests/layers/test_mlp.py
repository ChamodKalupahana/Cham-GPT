"""
Tests for GPT-2/layers/mlp.py MLP layer.
"""
import pytest
import torch as t
import torch.nn as nn
import importlib.util
from pathlib import Path


# Load MLP class from hyphenated directory
_path = Path(__file__).parent.parent.parent / "GPT-2" / "layers" / "mlp.py"
_spec = importlib.util.spec_from_file_location("mlp", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
MLP = _module.MLP


def test_mlp_output_shape():
    """MLP should produce correct output shape."""
    mlp = MLP(in_feats=64, num_hidden_layers=128, out_feats=32)
    x = t.randn(8, 64)
    out = mlp(x)
    assert out.shape == (8, 32)


def test_mlp_output_non_negative():
    """MLP output should be non-negative (ReLU applied)."""
    mlp = MLP(in_feats=64, num_hidden_layers=128, out_feats=32)
    x = t.randn(8, 64)
    out = mlp(x)
    assert (out >= 0).all(), "Output should be non-negative after ReLU"


def test_mlp_has_parameters():
    """MLP should have learnable parameters from Linear layers."""
    mlp = MLP(in_feats=64, num_hidden_layers=128, out_feats=32)
    params = list(mlp.parameters())
    assert len(params) > 0, "MLP should have parameters"


def test_mlp_gradients():
    """MLP should properly propagate gradients."""
    mlp = MLP(in_feats=16, num_hidden_layers=32, out_feats=8)
    x = t.randn(4, 16, requires_grad=True)
    
    out = mlp(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Input should have gradients"


def test_mlp_3d_input():
    """MLP should handle 3D input (batch, seq, features)."""
    mlp = MLP(in_feats=64, num_hidden_layers=128, out_feats=32)
    x = t.randn(4, 10, 64)
    out = mlp(x)
    assert out.shape == (4, 10, 32)
