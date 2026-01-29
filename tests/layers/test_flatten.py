"""
Tests for GPT-2/layers/flatten.py Flatten layer.
"""
import pytest
import torch as t
import torch.nn as nn
import importlib.util
from pathlib import Path


# Load Flatten class from hyphenated directory
_path = Path(__file__).parent.parent.parent / "GPT-2" / "layers" / "flatten.py"
_spec = importlib.util.spec_from_file_location("flatten", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
Flatten = _module.Flatten


def test_flatten_default():
    """Default flatten should flatten from dim 1 to end."""
    layer = Flatten()
    x = t.randn(2, 3, 4, 5)
    out = layer(x)
    assert out.shape == (2, 60)  # 3*4*5 = 60


def test_flatten_matches_pytorch():
    """Custom Flatten should match nn.Flatten."""
    layer = Flatten()
    official = nn.Flatten()
    x = t.randn(4, 3, 8, 8)
    t.testing.assert_close(layer(x), official(x))


def test_flatten_custom_dims():
    """Flatten with custom start/end dims."""
    layer = Flatten(start_dim=1, end_dim=2)
    x = t.randn(2, 3, 4, 5)
    out = layer(x)
    assert out.shape == (2, 12, 5)  # 3*4 = 12


def test_flatten_negative_end_dim():
    """Flatten with negative end_dim."""
    layer = Flatten(start_dim=1, end_dim=-1)
    x = t.randn(2, 3, 4)
    out = layer(x)
    assert out.shape == (2, 12)  # 3*4 = 12


def test_flatten_no_parameters():
    """Flatten should have no learnable parameters."""
    layer = Flatten()
    assert len(list(layer.parameters())) == 0
