"""
Pytest configuration and fixtures for the Cham-GPT project.
"""
import sys
import importlib.util
from pathlib import Path

import pytest

# Project root directory
PROJECT_ROOT = Path(__file__).parent


def load_module_from_path(module_name: str, file_path: Path):
    """
    Helper to load a module from a file path.
    Useful for importing from directories with hyphens (like GPT-2).
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gpt2_layers_path():
    """Returns the path to GPT-2/layers directory."""
    return PROJECT_ROOT / "GPT-2" / "layers"


@pytest.fixture
def linear_module(gpt2_layers_path):
    """Loads and returns the linear module from GPT-2/layers."""
    return load_module_from_path("linear", gpt2_layers_path / "linear.py")


@pytest.fixture
def Linear(linear_module):
    """Returns the Linear class from GPT-2/layers/linear.py."""
    return linear_module.Linear
