"""Tests for model factory utility."""

import pytest
import torch.nn as nn

from lmt.models.config import ModelConfig
from lmt.models.factory import create_model, list_architectures


def _tiny_config(**overrides) -> ModelConfig:
    """Create a minimal ModelConfig for testing."""
    defaults = dict(
        embed_dim=32,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        ffn_hidden_dim=64,
        vocab_size=100,
        context_length=16,
        tie_weights=True,
        dropout=0.0,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestCreateModel:
    """Test create_model factory function."""

    def test_create_gpt(self):
        """Should create a GPT model."""
        model = create_model('gpt', _tiny_config())
        assert isinstance(model, nn.Module)

    def test_create_llama(self):
        """Should create a LLaMA model."""
        model = create_model('llama', _tiny_config())
        assert isinstance(model, nn.Module)

    def test_create_qwen3(self):
        """Should create a Qwen3 model."""
        model = create_model('qwen3', _tiny_config())
        assert isinstance(model, nn.Module)

    def test_create_gemma(self):
        """Should create a Gemma model."""
        model = create_model('gemma', _tiny_config())
        assert isinstance(model, nn.Module)

    def test_create_mixtral(self):
        """Should create a Mixtral model."""
        config = _tiny_config()
        model = create_model('mixtral', config)
        assert isinstance(model, nn.Module)

    def test_case_insensitive(self):
        """Should handle different cases."""
        model1 = create_model('GPT', _tiny_config())
        model2 = create_model('gpt', _tiny_config())
        assert type(model1) is type(model2)

    def test_unknown_architecture(self):
        """Should raise ValueError for unknown architecture."""
        with pytest.raises(ValueError, match='Unknown architecture'):
            create_model('transformer_xl', _tiny_config())

    def test_returns_nn_module(self):
        """Should always return an nn.Module."""
        for arch in ['gpt', 'llama', 'qwen3', 'gemma']:
            model = create_model(arch, _tiny_config())
            assert isinstance(model, nn.Module)


class TestListArchitectures:
    """Test list_architectures function."""

    def test_returns_list(self):
        """Should return a list of strings."""
        archs = list_architectures()
        assert isinstance(archs, list)
        assert all(isinstance(a, str) for a in archs)

    def test_includes_main_architectures(self):
        """Should include the main architectures."""
        archs = list_architectures()
        for expected in ['gpt', 'llama', 'qwen3', 'gemma', 'mixtral']:
            assert expected in archs

    def test_sorted(self):
        """Should return sorted list."""
        archs = list_architectures()
        assert archs == sorted(archs)
