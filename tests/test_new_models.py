"""Tests for new model architectures (Qwen3, Gemma)."""

import torch

from lmt.models.config import ModelConfig


def _small_config(**kwargs):
    """Create a small config for testing."""
    defaults = dict(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=4,
        context_length=32,
        vocab_size=256,
        num_layers=2,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


class TestQwen3:
    """Test Qwen3 model architecture."""

    def test_output_shape(self):
        """Qwen3 produces correct output shape."""
        from lmt.models.qwen3 import Qwen3

        config = _small_config(qk_norm=True, tie_weights=True)
        model = Qwen3(config)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_has_qk_norm(self):
        """Qwen3 should use QK-Norm."""
        from lmt.models.qwen3 import Qwen3

        config = _small_config()
        model = Qwen3(config)
        # QK-Norm is forced on regardless of config
        block = model.blocks[0]
        assert hasattr(block.attn, 'qk_norm')
        assert block.attn.qk_norm is True

    def test_has_weight_tying(self):
        """Qwen3 should tie input/output embeddings."""
        from lmt.models.qwen3 import Qwen3

        config = _small_config()
        model = Qwen3(config)
        assert model.tok_embed.weight is model.out_head.weight

    def test_gradient_flow(self):
        """Gradients flow through Qwen3."""
        from lmt.models.qwen3 import Qwen3

        config = _small_config()
        model = Qwen3(config)
        x = torch.randint(0, 256, (1, 8))
        logits = model(x)
        logits.sum().backward()
        assert model.tok_embed.weight.grad is not None

    def test_is_base_model_subclass(self):
        """Qwen3 is a BaseModel subclass."""
        from lmt.models.base import BaseModel
        from lmt.models.qwen3 import Qwen3

        assert issubclass(Qwen3, BaseModel)


class TestGemma:
    """Test Gemma model architecture."""

    def test_output_shape(self):
        """Gemma produces correct output shape."""
        from lmt.models.gemma import Gemma

        config = _small_config()
        model = Gemma(config)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_has_qk_norm(self):
        """Gemma should use QK-Norm."""
        from lmt.models.gemma import Gemma

        config = _small_config()
        model = Gemma(config)
        block = model.blocks[0]
        assert hasattr(block.attn, 'qk_norm')
        assert block.attn.qk_norm is True

    def test_has_weight_tying(self):
        """Gemma should tie input/output embeddings."""
        from lmt.models.gemma import Gemma

        config = _small_config()
        model = Gemma(config)
        assert model.tok_embed.weight is model.out_head.weight

    def test_gradient_flow(self):
        """Gradients flow through Gemma."""
        from lmt.models.gemma import Gemma

        config = _small_config()
        model = Gemma(config)
        x = torch.randint(0, 256, (1, 8))
        logits = model(x)
        logits.sum().backward()
        assert model.tok_embed.weight.grad is not None

    def test_is_base_model_subclass(self):
        """Gemma is a BaseModel subclass."""
        from lmt.models.base import BaseModel
        from lmt.models.gemma import Gemma

        assert issubclass(Gemma, BaseModel)


class TestModelImports:
    """Test that new models are importable from top-level."""

    def test_qwen3_importable(self):
        """Qwen3 is importable from lmt."""
        from lmt import Qwen3  # noqa: F401

    def test_gemma_importable(self):
        """Gemma is importable from lmt."""
        from lmt import Gemma  # noqa: F401
