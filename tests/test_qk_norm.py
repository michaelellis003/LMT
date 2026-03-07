"""Tests for QK-Norm (Query-Key Normalization)."""

import torch

from lmt.models.config import ModelConfig


class TestQKNorm:
    """Test QK normalization in GQA attention."""

    def _make_config(self, qk_norm: bool = True) -> ModelConfig:
        """Create a small config."""
        return ModelConfig(
            context_length=16,
            vocab_size=64,
            num_layers=1,
            num_heads=4,
            embed_dim=32,
            dropout=0.0,
            qk_norm=qk_norm,
        )

    def test_config_has_qk_norm_field(self):
        """ModelConfig should have qk_norm parameter."""
        config = self._make_config(qk_norm=True)
        assert config.qk_norm is True

    def test_default_is_false(self):
        """Default should be qk_norm=False."""
        config = ModelConfig()
        assert config.qk_norm is False

    def test_gqa_with_qk_norm(self):
        """GQA should work with qk_norm enabled."""
        from lmt.layers.attention.grouped_query_attention import (
            GroupedQueryAttention,
        )

        config = self._make_config(qk_norm=True)
        attn = GroupedQueryAttention(config)
        x = torch.randn(2, 8, config.embed_dim)
        out = attn(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_gqa_without_qk_norm(self):
        """GQA should work without qk_norm (backward compat)."""
        from lmt.layers.attention.grouped_query_attention import (
            GroupedQueryAttention,
        )

        config = self._make_config(qk_norm=False)
        attn = GroupedQueryAttention(config)
        x = torch.randn(2, 8, config.embed_dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_qk_norm_has_norm_layers(self):
        """When qk_norm=True, GQA should have q_norm and k_norm."""
        from lmt.layers.attention.grouped_query_attention import (
            GroupedQueryAttention,
        )

        config = self._make_config(qk_norm=True)
        attn = GroupedQueryAttention(config)
        assert hasattr(attn, 'q_norm')
        assert hasattr(attn, 'k_norm')

    def test_qk_norm_prevents_logit_explosion(self):
        """QK-Norm should keep attention logits bounded."""
        from lmt.layers.attention.grouped_query_attention import (
            GroupedQueryAttention,
        )

        config = self._make_config(qk_norm=True)
        attn = GroupedQueryAttention(config)
        # Large input to provoke potential explosion
        x = torch.randn(1, 8, config.embed_dim) * 100
        out = attn(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
