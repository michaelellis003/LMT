"""Tests for value-residual connections (Gemma 2 style).

Value-residual blends the first layer's V projection into subsequent
layers' V: v = (1 - lambda) * v_current + lambda * v_first.
This improves gradient flow to early layers.
"""

import torch

from lmt.models.config import ModelConfig


def _make_config(**overrides: object) -> ModelConfig:
    """Create a tiny model config for testing."""
    defaults = {
        'embed_dim': 32,
        'num_heads': 4,
        'num_kv_heads': 2,
        'num_layers': 4,
        'vocab_size': 64,
        'context_length': 16,
        'dropout': 0.0,
        'value_residual_mix': 0.0,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


class TestValueResidualConfig:
    """Test value_residual_mix config parameter."""

    def test_default_is_zero(self):
        """Value residual should be disabled by default."""
        config = ModelConfig()
        assert config.value_residual_mix == 0.0

    def test_can_set_mix(self):
        """Should be able to set a mix value."""
        config = _make_config(value_residual_mix=0.5)
        assert config.value_residual_mix == 0.5


class TestGQAValueResidual:
    """Test that GQA stores and blends V correctly."""

    def test_gqa_stores_raw_v(self):
        """After forward, GQA should store _raw_v attribute."""
        from lmt.layers.attention.grouped_query_attention import (
            GroupedQueryAttention,
        )

        config = _make_config()
        gqa = GroupedQueryAttention(config)
        x = torch.randn(1, 8, 32)
        gqa(x)

        assert hasattr(gqa, '_raw_v')
        assert gqa._raw_v is not None
        # V shape: [batch, kv_heads, seq_len, head_dim]
        assert gqa._raw_v.shape == (1, 2, 8, 8)

    def test_gqa_blends_first_layer_v(self):
        """When _first_layer_v is set, V should be blended."""
        from lmt.layers.attention.grouped_query_attention import (
            GroupedQueryAttention,
        )

        config = _make_config(value_residual_mix=0.5)
        gqa = GroupedQueryAttention(config)

        x = torch.randn(1, 8, 32)

        # Set a first-layer V (all ones for easy checking)
        fake_v1 = torch.ones(1, 2, 8, 8)
        gqa._first_layer_v = fake_v1
        gqa._value_residual_mix = 0.5

        out = gqa(x)
        assert out.shape == (1, 8, 32)

    def test_no_blend_when_mix_is_zero(self):
        """With mix=0.0, first_layer_v should have no effect."""
        from lmt.layers.attention.grouped_query_attention import (
            GroupedQueryAttention,
        )

        config = _make_config(value_residual_mix=0.0)
        gqa = GroupedQueryAttention(config)
        x = torch.randn(1, 8, 32)

        # Run without first_layer_v
        torch.manual_seed(42)
        out_no_blend = gqa(x.clone())

        # Run with first_layer_v but mix=0
        gqa._first_layer_v = torch.randn(1, 2, 8, 8)
        gqa._value_residual_mix = 0.0
        torch.manual_seed(42)
        out_with_v1 = gqa(x.clone())

        assert torch.allclose(out_no_blend, out_with_v1, atol=1e-5)


class TestBaseModelValueResidual:
    """Test model-level value-residual wiring."""

    def test_model_propagates_first_layer_v(self):
        """BaseModel should propagate first layer's V to later layers."""
        from lmt.models.base import BaseModel

        config = _make_config(value_residual_mix=0.5)
        model = BaseModel(config)
        x = torch.randint(0, 64, (1, 8))

        # Forward should work without errors
        logits = model(x)
        assert logits.shape == (1, 8, 64)

    def test_model_works_without_value_residual(self):
        """Model should work normally with mix=0.0."""
        from lmt.models.base import BaseModel

        config = _make_config(value_residual_mix=0.0)
        model = BaseModel(config)
        x = torch.randint(0, 64, (1, 8))

        logits = model(x)
        assert logits.shape == (1, 8, 64)

    def test_gradient_flows_with_value_residual(self):
        """Gradients should flow through value-residual connections."""
        from lmt.models.base import BaseModel

        config = _make_config(value_residual_mix=0.5)
        model = BaseModel(config)
        x = torch.randint(0, 64, (2, 8))

        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f'No gradient for {name}'

    def test_value_residual_changes_output(self):
        """Enabling value-residual should change model output."""
        from lmt.models.base import BaseModel

        config_off = _make_config(value_residual_mix=0.0)
        config_on = _make_config(value_residual_mix=0.5)

        model_off = BaseModel(config_off)
        model_on = BaseModel(config_on)

        # Copy weights so only the value-residual differs
        model_on.load_state_dict(model_off.state_dict())

        x = torch.randint(0, 64, (1, 8))
        out_off = model_off(x)
        out_on = model_on(x)

        # Outputs should differ because V blending changes attention
        assert not torch.allclose(out_off, out_on, atol=1e-6)
