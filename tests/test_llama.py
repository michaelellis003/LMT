"""Unit tests for LLaMA model."""

import torch

from lmt.models.config import ModelConfig
from lmt.models.llama import LLaMA


class TestLLaMA:
    """Test suite for LLaMA model."""

    def _make_config(self, **kwargs):
        """Create a ModelConfig suitable for LLaMA tests."""
        defaults = dict(
            embed_dim=64,
            num_heads=8,
            num_kv_heads=2,
            context_length=32,
            vocab_size=256,
            num_layers=2,
            dropout=0.0,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)

    def test_initialization(self):
        """Test that LLaMA initializes with valid config."""
        config = self._make_config()
        model = LLaMA(config)
        assert hasattr(model, 'tok_embed')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'final_norm')
        assert hasattr(model, 'out_head')

    def test_no_positional_embedding(self):
        """LLaMA uses RoPE, so no learnable positional embedding."""
        config = self._make_config()
        model = LLaMA(config)
        assert not hasattr(model, 'pos_embed')

    def test_uses_rmsnorm(self):
        """Verify LLaMA uses RMSNorm, not LayerNorm."""
        from lmt.layers.normalization import RMSNorm

        config = self._make_config()
        model = LLaMA(config)
        assert isinstance(model.final_norm, RMSNorm)

    def test_forward_shape(self):
        """Test output shape: [batch, seq_len, vocab_size]."""
        config = self._make_config()
        model = LLaMA(config)
        x = torch.randint(0, 256, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 256)

    def test_gradient_flow(self):
        """Test gradients propagate through the full model."""
        config = self._make_config()
        model = LLaMA(config)
        x = torch.randint(0, 256, (2, 8))
        out = model(x)
        out.sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'No gradient for {name}'
                assert not torch.isnan(p.grad).any(), (
                    f'NaN gradient for {name}'
                )

    def test_different_configs(self):
        """Test various valid configurations."""
        configs = [
            self._make_config(num_layers=1, num_kv_heads=1),
            self._make_config(num_layers=4, num_kv_heads=8),
            self._make_config(embed_dim=128, num_heads=4, num_kv_heads=2),
        ]
        for config in configs:
            model = LLaMA(config)
            x = torch.randint(0, config.vocab_size, (1, 8))
            out = model(x)
            assert out.shape == (1, 8, config.vocab_size)

    def test_deterministic_eval(self):
        """Test eval mode produces identical outputs."""
        config = self._make_config()
        model = LLaMA(config)
        model.eval()
        x = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_causal(self):
        """Test that the model is causal (autoregressive)."""
        config = self._make_config()
        model = LLaMA(config)
        model.eval()

        x = torch.randint(0, 256, (1, 8))
        x_extended = torch.cat([x, torch.randint(0, 256, (1, 1))], dim=1)

        with torch.no_grad():
            out_short = model(x)
            out_long = model(x_extended)

        # First 8 positions should match
        torch.testing.assert_close(
            out_short,
            out_long[:, :8, :],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_single_token(self):
        """Test with seq_len=1."""
        config = self._make_config()
        model = LLaMA(config)
        x = torch.randint(0, 256, (1, 1))
        out = model(x)
        assert out.shape == (1, 1, 256)

    def test_parameter_count_reasonable(self):
        """Sanity check: model has parameters."""
        config = self._make_config()
        model = LLaMA(config)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0
        # With small config, should be < 1M params
        assert total < 1_000_000
