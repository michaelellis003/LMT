"""Unit tests for Mixtral model."""

import torch

from lmt.models.config import ModelConfig
from lmt.models.mixtral import Mixtral


class TestMixtral:
    """Test suite for Mixtral model."""

    def _make_config(self, **kwargs):
        """Create a ModelConfig for Mixtral tests."""
        defaults = dict(
            embed_dim=64,
            num_heads=4,
            num_kv_heads=2,
            context_length=32,
            vocab_size=256,
            num_layers=2,
            dropout=0.0,
            window_size=16,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)

    def test_initialization(self):
        """Test Mixtral initializes correctly."""
        config = self._make_config()
        model = Mixtral(config)
        assert hasattr(model, 'tok_embed')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'final_norm')
        assert hasattr(model, 'out_head')

    def test_uses_moe(self):
        """Verify blocks use MoE FFN."""
        from lmt.layers.ffn.moe import MoEFeedForward

        config = self._make_config()
        model = Mixtral(config)
        for block in model.blocks:
            assert isinstance(block.ffn, MoEFeedForward)

    def test_forward_shape(self):
        """Test output: [batch, seq_len, vocab_size]."""
        config = self._make_config()
        model = Mixtral(config)
        x = torch.randint(0, 256, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 256)

    def test_aux_loss_collected(self):
        """Test total aux_loss is accessible after forward."""
        config = self._make_config()
        model = Mixtral(config)
        x = torch.randint(0, 256, (1, 8))
        model(x)
        assert hasattr(model, 'aux_loss')
        assert model.aux_loss > 0

    def test_gradient_flow(self):
        """Test gradients propagate through the model.

        Not all experts get gradients in every batch (sparse
        activation by design), but non-expert params and the
        router must always get gradients.
        """
        config = self._make_config()
        model = Mixtral(config)
        x = torch.randint(0, 256, (1, 8))
        out = model(x)
        (out.sum() + model.aux_loss).backward()
        for name, p in model.named_parameters():
            if p.requires_grad and 'experts.' not in name:
                assert p.grad is not None, f'No gradient for {name}'

    def test_deterministic_eval(self):
        """Test eval mode produces identical outputs."""
        config = self._make_config()
        model = Mixtral(config)
        model.eval()
        x = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_causal(self):
        """Test that the model is autoregressive."""
        config = self._make_config()
        model = Mixtral(config)
        model.eval()

        x = torch.randint(0, 256, (1, 8))
        x_ext = torch.cat(
            [x, torch.randint(0, 256, (1, 1))], dim=1
        )

        with torch.no_grad():
            out_short = model(x)
            out_long = model(x_ext)

        torch.testing.assert_close(
            out_short, out_long[:, :8, :],
            atol=1e-5, rtol=1e-5,
        )

    def test_single_token(self):
        """Test with seq_len=1."""
        config = self._make_config()
        model = Mixtral(config)
        x = torch.randint(0, 256, (1, 1))
        out = model(x)
        assert out.shape == (1, 1, 256)

    def test_different_expert_configs(self):
        """Test with various num_experts / top_k combos."""
        config = self._make_config()
        for ne, tk in [(4, 1), (8, 2), (16, 4)]:
            model = Mixtral(config, num_experts=ne, top_k=tk)
            x = torch.randint(0, 256, (1, 4))
            out = model(x)
            assert out.shape == (1, 4, 256)
