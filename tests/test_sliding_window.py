"""Unit tests for Sliding Window Attention."""

import torch

from lmt.layers.attention import SlidingWindowAttention
from lmt.models.config import ModelConfig


class TestSlidingWindowAttention:
    """Test suite for SlidingWindowAttention."""

    def _make_config(self, **kwargs):
        """Create a ModelConfig for sliding window tests."""
        defaults = dict(
            embed_dim=64,
            num_heads=4,
            context_length=32,
            vocab_size=1000,
            num_layers=1,
            dropout=0.0,
            window_size=4,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)

    def test_initialization(self):
        """Test that SlidingWindowAttention initializes."""
        config = self._make_config()
        attn = SlidingWindowAttention(config)
        assert attn.window_size == 4

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        config = self._make_config()
        attn = SlidingWindowAttention(config)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients propagate through all parameters."""
        config = self._make_config()
        attn = SlidingWindowAttention(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        for name, p in attn.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'No gradient for {name}'
                assert not torch.isnan(p.grad).any(), (
                    f'NaN gradient for {name}'
                )
        assert x.grad is not None

    def test_window_masking(self):
        """Test that tokens beyond the window get zero attention.

        With window_size=2, position i should only attend to
        positions max(0, i-1) through i.
        """
        config = self._make_config(
            embed_dim=16,
            num_heads=1,
            window_size=2,
            context_length=8,
        )
        attn = SlidingWindowAttention(config)
        attn.eval()

        # If we change a token far outside the window, output
        # at position 0 should be unaffected
        x1 = torch.randn(1, 6, 16)
        x2 = x1.clone()
        x2[0, 5, :] = torch.randn(16)  # Change pos 5

        with torch.no_grad():
            out1 = attn(x1)
            out2 = attn(x2)

        # Position 0-3 can't see position 5 (window=2)
        for pos in range(4):
            torch.testing.assert_close(
                out1[0, pos],
                out2[0, pos],
                atol=1e-5,
                rtol=1e-5,
            )

    def test_large_window_equals_full_attention(self):
        """Window >= seq_len should behave like full causal attention."""
        config = self._make_config(window_size=100)
        attn = SlidingWindowAttention(config)
        # With a very large window, every position attends to all
        # prior positions, same as standard causal attention
        x = torch.randn(1, 8, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_deterministic_eval(self):
        """Test eval mode produces identical outputs."""
        config = self._make_config()
        attn = SlidingWindowAttention(config)
        attn.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out1 = attn(x)
            out2 = attn(x)
        torch.testing.assert_close(out1, out2)

    def test_edge_case_single_token(self):
        """Test with seq_len=1."""
        config = self._make_config()
        attn = SlidingWindowAttention(config)
        x = torch.randn(1, 1, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_window_size_one(self):
        """With window_size=1, each token attends only to itself."""
        config = self._make_config(
            embed_dim=16,
            num_heads=1,
            window_size=1,
            context_length=8,
        )
        attn = SlidingWindowAttention(config)
        attn.eval()

        x1 = torch.randn(1, 4, 16)
        x2 = x1.clone()
        x2[0, 1, :] = torch.randn(16)

        with torch.no_grad():
            out1 = attn(x1)
            out2 = attn(x2)

        # Position 0 can't see position 1
        torch.testing.assert_close(
            out1[0, 0], out2[0, 0], atol=1e-5, rtol=1e-5
        )
        # Position 2 can't see position 1 either (window=1)
        torch.testing.assert_close(
            out1[0, 2], out2[0, 2], atol=1e-5, rtol=1e-5
        )
