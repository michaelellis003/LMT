"""Unit tests for SwiGLU feed-forward network."""

import torch

from lmt.layers.ffn import SwiGLU


class TestSwiGLU:
    """Test suite for SwiGLU FFN."""

    def test_initialization(self):
        """Test that SwiGLU initializes with valid d_model."""
        ffn = SwiGLU(d_model=64)
        assert hasattr(ffn, 'w1')
        assert hasattr(ffn, 'wg')
        assert hasattr(ffn, 'w2')

    def test_default_hidden_dim(self):
        """Test default hidden_dim calculation: int(2/3 * 4 * d_model)."""
        ffn = SwiGLU(d_model=64)
        expected_hidden = int(2 / 3 * 4 * 64)  # 170
        assert ffn.w1.in_features == 64
        assert ffn.w1.out_features == expected_hidden
        assert ffn.wg.out_features == expected_hidden
        assert ffn.w2.in_features == expected_hidden

    def test_custom_hidden_dim(self):
        """Test initialization with explicit hidden_dim."""
        ffn = SwiGLU(d_model=64, hidden_dim=256)
        assert ffn.w1.out_features == 256
        assert ffn.wg.out_features == 256
        assert ffn.w2.in_features == 256

    def test_no_bias(self):
        """Test that all linear layers have no bias."""
        ffn = SwiGLU(d_model=64)
        assert ffn.w1.bias is None
        assert ffn.wg.bias is None
        assert ffn.w2.bias is None

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        ffn = SwiGLU(d_model=64)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients propagate through all parameters."""
        ffn = SwiGLU(d_model=64)
        x = torch.randn(2, 4, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        for name, p in ffn.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'No gradient for {name}'
                assert not torch.isnan(p.grad).any(), (
                    f'NaN gradient for {name}'
                )
        assert x.grad is not None

    def test_parameter_count(self):
        """Test parameter count: 3 * d_model * hidden_dim (no bias)."""
        d_model = 64
        hidden_dim = 128
        ffn = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)
        expected = 3 * d_model * hidden_dim
        actual = sum(p.numel() for p in ffn.parameters())
        assert actual == expected

    def test_deterministic_eval(self):
        """Test eval mode produces identical outputs."""
        ffn = SwiGLU(d_model=64)
        ffn.eval()
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            out1 = ffn(x)
            out2 = ffn(x)
        torch.testing.assert_close(out1, out2)

    def test_edge_case_single_token(self):
        """Test with batch_size=1, seq_len=1."""
        ffn = SwiGLU(d_model=64)
        x = torch.randn(1, 1, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_numerical_correctness(self):
        """Test forward matches manual SwiGLU computation."""
        torch.manual_seed(42)
        d_model = 4
        hidden_dim = 8
        ffn = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)
        x = torch.randn(1, 2, d_model)

        # Manual: w2(silu(wg(x)) * w1(x))
        expected = ffn.w2(torch.nn.functional.silu(ffn.wg(x)) * ffn.w1(x))
        actual = ffn(x)
        torch.testing.assert_close(actual, expected)
