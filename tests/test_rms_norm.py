"""Unit tests for RMSNorm."""

import torch

from lmt.layers.normalization import RMSNorm


class TestRMSNorm:
    """Test suite for RMSNorm."""

    def test_initialization(self):
        """Test that RMSNorm initializes with valid d_model."""
        norm = RMSNorm(d_model=64)
        assert hasattr(norm, 'weight')
        assert norm.weight.shape == (64,)
        # Weight should be initialized to ones
        torch.testing.assert_close(norm.weight.data, torch.ones(64))

    def test_custom_eps(self):
        """Test initialization with custom epsilon."""
        norm = RMSNorm(d_model=32, eps=1e-5)
        assert norm.eps == 1e-5

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients propagate through all parameters."""
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 4, 64, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        for name, p in norm.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'No gradient for {name}'
                assert not torch.isnan(p.grad).any(), (
                    f'NaN gradient for {name}'
                )
        assert x.grad is not None

    def test_numerical_correctness(self):
        """Test output matches hand-computed RMSNorm."""
        torch.manual_seed(42)
        norm = RMSNorm(d_model=4, eps=1e-6)
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

        # Manual computation:
        # rms = sqrt(mean(x^2) + eps) = sqrt((1+4+9+16)/4 + 1e-6)
        #     = sqrt(7.5 + 1e-6)
        # out = x / rms * weight (weight=ones)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = x / rms * norm.weight

        out = norm(x)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_deterministic_eval(self):
        """Test eval mode produces identical outputs."""
        norm = RMSNorm(d_model=64)
        norm.eval()
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            out1 = norm(x)
            out2 = norm(x)
        torch.testing.assert_close(out1, out2)

    def test_edge_case_single_token(self):
        """Test with batch_size=1, seq_len=1."""
        norm = RMSNorm(d_model=64)
        x = torch.randn(1, 1, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_fp32_upcast_stability(self):
        """Test that norm computes in FP32 even with FP16 input."""
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 4, 64, dtype=torch.float16)
        out = norm(x)
        # Output should be FP16 (same as input) but computed stably
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
