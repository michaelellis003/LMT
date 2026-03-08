"""Tests for ReluSquaredGLU feed-forward network.

ReLU² GLU uses squared ReLU gating instead of SiLU (SwiGLU).
Key properties to verify:
- Sparser activations than SwiGLU (ReLU kills negatives, squaring
  further suppresses small values)
- Same interface as SwiGLU (drop-in replacement)
- Gradient flow works (squaring doesn't kill gradients for active neurons)
"""

import torch
import torch.nn as nn


class TestReluSquaredGLU:
    """Test the ReluSquaredGLU feed-forward network."""

    def test_initialization(self):
        """Should initialize with d_model and optional hidden_dim."""
        from lmt.layers.ffn.relu_squared import ReluSquaredGLU

        ffn = ReluSquaredGLU(d_model=64, hidden_dim=128)
        assert isinstance(ffn, nn.Module)

    def test_default_hidden_dim(self):
        """Default hidden_dim should be int(2/3 * 4 * d_model)."""
        from lmt.layers.ffn.relu_squared import ReluSquaredGLU

        ffn = ReluSquaredGLU(d_model=64)
        # Check w1 shape to infer hidden_dim
        expected = int(2 / 3 * 4 * 64)
        assert ffn.w1.out_features == expected

    def test_forward_shape(self):
        """Output shape should match input shape."""
        from lmt.layers.ffn.relu_squared import ReluSquaredGLU

        ffn = ReluSquaredGLU(d_model=64, hidden_dim=128)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_gradient_flow(self):
        """Gradients should flow through the module."""
        from lmt.layers.ffn.relu_squared import ReluSquaredGLU

        ffn = ReluSquaredGLU(d_model=64, hidden_dim=128)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        for param in ffn.parameters():
            assert param.grad is not None

    def test_no_bias(self):
        """All linear layers should be bias-free."""
        from lmt.layers.ffn.relu_squared import ReluSquaredGLU

        ffn = ReluSquaredGLU(d_model=64, hidden_dim=128)
        assert ffn.w1.bias is None
        assert ffn.wg.bias is None
        assert ffn.w2.bias is None

    def test_sparser_than_swiglu(self):
        """ReLU² should produce sparser hidden activations than SwiGLU.

        ReLU zeros out all negative values (50% sparsity expected for
        Gaussian inputs), then squaring further compresses small values.
        SiLU (in SwiGLU) is smooth and non-zero everywhere.
        """
        from lmt.layers.ffn.relu_squared import ReluSquaredGLU
        from lmt.layers.ffn.swiglu import SwiGLU

        torch.manual_seed(42)
        d_model, hidden = 64, 128

        relu_sq = ReluSquaredGLU(d_model=d_model, hidden_dim=hidden)
        swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden)

        # Use same weights for fair comparison
        with torch.no_grad():
            swiglu.wg.weight.copy_(relu_sq.wg.weight)

        x = torch.randn(8, 16, d_model)

        # Get gate activations
        with torch.no_grad():
            relu_sq_gate = torch.relu(relu_sq.wg(x)) ** 2
            swiglu_gate = torch.nn.functional.silu(swiglu.wg(x))

        # Count near-zero activations (< 1e-6)
        relu_sq_sparse = (relu_sq_gate.abs() < 1e-6).float().mean()
        swiglu_sparse = (swiglu_gate.abs() < 1e-6).float().mean()

        # ReLU² should be significantly sparser
        assert relu_sq_sparse > swiglu_sparse, (
            f'ReLU² sparsity ({relu_sq_sparse:.2%}) should exceed '
            f'SwiGLU sparsity ({swiglu_sparse:.2%})'
        )

    def test_same_interface_as_swiglu(self):
        """Should have same w1/wg/w2 structure as SwiGLU."""
        from lmt.layers.ffn.relu_squared import ReluSquaredGLU
        from lmt.layers.ffn.swiglu import SwiGLU

        relu_sq = ReluSquaredGLU(d_model=64, hidden_dim=128)
        swiglu = SwiGLU(d_model=64, hidden_dim=128)

        # Same parameter names
        relu_sq_params = {n for n, _ in relu_sq.named_parameters()}
        swiglu_params = {n for n, _ in swiglu.named_parameters()}
        assert relu_sq_params == swiglu_params

    def test_deterministic(self):
        """Same input should produce same output."""
        from lmt.layers.ffn.relu_squared import ReluSquaredGLU

        torch.manual_seed(42)
        ffn = ReluSquaredGLU(d_model=64, hidden_dim=128)
        x = torch.randn(2, 10, 64)

        out1 = ffn(x)
        out2 = ffn(x)
        assert torch.equal(out1, out2)

    def test_registered_in_ffn_registry(self):
        """Should be accessible via FFN_REGISTRY."""
        from lmt.layers.ffn import FFN_REGISTRY

        assert 'relu_squared' in FFN_REGISTRY

    def test_param_count_matches_swiglu(self):
        """Should have same number of parameters as SwiGLU."""
        from lmt.layers.ffn.relu_squared import ReluSquaredGLU
        from lmt.layers.ffn.swiglu import SwiGLU

        d_model, hidden = 64, 128
        relu_sq = ReluSquaredGLU(d_model=d_model, hidden_dim=hidden)
        swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden)

        relu_sq_params = sum(p.numel() for p in relu_sq.parameters())
        swiglu_params = sum(p.numel() for p in swiglu.parameters())
        assert relu_sq_params == swiglu_params
