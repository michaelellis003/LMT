r"""Tests for Gated Delta Networks (linear attention variant).

Reference: Yang et al., 2024 -- "Gated Delta Networks: Improving
Mamba2 with Delta Rule" https://arxiv.org/abs/2412.06464

The delta rule treats the state matrix S as associative memory updated
via gradient descent: S_t = S_{t-1} - \beta (S_{t-1} k_t - v_t) k_t^T.
Gating adds a decay factor \alpha for fast memory erasure.
"""

import torch

from lmt.models.config import ModelConfig


def _make_config(
    embed_dim: int = 64,
    num_heads: int = 4,
    context_length: int = 128,
) -> ModelConfig:
    """Create a small config for testing."""
    return ModelConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=1,
        vocab_size=256,
        context_length=context_length,
        dropout=0.0,
        qkv_bias=False,
    )


class TestGatedDeltaNet:
    """Test the GatedDeltaNet attention layer."""

    def _make_layer(self, embed_dim: int = 64, num_heads: int = 4):
        """Create a GatedDeltaNet layer."""
        from lmt.layers.attention.gated_delta_net import GatedDeltaNet

        config = _make_config(embed_dim=embed_dim, num_heads=num_heads)
        return GatedDeltaNet(config)

    def test_output_shape(self) -> None:
        """Output shape matches input: [B, L, d_model]."""
        layer = self._make_layer(embed_dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        y = layer(x)
        assert y.shape == (2, 16, 64)

    def test_single_token(self) -> None:
        """Handles single-token sequences."""
        layer = self._make_layer()
        x = torch.randn(1, 1, 64)
        y = layer(x)
        assert y.shape == (1, 1, 64)

    def test_gradient_flow(self) -> None:
        """Gradients flow through the layer."""
        layer = self._make_layer()
        x = torch.randn(2, 8, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_decay_gate_range(self) -> None:
        """Decay gate alpha should be in (0, 1]."""
        from lmt.layers.attention.gated_delta_net import GatedDeltaNet

        config = _make_config()
        layer = GatedDeltaNet(config)
        x = torch.randn(2, 8, 64)
        alpha = layer._compute_alpha(x)
        assert torch.all(alpha > 0), 'alpha must be positive'
        assert torch.all(alpha <= 1.0 + 1e-6), 'alpha must be <= 1'

    def test_update_gate_range(self) -> None:
        """Update gate beta should be in (0, 1)."""
        from lmt.layers.attention.gated_delta_net import GatedDeltaNet

        config = _make_config()
        layer = GatedDeltaNet(config)
        x = torch.randn(2, 8, 64)
        beta = layer._compute_beta(x)
        assert torch.all(beta >= 0), 'beta must be non-negative'
        assert torch.all(beta <= 1), 'beta must be <= 1'

    def test_qk_normalized(self) -> None:
        """Q and K should be L2-normalized for stability."""
        from lmt.layers.attention.gated_delta_net import GatedDeltaNet

        config = _make_config(embed_dim=64, num_heads=4)
        layer = GatedDeltaNet(config)
        x = torch.randn(2, 8, 64)
        q, k, _ = layer._project_qkv(x)
        # After L2 norm, vectors should have unit norm along last dim
        q_norms = q.norm(dim=-1)
        k_norms = k.norm(dim=-1)
        # q is scaled by 1/sqrt(head_dim), so norm = 1/sqrt(head_dim)
        expected = 1.0 / (layer.head_dim**0.5)
        assert torch.allclose(
            q_norms, torch.full_like(q_norms, expected), atol=1e-5
        )
        assert torch.allclose(k_norms, torch.ones_like(k_norms), atol=1e-5)

    def test_different_head_counts(self) -> None:
        """Works with different numbers of heads."""
        for num_heads in [1, 2, 8]:
            layer = self._make_layer(embed_dim=64, num_heads=num_heads)
            x = torch.randn(1, 8, 64)
            y = layer(x)
            assert y.shape == (1, 8, 64)

    def test_causal(self) -> None:
        """Output at position t depends only on positions <= t."""
        layer = self._make_layer()
        layer.eval()
        x = torch.randn(1, 8, 64)
        y_full = layer(x)
        # Run only first 4 tokens
        y_short = layer(x[:, :4])
        # First 4 positions should match (causal property)
        assert torch.allclose(y_full[:, :4], y_short, atol=1e-5)

    def test_param_count(self) -> None:
        """Layer has expected parameter groups."""
        layer = self._make_layer(embed_dim=64, num_heads=4)
        param_names = {n for n, _ in layer.named_parameters()}
        # Should have: q, k, v projections, alpha gate, beta gate,
        # output gate, output projection, output norm, A_log, dt_bias
        assert any('q_proj' in n for n in param_names)
        assert any('k_proj' in n for n in param_names)
        assert any('v_proj' in n for n in param_names)
        assert any('out_proj' in n for n in param_names)

    def test_deterministic(self) -> None:
        """Same input produces same output."""
        layer = self._make_layer()
        layer.eval()
        x = torch.randn(1, 8, 64)
        y1 = layer(x)
        y2 = layer(x)
        assert torch.equal(y1, y2)

    def test_output_finite(self) -> None:
        """Output contains no NaN or Inf values."""
        layer = self._make_layer()
        x = torch.randn(2, 32, 64)
        y = layer(x)
        assert torch.all(torch.isfinite(y))

    def test_long_sequence(self) -> None:
        """Handles longer sequences (linear complexity advantage)."""
        layer = self._make_layer()
        x = torch.randn(1, 128, 64)
        y = layer(x)
        assert y.shape == (1, 128, 64)
        assert torch.all(torch.isfinite(y))
