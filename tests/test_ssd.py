r"""Tests for the Structured State Space Duality (SSD) layer.

Validates the core claim from Dao & Gu (2024, "Transformers are SSMs"):
the recurrent (linear) form and attention-like (quadratic) form of an
SSM with scalar decay produce **identical** outputs.

This is a direct replication of the paper's central theoretical result.

Reference: https://arxiv.org/abs/2405.21060
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


class TestSSD:
    """Test the SSD attention layer."""

    def _make_layer(self, embed_dim: int = 64, num_heads: int = 4):
        """Create an SSD layer."""
        from lmt.layers.attention.ssd import SSDAttention

        config = _make_config(embed_dim=embed_dim, num_heads=num_heads)
        return SSDAttention(config)

    def test_output_shape(self) -> None:
        """Output shape matches input: [B, L, d_model]."""
        layer = self._make_layer()
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

    def test_causal(self) -> None:
        """Output at position t depends only on positions <= t."""
        layer = self._make_layer()
        layer.eval()
        x = torch.randn(1, 8, 64)
        y_full = layer(x)
        y_short = layer(x[:, :4])
        assert torch.allclose(y_full[:, :4], y_short, atol=1e-5)

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

    def test_duality_recurrent_vs_quadratic(self) -> None:
        """THE CORE TEST: recurrent and quadratic forms produce same output.

        This verifies the central claim of the Mamba-2 paper:
        Y_recurrent == Y_quadratic for scalar-decay SSMs.
        """
        from lmt.layers.attention.ssd import SSDAttention

        torch.manual_seed(42)
        config = _make_config(embed_dim=32, num_heads=4)
        layer = SSDAttention(config)
        layer.eval()

        x = torch.randn(2, 16, 32)

        y_recurrent = layer.forward_recurrent(x)
        y_quadratic = layer.forward_quadratic(x)

        assert torch.allclose(y_recurrent, y_quadratic, atol=1e-5), (
            f'Duality violated! Max diff: '
            f'{(y_recurrent - y_quadratic).abs().max().item():.8f}'
        )

    def test_duality_single_token(self) -> None:
        """Duality holds for single-token input."""
        from lmt.layers.attention.ssd import SSDAttention

        torch.manual_seed(123)
        config = _make_config(embed_dim=32, num_heads=4)
        layer = SSDAttention(config)
        layer.eval()

        x = torch.randn(1, 1, 32)

        y_rec = layer.forward_recurrent(x)
        y_quad = layer.forward_quadratic(x)

        assert torch.allclose(y_rec, y_quad, atol=1e-5)

    def test_duality_long_sequence(self) -> None:
        """Duality holds for longer sequences too."""
        from lmt.layers.attention.ssd import SSDAttention

        torch.manual_seed(7)
        config = _make_config(embed_dim=32, num_heads=4)
        layer = SSDAttention(config)
        layer.eval()

        x = torch.randn(1, 64, 32)

        y_rec = layer.forward_recurrent(x)
        y_quad = layer.forward_quadratic(x)

        assert torch.allclose(y_rec, y_quad, atol=1e-4), (
            f'Duality violated on long seq! Max diff: '
            f'{(y_rec - y_quad).abs().max().item():.8f}'
        )

    def test_alpha_one_is_causal_linear_attention(self) -> None:
        """When all decay factors are 1, SSD equals causal linear attention.

        L becomes the standard causal mask (all 1s below diagonal), so
        Y = (mask ⊙ Q·K^T) · V, which is exactly causal linear attention.
        """
        from lmt.layers.attention.ssd import SSDAttention

        torch.manual_seed(42)
        config = _make_config(embed_dim=32, num_heads=4)
        layer = SSDAttention(config)
        layer.eval()

        # Force alpha ≈ 1 by making A_log very negative
        with torch.no_grad():
            layer.A_log.fill_(-100.0)

        x = torch.randn(1, 8, 32)

        # Quadratic form should produce causal linear attention
        y_quad = layer.forward_quadratic(x)

        # Manually compute causal linear attention
        q, k, v = layer._project_qkv(x)
        # q, k, v: [b, h, seq, d]
        scores = torch.einsum('bhid,bhjd->bhij', q, k)
        # Apply causal mask
        seq_len = x.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores * mask.unsqueeze(0).unsqueeze(0)
        y_manual = torch.einsum('bhij,bhjd->bhid', scores, v)

        # Apply the same output path
        y_manual = layer.out_norm(y_manual)
        b, _, _, _ = y_manual.shape
        y_manual = y_manual.transpose(1, 2).contiguous().view(b, seq_len, -1)
        gate = torch.nn.functional.silu(layer.gate_proj(x))
        y_manual = layer.out_proj(y_manual * gate)

        assert torch.allclose(y_quad, y_manual, atol=1e-4), (
            f'Alpha=1 should equal causal linear attention! '
            f'Max diff: {(y_quad - y_manual).abs().max().item():.8f}'
        )

    def test_decay_mask_structure(self) -> None:
        """The L matrix should be lower-triangular with cumulative products.

        L_{ij} = prod(alpha_k for k=j+1..i) for i >= j
        L_{ij} = 0 for i < j
        L_{ii} = 1
        """
        from lmt.layers.attention.ssd import SSDAttention

        torch.manual_seed(42)
        config = _make_config(embed_dim=16, num_heads=2)
        layer = SSDAttention(config)
        layer.eval()

        x = torch.randn(1, 4, 16)
        alpha = layer._compute_alpha(x)  # [1, 4, 2]

        l_mat = layer._build_decay_mask(alpha)  # [1, 2, 4, 4]

        # Check lower-triangular
        assert torch.allclose(
            l_mat[:, :, 0, 1:], torch.zeros(1, 2, 3), atol=1e-6
        )

        # Check diagonal is 1
        for i in range(4):
            assert torch.allclose(
                l_mat[:, :, i, i], torch.ones(1, 2), atol=1e-6
            )

        # Check L[2,0] = alpha[1] * alpha[2] for head 0
        h = 0
        a1 = alpha[0, 1, h].item()
        a2 = alpha[0, 2, h].item()
        expected = a1 * a2
        actual = l_mat[0, h, 2, 0].item()
        assert abs(actual - expected) < 1e-5, (
            f'L[2,0] = {actual:.6f}, expected alpha[1]*alpha[2] = {expected:.6f}'
        )
