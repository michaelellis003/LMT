"""Unit tests for Grouped Query Attention (GQA)."""

import torch

from lmt.layers.attention import GroupedQueryAttention
from lmt.models.config import ModelConfig


class TestGroupedQueryAttention:
    """Test suite for GroupedQueryAttention."""

    def _make_config(self, **kwargs):
        """Create a ModelConfig with sensible defaults for GQA tests."""
        defaults = dict(
            embed_dim=64,
            num_heads=8,
            num_kv_heads=2,
            context_length=32,
            vocab_size=1000,
            num_layers=1,
            dropout=0.0,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)

    def test_initialization(self):
        """Test that GQA initializes with valid config."""
        config = self._make_config()
        gqa = GroupedQueryAttention(config)
        assert hasattr(gqa, 'q_proj')
        assert hasattr(gqa, 'k_proj')
        assert hasattr(gqa, 'v_proj')
        assert hasattr(gqa, 'out_proj')

    def test_projection_dimensions(self):
        """Test Q projects to full dim, K/V to reduced dim."""
        config = self._make_config(embed_dim=64, num_heads=8, num_kv_heads=2)
        gqa = GroupedQueryAttention(config)
        head_dim = 64 // 8  # 8
        # Q: full num_heads * head_dim = 64
        assert gqa.q_proj.out_features == 8 * head_dim
        # K, V: num_kv_heads * head_dim = 16
        assert gqa.k_proj.out_features == 2 * head_dim
        assert gqa.v_proj.out_features == 2 * head_dim

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        config = self._make_config()
        gqa = GroupedQueryAttention(config)
        x = torch.randn(2, 16, 64)
        out = gqa(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients propagate through all parameters."""
        config = self._make_config()
        gqa = GroupedQueryAttention(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = gqa(x)
        out.sum().backward()
        for name, p in gqa.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'No gradient for {name}'
                assert not torch.isnan(p.grad).any(), (
                    f'NaN gradient for {name}'
                )
        assert x.grad is not None

    def test_num_kv_heads_equals_num_heads_is_mha(self):
        """When num_kv_heads == num_heads, GQA == standard MHA."""
        config = self._make_config(num_heads=8, num_kv_heads=8)
        gqa = GroupedQueryAttention(config)
        # K, V should have full dimension
        head_dim = 64 // 8
        assert gqa.k_proj.out_features == 8 * head_dim
        assert gqa.v_proj.out_features == 8 * head_dim

    def test_num_kv_heads_one_is_mqa(self):
        """When num_kv_heads == 1, GQA == Multi-Query Attention."""
        config = self._make_config(num_heads=8, num_kv_heads=1)
        gqa = GroupedQueryAttention(config)
        head_dim = 64 // 8
        assert gqa.k_proj.out_features == 1 * head_dim
        assert gqa.v_proj.out_features == 1 * head_dim
        # Should still work
        x = torch.randn(1, 8, 64)
        out = gqa(x)
        assert out.shape == x.shape

    def test_none_kv_heads_defaults_to_mha(self):
        """When num_kv_heads is None, default to num_heads (MHA)."""
        config = self._make_config(num_kv_heads=None)
        gqa = GroupedQueryAttention(config)
        head_dim = 64 // 8
        assert gqa.k_proj.out_features == 8 * head_dim

    def test_parameter_count_reduction(self):
        """GQA with fewer KV heads has fewer params than full MHA."""
        config_mha = self._make_config(num_kv_heads=8)
        config_gqa = self._make_config(num_kv_heads=2)
        mha = GroupedQueryAttention(config_mha)
        gqa = GroupedQueryAttention(config_gqa)
        mha_params = sum(p.numel() for p in mha.parameters())
        gqa_params = sum(p.numel() for p in gqa.parameters())
        assert gqa_params < mha_params

    def test_causal_masking(self):
        """Test that attention is causal (no future token leakage)."""
        config = self._make_config(
            embed_dim=16, num_heads=2, num_kv_heads=1, context_length=8
        )
        gqa = GroupedQueryAttention(config)
        gqa.eval()

        # Create two inputs that differ only at position 3
        x1 = torch.randn(1, 4, 16)
        x2 = x1.clone()
        x2[0, 3, :] = torch.randn(16)

        with torch.no_grad():
            out1 = gqa(x1)
            out2 = gqa(x2)

        # Positions 0, 1, 2 should be identical (can't see pos 3)
        for pos in range(3):
            torch.testing.assert_close(
                out1[0, pos],
                out2[0, pos],
                atol=1e-5,
                rtol=1e-5,
            )
        # Position 3 should differ
        assert not torch.allclose(out1[0, 3], out2[0, 3])

    def test_deterministic_eval(self):
        """Test eval mode produces identical outputs."""
        config = self._make_config()
        gqa = GroupedQueryAttention(config)
        gqa.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out1 = gqa(x)
            out2 = gqa(x)
        torch.testing.assert_close(out1, out2)

    def test_edge_case_single_token(self):
        """Test with batch_size=1, seq_len=1."""
        config = self._make_config()
        gqa = GroupedQueryAttention(config)
        x = torch.randn(1, 1, 64)
        out = gqa(x)
        assert out.shape == x.shape

    def test_num_heads_must_divide_by_kv_heads(self):
        """Test that num_heads must be divisible by num_kv_heads."""
        import pytest

        config = self._make_config(num_heads=8, num_kv_heads=3)
        with pytest.raises((AssertionError, ValueError)):
            GroupedQueryAttention(config)
