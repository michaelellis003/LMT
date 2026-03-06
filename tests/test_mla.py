"""Unit tests for Multi-Head Latent Attention (MLA)."""

import torch

from lmt.layers.attention import MultiHeadLatentAttention
from lmt.models.config import ModelConfig


class TestMultiHeadLatentAttention:
    """Test suite for MultiHeadLatentAttention."""

    def _make_config(self, **kwargs):
        """Create a ModelConfig for MLA tests."""
        defaults = dict(
            embed_dim=64,
            num_heads=4,
            context_length=32,
            vocab_size=1000,
            num_layers=1,
            dropout=0.0,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)

    def test_initialization(self):
        """Test MLA initializes with valid config."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32
        )
        assert hasattr(mla, 'w_dkv')
        assert hasattr(mla, 'w_ukv')
        assert hasattr(mla, 'w_dq')
        assert hasattr(mla, 'w_uq')
        assert hasattr(mla, 'out_proj')

    def test_compression_dimensions(self):
        """Test that compression projections have correct shapes."""
        config = self._make_config(embed_dim=64, num_heads=4)
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=16, q_compress_dim=24
        )
        # Down-project: d_model -> compress_dim
        assert mla.w_dkv.in_features == 64
        assert mla.w_dkv.out_features == 16
        assert mla.w_dq.in_features == 64
        assert mla.w_dq.out_features == 24

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32
        )
        x = torch.randn(2, 16, 64)
        out = mla(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients propagate through all parameters."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32
        )
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = mla(x)
        out.sum().backward()
        for name, p in mla.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'No gradient for {name}'
                assert not torch.isnan(p.grad).any(), (
                    f'NaN gradient for {name}'
                )
        assert x.grad is not None

    def test_fewer_params_than_mha(self):
        """MLA with compression should have fewer KV params than MHA."""
        config = self._make_config(embed_dim=64, num_heads=4)
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=16, q_compress_dim=16
        )
        mla_params = sum(p.numel() for p in mla.parameters())

        from lmt.layers.attention import MultiHeadAttention

        mha = MultiHeadAttention(config)
        mha_params = sum(p.numel() for p in mha.parameters())
        # MLA should generally have fewer params with small compress_dim
        # (the savings come from not having full-size K,V projections)
        assert mla_params < mha_params

    def test_causal_masking(self):
        """Test that attention is causal."""
        config = self._make_config(embed_dim=16, num_heads=2, context_length=8)
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=8, q_compress_dim=8
        )
        mla.eval()

        x1 = torch.randn(1, 4, 16)
        x2 = x1.clone()
        x2[0, 3, :] = torch.randn(16)

        with torch.no_grad():
            out1 = mla(x1)
            out2 = mla(x2)

        for pos in range(3):
            torch.testing.assert_close(
                out1[0, pos], out2[0, pos], atol=1e-5, rtol=1e-5
            )
        assert not torch.allclose(out1[0, 3], out2[0, 3])

    def test_deterministic_eval(self):
        """Test eval mode produces identical outputs."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32
        )
        mla.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out1 = mla(x)
            out2 = mla(x)
        torch.testing.assert_close(out1, out2)

    def test_edge_case_single_token(self):
        """Test with seq_len=1."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32
        )
        x = torch.randn(1, 1, 64)
        out = mla(x)
        assert out.shape == x.shape

    def test_latent_bottleneck(self):
        """Test that very small compress dims still work.

        This tests the core idea: you can compress KV into a tiny
        latent and still get a valid output.
        """
        config = self._make_config(embed_dim=64, num_heads=4)
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=4, q_compress_dim=4
        )
        x = torch.randn(1, 8, 64)
        out = mla(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
