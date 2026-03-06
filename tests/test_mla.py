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
        assert hasattr(mla, 'w_uk')
        assert hasattr(mla, 'w_uv')
        assert hasattr(mla, 'w_dq')
        assert hasattr(mla, 'w_uq')
        assert hasattr(mla, 'out_proj')

    def test_initialization_with_rope(self):
        """Test MLA initializes with decoupled RoPE."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32, rope_dim=8
        )
        assert hasattr(mla, 'w_qr')
        assert hasattr(mla, 'w_kr')
        assert hasattr(mla, 'rope')

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

    def test_rope_dimensions(self):
        """Test decoupled RoPE projection shapes."""
        config = self._make_config(embed_dim=64, num_heads=4)
        mla = MultiHeadLatentAttention(
            config,
            kv_compress_dim=16,
            q_compress_dim=24,
            rope_dim=8,
        )
        # q_R: from q_compress_dim -> num_heads * rope_dim
        assert mla.w_qr.in_features == 24
        assert mla.w_qr.out_features == 4 * 8  # num_heads * rope_dim
        # k_R: from embed_dim -> num_heads * rope_dim (asymmetric!)
        assert mla.w_kr.in_features == 64
        assert mla.w_kr.out_features == 4 * 8

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32
        )
        x = torch.randn(2, 16, 64)
        out = mla(x)
        assert out.shape == x.shape

    def test_forward_shape_with_rope(self):
        """Test output shape with decoupled RoPE."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32, rope_dim=8
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

    def test_gradient_flow_with_rope(self):
        """Test gradients flow through decoupled RoPE path."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32, rope_dim=8
        )
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = mla(x)
        out.sum().backward()
        # RoPE projections must have gradients
        assert mla.w_qr.weight.grad is not None
        assert mla.w_kr.weight.grad is not None
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

    def test_causal_masking_with_rope(self):
        """Test causality is preserved with decoupled RoPE."""
        config = self._make_config(embed_dim=16, num_heads=2, context_length=8)
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=8, q_compress_dim=8, rope_dim=4
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

    def test_edge_case_single_token_with_rope(self):
        """Test seq_len=1 with decoupled RoPE."""
        config = self._make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32, rope_dim=8
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

    def test_separate_k_v_projections(self):
        """Test that K and V are separate projections from shared latent."""
        config = self._make_config(embed_dim=64, num_heads=4)
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=16, q_compress_dim=16
        )
        # w_uk and w_uv should be different projections
        assert not torch.equal(mla.w_uk.weight, mla.w_uv.weight)

    def test_kr_from_input_not_latent(self):
        """Test k_R projects from input x, not from KV latent.

        This is the key asymmetry in decoupled RoPE: q_R comes from
        the query latent c_Q, but k_R comes from the raw input x.
        """
        config = self._make_config(embed_dim=64, num_heads=4)
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=16, q_compress_dim=24, rope_dim=8
        )
        # k_R input is embed_dim (from x), not kv_compress_dim
        assert mla.w_kr.in_features == 64
        # q_R input is q_compress_dim (from c_Q)
        assert mla.w_qr.in_features == 24
