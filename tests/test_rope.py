"""Unit tests for Rotary Positional Encoding (RoPE)."""

import torch

from lmt.layers.positional import RoPE


class TestRoPE:
    """Test suite for RoPE."""

    def test_initialization(self):
        """Test that RoPE initializes with valid d_model."""
        rope = RoPE(d_model=64)
        assert hasattr(rope, 'cos_cache')
        assert hasattr(rope, 'sin_cache')

    def test_cache_shapes(self):
        """Test cos/sin caches have correct shape."""
        rope = RoPE(d_model=64, max_seq_len=128)
        # Caches: [max_seq_len, d_model/2] -- one freq per dim pair
        assert rope.cos_cache.shape == (128, 32)
        assert rope.sin_cache.shape == (128, 32)

    def test_forward_shape(self):
        """Test output shape matches input shape.

        RoPE forward takes x of [batch, seq, d_model] for registry
        compatibility, and returns the same shape.
        """
        rope = RoPE(d_model=64, max_seq_len=128)
        x = torch.randn(2, 16, 64)
        out = rope(x)
        assert out.shape == x.shape

    def test_apply_rotary_emb_shapes(self):
        """Test apply_rotary_emb returns correct shapes for q, k."""
        rope = RoPE(d_model=64, max_seq_len=128)
        q = torch.randn(2, 16, 64)
        k = torch.randn(2, 16, 64)
        q_rot, k_rot = rope.apply_rotary_emb(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_gradient_flow(self):
        """Test gradients propagate through RoPE."""
        rope = RoPE(d_model=64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = rope(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_learnable_params(self):
        """Test that RoPE has no learnable parameters."""
        rope = RoPE(d_model=64)
        params = list(rope.parameters())
        assert len(params) == 0

    def test_caches_are_buffers(self):
        """Test cos/sin caches are registered as buffers."""
        rope = RoPE(d_model=64)
        buffers = dict(rope.named_buffers())
        assert 'cos_cache' in buffers
        assert 'sin_cache' in buffers

    def test_different_seq_lengths(self):
        """Test RoPE works with various sequence lengths."""
        rope = RoPE(d_model=32, max_seq_len=128)
        for seq_len in [1, 4, 16, 64, 128]:
            x = torch.randn(1, seq_len, 32)
            out = rope(x)
            assert out.shape == x.shape

    def test_deterministic(self):
        """Test identical inputs produce identical outputs."""
        rope = RoPE(d_model=64)
        x = torch.randn(2, 8, 64)
        out1 = rope(x)
        out2 = rope(x)
        torch.testing.assert_close(out1, out2)

    def test_position_dependent(self):
        """Test that different positions produce different rotations.

        The same token at different positions should get different
        rotated representations.
        """
        rope = RoPE(d_model=64)
        # Create input where position 0 and position 1 have same content
        x = torch.randn(1, 1, 64)
        x_repeated = x.expand(1, 4, 64).clone()
        out = rope(x_repeated)
        # Outputs at different positions should differ
        assert not torch.allclose(out[0, 0], out[0, 1])
        assert not torch.allclose(out[0, 1], out[0, 2])

    def test_relative_position_property(self):
        """Test the key RoPE property: dot product depends on relative pos.

        For vectors at positions m and n:
        <R(m)q, R(n)k> should equal <R(m-n)q, k> (approximately).
        This is what makes RoPE encode relative position.
        """
        torch.manual_seed(42)
        rope = RoPE(d_model=16, max_seq_len=32)

        q_vec = torch.randn(1, 1, 16)
        k_vec = torch.randn(1, 1, 16)

        # Rotate q to position m=5, k to position n=3
        # relative distance = 2
        cos = rope.cos_cache
        sin = rope.sin_cache

        def rotate_at_pos(x, pos):
            """Rotate x using the cached cos/sin at given position."""
            c = cos[pos].unsqueeze(0).unsqueeze(0)  # [1, 1, d/2]
            s = sin[pos].unsqueeze(0).unsqueeze(0)
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat(
                [x1 * c - x2 * s, x2 * c + x1 * s], dim=-1
            )

        # <R(5)q, R(3)k>
        rq5 = rotate_at_pos(q_vec, 5)
        rk3 = rotate_at_pos(k_vec, 3)
        dot1 = (rq5 * rk3).sum()

        # <R(2)q, R(0)k> -- same relative distance
        rq2 = rotate_at_pos(q_vec, 2)
        rk0 = rotate_at_pos(k_vec, 0)
        dot2 = (rq2 * rk0).sum()

        torch.testing.assert_close(
            dot1, dot2, atol=1e-5, rtol=1e-5
        )

    def test_custom_base(self):
        """Test that different base frequencies produce different caches."""
        rope1 = RoPE(d_model=64, base=10000.0)
        rope2 = RoPE(d_model=64, base=500000.0)
        assert not torch.allclose(rope1.cos_cache, rope2.cos_cache)
