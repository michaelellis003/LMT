"""Tests for Flash Attention (educational tiled implementation).

Tests verify that our tiled attention produces identical results to standard
attention, while teaching the online softmax + tiling algorithm from:
Dao et al. (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness" https://arxiv.org/abs/2205.14135

The implementation is NOT a fused CUDA kernel -- it's a pure Python/PyTorch
educational implementation that demonstrates the algorithm's correctness.
"""

import torch

from lmt.models.config import ModelConfig


class TestFlashAttentionFunction:
    """Test the standalone flash_attention function (tiled algorithm)."""

    def _get_flash_attn(self):
        """Import the flash attention function."""
        from lmt.layers.attention.flash_attention import flash_attention

        return flash_attention

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Reference standard attention for comparison.

        Args:
            q: [B, H, L, D]
            k: [B, H, L, D]
            v: [B, H, L, D]
            causal: Whether to apply causal mask.

        Returns:
            Output tensor [B, H, L, D].
        """
        scale = q.shape[-1] ** -0.5
        scores = q @ k.transpose(-2, -1) * scale
        if causal:
            seq_len = q.shape[-2]
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device),
                diagonal=1,
            ).bool()
            scores.masked_fill_(mask, float('-inf'))
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return weights @ v

    def test_matches_standard_attention(self) -> None:
        """Tiled flash attention matches standard attention output."""
        flash_attention = self._get_flash_attn()
        torch.manual_seed(42)
        b, h, seq, d = 2, 4, 16, 32
        q = torch.randn(b, h, seq, d)
        k = torch.randn(b, h, seq, d)
        v = torch.randn(b, h, seq, d)

        expected = self._standard_attention(q, k, v, causal=True)
        result = flash_attention(q, k, v, block_size=4, causal=True)

        assert torch.allclose(expected, result, atol=1e-5), (
            f'Max diff: {(expected - result).abs().max().item()}'
        )

    def test_matches_noncausal(self) -> None:
        """Tiled flash attention matches standard non-causal attention."""
        flash_attention = self._get_flash_attn()
        torch.manual_seed(123)
        b, h, seq, d = 2, 4, 16, 32
        q = torch.randn(b, h, seq, d)
        k = torch.randn(b, h, seq, d)
        v = torch.randn(b, h, seq, d)

        expected = self._standard_attention(q, k, v, causal=False)
        result = flash_attention(q, k, v, block_size=4, causal=False)

        assert torch.allclose(expected, result, atol=1e-5)

    def test_different_block_sizes(self) -> None:
        """Results are identical regardless of block size."""
        flash_attention = self._get_flash_attn()
        torch.manual_seed(7)
        b, h, seq, d = 1, 2, 32, 16
        q = torch.randn(b, h, seq, d)
        k = torch.randn(b, h, seq, d)
        v = torch.randn(b, h, seq, d)

        ref = self._standard_attention(q, k, v, causal=True)
        for bs in [4, 8, 16, 32]:
            result = flash_attention(q, k, v, block_size=bs, causal=True)
            assert torch.allclose(ref, result, atol=1e-5), (
                f'Block size {bs} failed, max diff: '
                f'{(ref - result).abs().max().item()}'
            )

    def test_single_token(self) -> None:
        """Handles single-token sequences."""
        flash_attention = self._get_flash_attn()
        q = torch.randn(1, 1, 1, 32)
        k = torch.randn(1, 1, 1, 32)
        v = torch.randn(1, 1, 1, 32)

        result = flash_attention(q, k, v, block_size=4, causal=True)
        expected = self._standard_attention(q, k, v, causal=True)
        assert torch.allclose(expected, result, atol=1e-5)

    def test_block_size_larger_than_seq(self) -> None:
        """Block size larger than sequence length still works."""
        flash_attention = self._get_flash_attn()
        torch.manual_seed(42)
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)

        result = flash_attention(q, k, v, block_size=64, causal=True)
        expected = self._standard_attention(q, k, v, causal=True)
        assert torch.allclose(expected, result, atol=1e-5)

    def test_non_divisible_seq_len(self) -> None:
        """Sequence length not divisible by block size works correctly."""
        flash_attention = self._get_flash_attn()
        torch.manual_seed(42)
        # seq_len=13 is not divisible by block_size=4
        q = torch.randn(1, 2, 13, 16)
        k = torch.randn(1, 2, 13, 16)
        v = torch.randn(1, 2, 13, 16)

        result = flash_attention(q, k, v, block_size=4, causal=True)
        expected = self._standard_attention(q, k, v, causal=True)
        assert torch.allclose(expected, result, atol=1e-5)

    def test_gradient_flow(self) -> None:
        """Gradients flow through flash attention."""
        flash_attention = self._get_flash_attn()
        q = torch.randn(1, 2, 8, 16, requires_grad=True)
        k = torch.randn(1, 2, 8, 16, requires_grad=True)
        v = torch.randn(1, 2, 8, 16, requires_grad=True)

        out = flash_attention(q, k, v, block_size=4, causal=True)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.all(q.grad == 0)

    def test_output_shape(self) -> None:
        """Output shape matches input Q shape."""
        flash_attention = self._get_flash_attn()
        b, h, seq, d = 3, 4, 20, 64
        q = torch.randn(b, h, seq, d)
        k = torch.randn(b, h, seq, d)
        v = torch.randn(b, h, seq, d)

        out = flash_attention(q, k, v, block_size=8, causal=True)
        assert out.shape == (b, h, seq, d)

    def test_numerical_stability(self) -> None:
        """Flash attention handles large scores without NaN/Inf."""
        flash_attention = self._get_flash_attn()
        # Large values that would cause overflow in naive softmax
        q = torch.randn(1, 1, 8, 16) * 10.0
        k = torch.randn(1, 1, 8, 16) * 10.0
        v = torch.randn(1, 1, 8, 16)

        out = flash_attention(q, k, v, block_size=4, causal=True)
        assert torch.all(torch.isfinite(out))


class TestFlashAttentionLayer:
    """Test the FlashAttention nn.Module layer."""

    def _make_layer(self, **kwargs):
        """Create a FlashAttention layer for testing."""
        from lmt.layers.attention.flash_attention import FlashAttention

        config = ModelConfig(
            vocab_size=256,
            embed_dim=kwargs.get('embed_dim', 64),
            num_heads=kwargs.get('num_heads', 4),
            num_layers=2,
            context_length=128,
            dropout=0.0,
        )
        block_size = kwargs.get('block_size', 16)
        return FlashAttention(config, block_size=block_size)

    def test_output_shape(self) -> None:
        """Layer output is [B, L, embed_dim]."""
        layer = self._make_layer(embed_dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_matches_mha(self) -> None:
        """FlashAttention layer matches MultiHeadAttention output.

        When using the same weights, flash and standard MHA should
        produce identical results.
        """
        from lmt.layers.attention.flash_attention import FlashAttention
        from lmt.layers.attention.multihead_attention import (
            MultiHeadAttention,
        )

        config = ModelConfig(
            vocab_size=256,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            context_length=32,
            dropout=0.0,
        )

        flash = FlashAttention(config, block_size=8)
        mha = MultiHeadAttention(config)

        # Copy weights from flash to mha so we can compare
        mha.qkv_proj.weight.data.copy_(flash.qkv_proj.weight.data)
        if flash.qkv_proj.bias is not None:
            mha.qkv_proj.bias.data.copy_(flash.qkv_proj.bias.data)
        mha.out_proj.weight.data.copy_(flash.out_proj.weight.data)

        torch.manual_seed(42)
        x = torch.randn(2, 16, 64)
        flash_out = flash(x)
        mha_out = mha(x)

        assert torch.allclose(flash_out, mha_out, atol=1e-5), (
            f'Max diff: {(flash_out - mha_out).abs().max().item()}'
        )

    def test_gradient_flow(self) -> None:
        """Gradients reach all parameters."""
        layer = self._make_layer()
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f'{name} has no gradient'

    def test_single_token(self) -> None:
        """Layer handles single-token input."""
        layer = self._make_layer()
        x = torch.randn(1, 1, 64)
        out = layer(x)
        assert out.shape == (1, 1, 64)

    def test_different_block_sizes(self) -> None:
        """Different block sizes produce same output."""
        from lmt.layers.attention.flash_attention import FlashAttention

        config = ModelConfig(
            vocab_size=256,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            context_length=64,
            dropout=0.0,
        )

        flash_4 = FlashAttention(config, block_size=4)
        flash_16 = FlashAttention(config, block_size=16)

        # Share weights
        flash_16.qkv_proj.weight.data.copy_(flash_4.qkv_proj.weight.data)
        flash_16.out_proj.weight.data.copy_(flash_4.out_proj.weight.data)

        x = torch.randn(1, 20, 64)
        out_4 = flash_4(x)
        out_16 = flash_16(x)

        assert torch.allclose(out_4, out_16, atol=1e-5)

    def test_causal_masking(self) -> None:
        """Verify causal masking: changing future tokens doesn't affect output.

        This tests that token i's output only depends on tokens 0..i.
        """
        layer = self._make_layer(embed_dim=32, num_heads=2)
        layer.eval()

        x1 = torch.randn(1, 8, 32)
        x2 = x1.clone()
        # Change tokens 5-7 (future tokens for position 4)
        x2[0, 5:] = torch.randn(3, 32)

        out1 = layer(x1)
        out2 = layer(x2)

        # Outputs for tokens 0-4 should be identical
        assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-6)
        # But token 5+ should differ (they see the changed tokens)
        assert not torch.allclose(out1[0, 5:], out2[0, 5:])
