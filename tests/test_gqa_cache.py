"""Tests for GQA with KV cache integration."""

import torch

from lmt.layers.attention import GroupedQueryAttention, KVCache
from lmt.models.config import ModelConfig


def _make_config(**kwargs: object) -> ModelConfig:
    """Create a small config for testing."""
    defaults = {
        'vocab_size': 100,
        'embed_dim': 64,
        'num_heads': 4,
        'num_kv_heads': 2,
        'num_layers': 1,
        'context_length': 128,
        'dropout': 0.0,
    }
    defaults.update(kwargs)
    return ModelConfig(**defaults)


class TestGQACacheBasic:
    """Test basic cache functionality."""

    def test_no_cache_unchanged(self) -> None:
        """Without cache set, forward works as before."""
        config = _make_config()
        gqa = GroupedQueryAttention(config)
        gqa.eval()
        x = torch.randn(1, 10, 64)
        out = gqa(x)
        assert out.shape == (1, 10, 64)

    def test_cache_attribute_default_none(self) -> None:
        """kv_cache defaults to None."""
        config = _make_config()
        gqa = GroupedQueryAttention(config)
        assert gqa.kv_cache is None

    def test_set_cache(self) -> None:
        """Can set a KVCache on the module."""
        config = _make_config()
        gqa = GroupedQueryAttention(config)
        gqa.kv_cache = KVCache()
        assert gqa.kv_cache is not None

    def test_prefill_populates_cache(self) -> None:
        """Forward with cache set populates the cache."""
        config = _make_config()
        gqa = GroupedQueryAttention(config)
        gqa.kv_cache = KVCache()
        gqa.eval()

        x = torch.randn(1, 8, 64)
        gqa(x)

        assert gqa.kv_cache.seq_len == 8


class TestGQACacheGeneration:
    """Test autoregressive generation with cache."""

    def test_single_token_generation(self) -> None:
        """Single token with cache produces correct shape."""
        config = _make_config()
        gqa = GroupedQueryAttention(config)
        gqa.kv_cache = KVCache()
        gqa.eval()

        # Prefill
        x = torch.randn(1, 5, 64)
        out1 = gqa(x)
        assert out1.shape == (1, 5, 64)
        assert gqa.kv_cache.seq_len == 5

        # Generate one token
        x_new = torch.randn(1, 1, 64)
        out2 = gqa(x_new)
        assert out2.shape == (1, 1, 64)
        assert gqa.kv_cache.seq_len == 6

    def test_multiple_generation_steps(self) -> None:
        """Multiple generation steps accumulate in cache."""
        config = _make_config()
        gqa = GroupedQueryAttention(config)
        gqa.kv_cache = KVCache()
        gqa.eval()

        # Prefill with 4 tokens
        gqa(torch.randn(1, 4, 64))
        assert gqa.kv_cache.seq_len == 4

        # Generate 3 more
        for i in range(3):
            out = gqa(torch.randn(1, 1, 64))
            assert out.shape == (1, 1, 64)
            assert gqa.kv_cache.seq_len == 5 + i

    def test_cached_output_matches_full(self) -> None:
        """Cached generation matches full-sequence computation."""
        config = _make_config()
        gqa = GroupedQueryAttention(config)
        gqa.eval()

        # Full sequence
        torch.manual_seed(42)
        x_full = torch.randn(1, 6, 64)
        out_full = gqa(x_full)

        # Cached: prefill then generate
        gqa.kv_cache = KVCache()
        torch.manual_seed(42)
        x_full2 = torch.randn(1, 6, 64)

        # Prefill with first 5 tokens
        gqa(x_full2[:, :5, :])

        # Generate 6th token
        out_gen = gqa(x_full2[:, 5:6, :])

        # The 6th token output should match
        assert torch.allclose(out_gen, out_full[:, 5:6, :], atol=1e-5), (
            'Cached generation output differs from full computation'
        )


class TestGQACacheReset:
    """Test cache reset behavior."""

    def test_reset_cache(self) -> None:
        """Setting kv_cache to None clears it."""
        config = _make_config()
        gqa = GroupedQueryAttention(config)
        gqa.kv_cache = KVCache()
        gqa.eval()

        gqa(torch.randn(1, 5, 64))
        assert gqa.kv_cache.seq_len == 5

        gqa.kv_cache = None
        assert gqa.kv_cache is None

    def test_fresh_cache_after_reset(self) -> None:
        """A new cache starts empty after reset."""
        config = _make_config()
        gqa = GroupedQueryAttention(config)
        gqa.eval()

        # First generation session
        gqa.kv_cache = KVCache()
        gqa(torch.randn(1, 5, 64))

        # Reset and start new session
        gqa.kv_cache = KVCache()
        assert gqa.kv_cache.seq_len == 0

        gqa(torch.randn(1, 3, 64))
        assert gqa.kv_cache.seq_len == 3
