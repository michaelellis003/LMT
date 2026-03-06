"""Tests for KV Cache."""

import torch

from lmt.layers.attention.kv_cache import KVCache


class TestKVCacheInit:
    """Test KVCache initialization."""

    def test_creates_empty_cache(self) -> None:
        """Cache starts with no stored keys or values."""
        cache = KVCache()
        assert cache.seq_len == 0

    def test_max_length_stored(self) -> None:
        """Max sequence length is stored."""
        cache = KVCache(max_seq_len=512)
        assert cache.max_seq_len == 512

    def test_default_max_length_none(self) -> None:
        """Default max_seq_len is None (unlimited)."""
        cache = KVCache()
        assert cache.max_seq_len is None


class TestKVCacheUpdate:
    """Test updating the cache with new K/V tensors."""

    def test_first_update_stores_kv(self) -> None:
        """First update stores key and value tensors."""
        cache = KVCache()
        k = torch.randn(2, 4, 10, 32)  # [batch, heads, seq, head_dim]
        v = torch.randn(2, 4, 10, 32)

        k_out, v_out = cache.update(k, v)

        assert k_out.shape == (2, 4, 10, 32)
        assert v_out.shape == (2, 4, 10, 32)
        assert cache.seq_len == 10

    def test_second_update_concatenates(self) -> None:
        """Subsequent updates concatenate along sequence dim."""
        cache = KVCache()
        k1 = torch.randn(2, 4, 10, 32)
        v1 = torch.randn(2, 4, 10, 32)
        cache.update(k1, v1)

        k2 = torch.randn(2, 4, 1, 32)
        v2 = torch.randn(2, 4, 1, 32)
        k_out, v_out = cache.update(k2, v2)

        assert k_out.shape == (2, 4, 11, 32)
        assert v_out.shape == (2, 4, 11, 32)
        assert cache.seq_len == 11

    def test_multiple_single_token_updates(self) -> None:
        """Simulates autoregressive generation one token at a time."""
        cache = KVCache()
        # Prefill with 5 tokens
        k = torch.randn(1, 8, 5, 64)
        v = torch.randn(1, 8, 5, 64)
        cache.update(k, v)

        # Generate 3 more tokens
        for i in range(3):
            k_new = torch.randn(1, 8, 1, 64)
            v_new = torch.randn(1, 8, 1, 64)
            k_out, v_out = cache.update(k_new, v_new)
            assert k_out.shape == (1, 8, 5 + i + 1, 64)
            assert v_out.shape == (1, 8, 5 + i + 1, 64)

        assert cache.seq_len == 8

    def test_values_preserved_across_updates(self) -> None:
        """Earlier K/V values are preserved after updates."""
        cache = KVCache()
        k1 = torch.ones(1, 2, 3, 16)
        v1 = torch.ones(1, 2, 3, 16) * 2

        cache.update(k1, v1)

        k2 = torch.zeros(1, 2, 1, 16)
        v2 = torch.zeros(1, 2, 1, 16)
        k_out, v_out = cache.update(k2, v2)

        # First 3 positions should still be ones/twos
        assert torch.allclose(k_out[:, :, :3, :], k1)
        assert torch.allclose(v_out[:, :, :3, :], v1)
        # Last position should be zeros
        assert torch.allclose(k_out[:, :, 3:, :], k2)
        assert torch.allclose(v_out[:, :, 3:, :], v2)


class TestKVCacheMaxLength:
    """Test cache eviction when max_seq_len is set."""

    def test_truncates_oldest_when_full(self) -> None:
        """When cache exceeds max_seq_len, oldest entries drop."""
        cache = KVCache(max_seq_len=8)
        # Fill to max
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        cache.update(k, v)

        # Add one more token -- should drop oldest
        k_new = torch.randn(1, 4, 1, 32)
        v_new = torch.randn(1, 4, 1, 32)
        k_out, v_out = cache.update(k_new, v_new)

        assert k_out.shape == (1, 4, 8, 32)
        assert v_out.shape == (1, 4, 8, 32)
        assert cache.seq_len == 8

    def test_keeps_newest_entries(self) -> None:
        """After truncation, the newest entries remain."""
        cache = KVCache(max_seq_len=4)
        k = torch.arange(5).float().view(1, 1, 5, 1)
        v = torch.arange(5).float().view(1, 1, 5, 1)
        k_out, v_out = cache.update(k, v)

        # Should keep positions 1,2,3,4 (drop 0)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 4, 1)
        assert torch.allclose(k_out, expected)
        assert torch.allclose(v_out, expected)

    def test_no_truncation_when_under_limit(self) -> None:
        """No truncation when cache is within max_seq_len."""
        cache = KVCache(max_seq_len=100)
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        k_out, v_out = cache.update(k, v)

        assert k_out.shape == (1, 4, 10, 32)
        assert cache.seq_len == 10


class TestKVCacheReset:
    """Test cache reset functionality."""

    def test_reset_clears_cache(self) -> None:
        """Reset empties the cache."""
        cache = KVCache()
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        cache.update(k, v)

        cache.reset()

        assert cache.seq_len == 0

    def test_usable_after_reset(self) -> None:
        """Cache works normally after reset."""
        cache = KVCache()
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        cache.update(k, v)

        cache.reset()

        k2 = torch.randn(1, 4, 5, 32)
        v2 = torch.randn(1, 4, 5, 32)
        k_out, v_out = cache.update(k2, v2)
        assert k_out.shape == (1, 4, 5, 32)
        assert cache.seq_len == 5


class TestKVCacheProperties:
    """Test cache property accessors."""

    def test_seq_len_empty(self) -> None:
        """Seq len is 0 when empty."""
        cache = KVCache()
        assert cache.seq_len == 0

    def test_seq_len_after_updates(self) -> None:
        """Seq len reflects total cached sequence length."""
        cache = KVCache()
        cache.update(torch.randn(1, 2, 7, 16), torch.randn(1, 2, 7, 16))
        assert cache.seq_len == 7

        cache.update(torch.randn(1, 2, 3, 16), torch.randn(1, 2, 3, 16))
        assert cache.seq_len == 10
