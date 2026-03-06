"""Tests for MLA with KV cache integration."""

import torch

from lmt.layers.attention import KVCache, MultiHeadLatentAttention
from lmt.models.config import ModelConfig


def _make_config(**kwargs: object) -> ModelConfig:
    """Create a small config for testing."""
    defaults = {
        'vocab_size': 100,
        'embed_dim': 64,
        'num_heads': 4,
        'num_kv_heads': 4,
        'num_layers': 1,
        'context_length': 128,
        'dropout': 0.0,
    }
    defaults.update(kwargs)
    return ModelConfig(**defaults)


class TestMLACacheBasic:
    """Test basic MLA cache functionality."""

    def test_no_cache_unchanged(self) -> None:
        """Without cache, MLA works as before."""
        config = _make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32
        )
        mla.eval()
        x = torch.randn(1, 8, 64)
        out = mla(x)
        assert out.shape == (1, 8, 64)

    def test_cache_attribute_default_none(self) -> None:
        """kv_cache defaults to None."""
        config = _make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32
        )
        assert mla.kv_cache is None

    def test_prefill_populates_cache(self) -> None:
        """Forward with cache populates it."""
        config = _make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32
        )
        mla.kv_cache = KVCache()
        mla.eval()

        mla(torch.randn(1, 6, 64))
        assert mla.kv_cache.seq_len == 6


class TestMLACacheGeneration:
    """Test autoregressive generation with MLA cache."""

    def test_single_token_generation(self) -> None:
        """Single token with cache produces correct shape."""
        config = _make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32
        )
        mla.kv_cache = KVCache()
        mla.eval()

        mla(torch.randn(1, 5, 64))
        out = mla(torch.randn(1, 1, 64))
        assert out.shape == (1, 1, 64)
        assert mla.kv_cache.seq_len == 6

    def test_cached_output_matches_full(self) -> None:
        """Cached generation matches full-sequence computation."""
        config = _make_config()
        mla = MultiHeadLatentAttention(
            config, kv_compress_dim=32, q_compress_dim=32
        )
        mla.eval()

        torch.manual_seed(42)
        x_full = torch.randn(1, 6, 64)
        out_full = mla(x_full)

        mla.kv_cache = KVCache()
        torch.manual_seed(42)
        x_full2 = torch.randn(1, 6, 64)
        mla(x_full2[:, :5, :])
        out_gen = mla(x_full2[:, 5:6, :])

        assert torch.allclose(out_gen, out_full[:, 5:6, :], atol=1e-5), (
            'Cached MLA output differs from full computation'
        )


class TestMLACacheWithRoPE:
    """Test MLA cache with decoupled RoPE."""

    def test_rope_cached_output_matches_full(self) -> None:
        """Cached generation with RoPE matches full-sequence."""
        config = _make_config()
        mla = MultiHeadLatentAttention(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
        )
        mla.eval()

        torch.manual_seed(42)
        x_full = torch.randn(1, 6, 64)
        out_full = mla(x_full)

        mla.kv_cache = KVCache()
        torch.manual_seed(42)
        x_full2 = torch.randn(1, 6, 64)
        mla(x_full2[:, :5, :])
        out_gen = mla(x_full2[:, 5:6, :])

        assert torch.allclose(out_gen, out_full[:, 5:6, :], atol=1e-5), (
            'Cached MLA+RoPE output differs from full computation'
        )

    def test_multiple_generation_steps_with_rope(self) -> None:
        """Multiple generation steps work with RoPE cache."""
        config = _make_config()
        mla = MultiHeadLatentAttention(
            config,
            kv_compress_dim=32,
            q_compress_dim=32,
            rope_dim=8,
        )
        mla.kv_cache = KVCache()
        mla.eval()

        mla(torch.randn(1, 4, 64))
        for i in range(3):
            out = mla(torch.randn(1, 1, 64))
            assert out.shape == (1, 1, 64)
            assert mla.kv_cache.seq_len == 5 + i
