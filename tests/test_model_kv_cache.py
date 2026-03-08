"""Tests for model-level KV cache API.

The model-level API manages KV caches across all attention layers,
enabling efficient autoregressive generation. Without it, generation
recomputes K/V for all tokens at every step (O(n^2) total work).
With it, each step only processes the new token (O(n) total).
"""

import torch

from lmt.layers.attention.kv_cache import KVCache
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig


def _make_tiny_model() -> BaseModel:
    """Create a small model for testing."""
    config = ModelConfig(
        embed_dim=32,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        ffn_hidden_dim=64,
        vocab_size=64,
        context_length=128,
        tie_weights=True,
        qk_norm=False,
        dropout=0.0,
    )
    return BaseModel(config)


class TestModelKVCacheAPI:
    """Test enable/disable/reset KV cache on BaseModel."""

    def test_enable_kv_cache(self):
        """Enabling KV cache should set cache on all attention layers."""
        model = _make_tiny_model()
        model.enable_kv_cache()

        for block in model.blocks:
            assert block.attn.kv_cache is not None
            assert isinstance(block.attn.kv_cache, KVCache)

    def test_disable_kv_cache(self):
        """Disabling KV cache should remove cache from all layers."""
        model = _make_tiny_model()
        model.enable_kv_cache()
        model.disable_kv_cache()

        for block in model.blocks:
            assert block.attn.kv_cache is None

    def test_reset_kv_cache(self):
        """Resetting should clear stored K/V but keep cache active."""
        model = _make_tiny_model()
        model.enable_kv_cache()

        # Run a forward pass to populate cache
        x = torch.randint(0, 64, (1, 10))
        model.eval()
        with torch.no_grad():
            model(x)

        # Cache should have content
        for block in model.blocks:
            assert block.attn.kv_cache.seq_len > 0

        model.reset_kv_cache()

        # Cache should be empty but still active
        for block in model.blocks:
            assert block.attn.kv_cache is not None
            assert block.attn.kv_cache.seq_len == 0

    def test_enable_with_max_seq_len(self):
        """Should pass max_seq_len to all caches."""
        model = _make_tiny_model()
        model.enable_kv_cache(max_seq_len=256)

        for block in model.blocks:
            assert block.attn.kv_cache.max_seq_len == 256

    def test_cached_output_matches_full(self):
        """Cached generation should produce same logits as full pass."""
        model = _make_tiny_model()
        model.eval()

        tokens = torch.randint(0, 64, (1, 20))

        # Full forward pass (no cache)
        with torch.no_grad():
            full_logits = model(tokens)

        # Cached: prefill with first 15, then 1-token steps
        model.enable_kv_cache()
        with torch.no_grad():
            # Prefill
            prefill_logits = model(tokens[:, :15])

            # Generate token by token
            step_logits = []
            for i in range(15, 20):
                out = model(tokens[:, i : i + 1])
                step_logits.append(out)
            step_logits = torch.cat(step_logits, dim=1)

        # Combine prefill + step logits
        cached_logits = torch.cat([prefill_logits, step_logits], dim=1)

        # Should match full pass exactly
        assert torch.allclose(full_logits, cached_logits, atol=1e-5)

        model.disable_kv_cache()

    def test_double_enable_resets(self):
        """Calling enable_kv_cache twice should reset the cache."""
        model = _make_tiny_model()
        model.enable_kv_cache()

        # Populate cache
        model.eval()
        with torch.no_grad():
            model(torch.randint(0, 64, (1, 10)))

        # Re-enable should give fresh cache
        model.enable_kv_cache()
        for block in model.blocks:
            assert block.attn.kv_cache.seq_len == 0


class TestCachedGeneration:
    """Test generate_responses with KV cache."""

    def test_cached_generation_shape(self):
        """Cached generation should produce same shape as uncached."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _make_tiny_model()
        prompt = torch.randint(0, 64, (5,))
        gen_config = GenerationConfig(
            group_size=4,
            max_response_len=10,
            temperature=0.7,
            top_k=10,
        )

        # Uncached
        model.disable_kv_cache()
        uncached = generate_responses(model, prompt, gen_config)

        # Cached
        cached = generate_responses(
            model, prompt, gen_config, use_kv_cache=True
        )

        assert uncached.shape == cached.shape
        assert uncached.shape == (4, 15)  # group_size x (prompt + response)

    def test_cached_generation_preserves_prompt(self):
        """Prompt tokens should be preserved in cached generation."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _make_tiny_model()
        prompt = torch.randint(0, 64, (8,))
        gen_config = GenerationConfig(
            group_size=2,
            max_response_len=5,
            temperature=1.0,
        )

        result = generate_responses(
            model, prompt, gen_config, use_kv_cache=True
        )

        # All sequences should start with the prompt
        for i in range(2):
            assert torch.equal(result[i, :8], prompt)

    def test_cached_generation_cleans_up(self):
        """KV cache should be disabled after generation completes."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _make_tiny_model()
        prompt = torch.randint(0, 64, (5,))
        gen_config = GenerationConfig(group_size=2, max_response_len=3)

        generate_responses(model, prompt, gen_config, use_kv_cache=True)

        # Cache should be removed after generation
        for block in model.blocks:
            assert block.attn.kv_cache is None

    def test_cached_generation_with_stop_token(self):
        """Cached generation should handle stop tokens correctly."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _make_tiny_model()
        prompt = torch.randint(1, 64, (5,))  # avoid 0 (pad)
        gen_config = GenerationConfig(
            group_size=2,
            max_response_len=10,
            temperature=1.0,
            stop_token_id=63,  # unlikely but possible
            pad_token_id=0,
        )

        result = generate_responses(
            model, prompt, gen_config, use_kv_cache=True
        )

        # Shape should be correct even with early stopping
        assert result.shape == (2, 15)
