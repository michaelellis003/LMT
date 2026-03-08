"""Tests for Bits Per Byte (BPB) evaluation metric.

BPB is a tokenizer-agnostic metric for comparing language models:
    BPB = total_nats / (ln(2) * total_bytes)

Lower is better. BPB ~1.0 means the model compresses text to about
1 bit per byte (very good for a small model).

When vocab_size=256 (byte-level), 2^BPB == perplexity.
"""

import math

import torch
import torch.nn as nn

from lmt.models.config import ModelConfig


def _make_tiny_model():
    """Create a tiny model for BPB testing."""
    from lmt.models.base import BaseModel

    config = ModelConfig(
        embed_dim=32,
        num_heads=2,
        num_kv_heads=2,
        context_length=16,
        vocab_size=64,
        num_layers=1,
        dropout=0.0,
    )
    return BaseModel(config), config


class TestBPB:
    """Test bits-per-byte computation."""

    def test_returns_finite_float(self):
        """BPB should return a finite positive float."""
        from lmt.eval.bpb import compute_bpb

        model, config = _make_tiny_model()
        # tokens: [num_sequences, seq_len]
        tokens = torch.randint(0, config.vocab_size, (2, 16))
        # bytes_per_token: average bytes consumed per token
        bpb = compute_bpb(model, tokens, bytes_per_token=2.5)

        assert isinstance(bpb, float)
        assert math.isfinite(bpb)
        assert bpb > 0

    def test_better_model_has_lower_bpb(self):
        """A model that memorizes data should have lower BPB."""
        from lmt.eval.bpb import compute_bpb

        model, config = _make_tiny_model()
        tokens = torch.randint(0, config.vocab_size, (1, 16))

        bpb_before = compute_bpb(model, tokens, bytes_per_token=2.0)

        # Overfit on the data
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(100):
            logits = model(tokens[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                tokens[:, 1:].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bpb_after = compute_bpb(model, tokens, bytes_per_token=2.0)

        assert bpb_after < bpb_before

    def test_bpb_relationship_to_perplexity(self):
        """For byte-level models, 2^BPB should approximate perplexity."""
        from lmt.eval.bpb import compute_bpb
        from lmt.training.eval import compute_perplexity

        # Use a byte-level vocab (256 = 1 byte per token)
        config = ModelConfig(
            embed_dim=32,
            num_heads=2,
            num_kv_heads=2,
            context_length=16,
            vocab_size=256,
            num_layers=1,
            dropout=0.0,
        )
        from lmt.models.base import BaseModel

        model = BaseModel(config)

        tokens = torch.randint(0, 256, (2, 16))

        ppl = compute_perplexity(model, tokens)
        # bytes_per_token=1 for byte-level tokenizer
        bpb = compute_bpb(model, tokens, bytes_per_token=1.0)

        # 2^BPB should equal perplexity for byte-level models
        ppl_from_bpb = 2**bpb
        assert abs(ppl_from_bpb - ppl) / ppl < 0.05, (
            f'2^BPB={ppl_from_bpb:.2f} != ppl={ppl:.2f}'
        )

    def test_different_bytes_per_token(self):
        """Higher bytes_per_token should give lower BPB.

        BPB = nats / (ln2 * total_bytes). More bytes means lower BPB
        for the same total nats (model sees same tokens).
        """
        from lmt.eval.bpb import compute_bpb

        model, config = _make_tiny_model()
        tokens = torch.randint(0, config.vocab_size, (1, 16))

        bpb_2 = compute_bpb(model, tokens, bytes_per_token=2.0)
        bpb_4 = compute_bpb(model, tokens, bytes_per_token=4.0)

        # Same nats, twice the bytes -> half the BPB
        assert abs(bpb_2 / bpb_4 - 2.0) < 0.01

    def test_empty_input_raises(self):
        """Empty token input should raise ValueError."""
        import pytest

        from lmt.eval.bpb import compute_bpb

        model, config = _make_tiny_model()
        tokens = torch.randint(0, config.vocab_size, (0, 16))

        with pytest.raises(ValueError, match='[Ee]mpty'):
            compute_bpb(model, tokens, bytes_per_token=2.0)
