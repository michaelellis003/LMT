"""Tests for text generation functions.

Tests cover greedy decoding, temperature sampling, top-k filtering,
EOS stopping, and context window cropping.
"""

import torch
import torch.nn as nn

from lmt.generate import generate


class _TinyModel(nn.Module):
    """Minimal model for testing generate()."""

    def __init__(self, vocab_size: int = 16, context_length: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed = nn.Embedding(vocab_size, 8)
        self.head = nn.Linear(8, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Return logits [batch, seq_len, vocab_size]."""
        return self.head(self.embed(idx))


class _ConstantModel(nn.Module):
    """Model that always predicts a fixed token."""

    def __init__(self, vocab_size: int = 16, predict_token: int = 5):
        super().__init__()
        self.vocab_size = vocab_size
        self.predict_token = predict_token

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Return logits strongly favoring predict_token."""
        b, seq_len = idx.shape
        logits = torch.full((b, seq_len, self.vocab_size), -100.0)
        logits[:, :, self.predict_token] = 100.0
        return logits


class TestGenerate:
    """Test the generate() function."""

    def test_output_length_greedy(self) -> None:
        """Greedy decoding produces exactly max_new_tokens."""
        model = _TinyModel(vocab_size=16)
        model.eval()
        prompt = torch.randint(0, 16, (1, 4))
        result = generate(
            model,
            prompt,
            max_new_tokens=10,
            context_size=32,
            temperature=0.0,
        )
        assert result.shape == (1, 14)  # 4 prompt + 10 generated

    def test_output_length_sampling(self) -> None:
        """Temperature sampling produces exactly max_new_tokens."""
        model = _TinyModel(vocab_size=16)
        model.eval()
        prompt = torch.randint(0, 16, (1, 4))
        result = generate(
            model,
            prompt,
            max_new_tokens=5,
            context_size=32,
            temperature=1.0,
        )
        assert result.shape == (1, 9)

    def test_greedy_deterministic(self) -> None:
        """Greedy decoding (temperature=0) is deterministic."""
        model = _TinyModel(vocab_size=16)
        model.eval()
        prompt = torch.randint(0, 16, (1, 4))
        r1 = generate(
            model,
            prompt.clone(),
            max_new_tokens=5,
            context_size=32,
            temperature=0.0,
        )
        r2 = generate(
            model,
            prompt.clone(),
            max_new_tokens=5,
            context_size=32,
            temperature=0.0,
        )
        assert torch.equal(r1, r2)

    def test_constant_model_greedy(self) -> None:
        """Greedy decoding with a constant model always picks that token."""
        model = _ConstantModel(vocab_size=16, predict_token=7)
        prompt = torch.randint(0, 16, (1, 3))
        result = generate(
            model,
            prompt,
            max_new_tokens=5,
            context_size=32,
            temperature=0.0,
        )
        generated = result[0, 3:]  # skip prompt
        assert torch.all(generated == 7)

    def test_top_k_filters_tokens(self) -> None:
        """Top-k limits the sampling pool."""
        model = _TinyModel(vocab_size=16)
        model.eval()
        prompt = torch.randint(0, 16, (1, 4))
        # Should not crash and should produce valid tokens
        result = generate(
            model,
            prompt,
            max_new_tokens=5,
            context_size=32,
            temperature=1.0,
            top_k=3,
        )
        assert result.shape == (1, 9)
        # All generated tokens should be valid vocab indices
        assert torch.all(result >= 0)
        assert torch.all(result < 16)

    def test_context_cropping(self) -> None:
        """Long sequences are cropped to context_size."""
        model = _TinyModel(vocab_size=16, context_length=8)
        model.eval()
        # Prompt longer than context_size
        prompt = torch.randint(0, 16, (1, 20))
        result = generate(
            model,
            prompt,
            max_new_tokens=3,
            context_size=8,
            temperature=0.0,
        )
        # Should still work — prompt preserved plus new tokens
        assert result.shape == (1, 23)

    def test_eos_stops_generation(self) -> None:
        """Generation stops when EOS token is produced."""
        eos_id = 5
        model = _ConstantModel(vocab_size=16, predict_token=eos_id)
        prompt = torch.randint(0, 16, (1, 3))
        result = generate(
            model,
            prompt,
            max_new_tokens=100,
            context_size=32,
            temperature=0.0,
            eos_id=eos_id,
        )
        # Should stop after first generated token (which is EOS)
        assert result.shape[1] == 3  # just the prompt, EOS not appended

    def test_prompt_preserved(self) -> None:
        """The original prompt is preserved in the output."""
        model = _TinyModel(vocab_size=16)
        model.eval()
        prompt = torch.tensor([[1, 2, 3, 4]])
        result = generate(
            model,
            prompt,
            max_new_tokens=5,
            context_size=32,
            temperature=0.0,
        )
        assert torch.equal(result[0, :4], prompt[0])

    def test_batch_size_one(self) -> None:
        """Works with batch_size=1."""
        model = _TinyModel(vocab_size=16)
        model.eval()
        prompt = torch.randint(0, 16, (1, 2))
        result = generate(
            model,
            prompt,
            max_new_tokens=3,
            context_size=32,
            temperature=0.0,
        )
        assert result.shape[0] == 1

    def test_zero_new_tokens(self) -> None:
        """Generating zero tokens returns the prompt unchanged."""
        model = _TinyModel(vocab_size=16)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        result = generate(
            model,
            prompt,
            max_new_tokens=0,
            context_size=32,
            temperature=0.0,
        )
        assert torch.equal(result, prompt)

    def test_all_tokens_valid(self) -> None:
        """All generated tokens are valid vocabulary indices."""
        model = _TinyModel(vocab_size=16)
        model.eval()
        prompt = torch.randint(0, 16, (1, 4))
        result = generate(
            model,
            prompt,
            max_new_tokens=20,
            context_size=32,
            temperature=1.0,
        )
        assert torch.all(result >= 0)
        assert torch.all(result < 16)
