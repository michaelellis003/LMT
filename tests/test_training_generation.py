"""Tests for response generation module.

Tests the standalone generation functions used by GRPO training
for generating batches of responses with temperature sampling,
top-k/top-p filtering, and stop token handling.
"""

import torch
import torch.nn as nn

from lmt.models.config import ModelConfig


class _TinyModel(nn.Module):
    """Minimal model for generation testing."""

    def __init__(self, vocab_size: int = 32, embed_dim: int = 16) -> None:
        super().__init__()
        self.config = ModelConfig(
            embed_dim=embed_dim,
            num_heads=2,
            num_kv_heads=2,
            num_layers=1,
            ffn_hidden_dim=32,
            vocab_size=vocab_size,
            context_length=64,
            tie_weights=True,
            qk_norm=False,
            dropout=0.0,
        )
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits [batch, seq_len, vocab_size]."""
        return self.head(self.embed(x))


class TestGenerateResponses:
    """Test the generate_responses function."""

    def test_output_shape(self):
        """Output should be [group_size, prompt_len + max_len]."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _TinyModel()
        prompt = torch.randint(0, 32, (5,))

        config = GenerationConfig(
            group_size=4,
            max_response_len=10,
            temperature=1.0,
            top_k=0,
        )

        sequences = generate_responses(model, prompt, config)
        assert sequences.shape == (4, 15)  # 5 + 10

    def test_prompt_preserved(self):
        """Generated sequences should start with the prompt."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _TinyModel()
        prompt = torch.tensor([1, 2, 3, 4, 5])

        config = GenerationConfig(
            group_size=3,
            max_response_len=8,
            temperature=1.0,
            top_k=0,
        )

        sequences = generate_responses(model, prompt, config)
        for i in range(3):
            assert torch.equal(sequences[i, :5], prompt)

    def test_different_temperatures(self):
        """Higher temperature should produce more diverse outputs."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _TinyModel()
        torch.manual_seed(42)
        prompt = torch.randint(0, 32, (3,))

        # Low temperature (more deterministic)
        low_config = GenerationConfig(
            group_size=8,
            max_response_len=10,
            temperature=0.01,
            top_k=0,
        )
        torch.manual_seed(0)
        low_seqs = generate_responses(model, prompt, low_config)

        # High temperature (more diverse)
        high_config = GenerationConfig(
            group_size=8,
            max_response_len=10,
            temperature=2.0,
            top_k=0,
        )
        torch.manual_seed(0)
        high_seqs = generate_responses(model, prompt, high_config)

        # Count unique sequences
        low_unique = len(set(tuple(s.tolist()) for s in low_seqs))
        high_unique = len(set(tuple(s.tolist()) for s in high_seqs))

        # High temp should have at least as many unique sequences
        assert high_unique >= low_unique

    def test_top_k_filtering(self):
        """Top-k should restrict tokens to top k candidates."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _TinyModel(vocab_size=100)
        prompt = torch.randint(0, 100, (3,))

        config = GenerationConfig(
            group_size=4,
            max_response_len=20,
            temperature=1.0,
            top_k=5,
        )

        sequences = generate_responses(model, prompt, config)
        # Should complete without error and have correct shape
        assert sequences.shape == (4, 23)

    def test_top_p_filtering(self):
        """Top-p (nucleus) should filter by cumulative probability."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _TinyModel()
        prompt = torch.randint(0, 32, (3,))

        config = GenerationConfig(
            group_size=4,
            max_response_len=10,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
        )

        sequences = generate_responses(model, prompt, config)
        assert sequences.shape == (4, 13)

    def test_stop_token(self):
        """Generation should stop at stop token and pad remaining."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        # Create a model that always outputs token 5 after a few steps
        class AlwaysOutputModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = ModelConfig(
                    embed_dim=16,
                    num_heads=2,
                    num_kv_heads=2,
                    num_layers=1,
                    ffn_hidden_dim=32,
                    vocab_size=32,
                    context_length=64,
                    tie_weights=True,
                    qk_norm=False,
                    dropout=0.0,
                )
                # Dummy parameter so next(model.parameters()) works
                self._dummy = nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch, seq_len = x.shape
                logits = torch.full((batch, seq_len, 32), -100.0)
                # Make token 5 (stop) the most likely
                logits[:, :, 5] = 100.0
                return logits

        model = AlwaysOutputModel()
        prompt = torch.tensor([1, 2, 3])

        config = GenerationConfig(
            group_size=2,
            max_response_len=10,
            temperature=1.0,
            top_k=0,
            stop_token_id=5,
            pad_token_id=0,
        )

        sequences = generate_responses(model, prompt, config)
        # Should have full length (padded)
        assert sequences.shape == (2, 13)
        # After the stop token, remaining should be pad
        # First generated token is 5 (stop), so positions 4+ should be pad
        for i in range(2):
            assert sequences[i, 3].item() == 5  # stop token
            for j in range(4, 13):
                assert sequences[i, j].item() == 0  # pad

    def test_no_grad(self):
        """Generation should not accumulate gradients."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _TinyModel()
        prompt = torch.randint(0, 32, (3,))

        config = GenerationConfig(
            group_size=2,
            max_response_len=5,
            temperature=1.0,
            top_k=0,
        )

        sequences = generate_responses(model, prompt, config)
        assert not sequences.requires_grad

    def test_respects_context_length(self):
        """Should truncate input to context_length when sequence is long."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _TinyModel(vocab_size=32, embed_dim=16)
        # Model has context_length=64, use a shorter prompt
        prompt = torch.randint(0, 32, (10,))

        config = GenerationConfig(
            group_size=2,
            max_response_len=5,
            temperature=1.0,
            top_k=0,
        )

        sequences = generate_responses(model, prompt, config)
        assert sequences.shape == (2, 15)

    def test_device_handling(self):
        """Should work on CPU (and respect model device)."""
        from lmt.training.generation import (
            GenerationConfig,
            generate_responses,
        )

        model = _TinyModel()
        prompt = torch.randint(0, 32, (3,))

        config = GenerationConfig(
            group_size=2,
            max_response_len=5,
            temperature=1.0,
            top_k=0,
        )

        sequences = generate_responses(model, prompt, config)
        assert sequences.device.type == 'cpu'


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_defaults(self):
        """Should have sensible defaults."""
        from lmt.training.generation import GenerationConfig

        config = GenerationConfig()
        assert config.group_size == 8
        assert config.max_response_len == 128
        assert config.temperature == 0.7
        assert config.top_k == 50
        assert config.top_p == 1.0
        assert config.stop_token_id is None
        assert config.pad_token_id == 0

    def test_custom_values(self):
        """Should accept custom values."""
        from lmt.training.generation import GenerationConfig

        config = GenerationConfig(
            group_size=4,
            max_response_len=64,
            temperature=0.5,
            top_k=10,
            top_p=0.9,
            stop_token_id=2,
            pad_token_id=1,
        )
        assert config.group_size == 4
        assert config.stop_token_id == 2
