"""Tests for perplexity evaluation."""

import torch

from lmt.models.config import ModelConfigPresets
from lmt.models.gpt import GPT
from lmt.training.eval import compute_perplexity


class TestComputePerplexity:
    """Test perplexity computation on token sequences."""

    def _make_model_and_data(self, seq_len=16, num_seqs=4):
        """Create a small model and token data for testing."""
        config = ModelConfigPresets.small_gpt(context_length=seq_len)
        model = GPT(config)
        model.eval()
        # Random token sequences
        tokens = torch.randint(0, config.vocab_size, (num_seqs, seq_len + 1))
        return model, tokens, config

    def test_returns_finite_float(self):
        """Perplexity should be a finite positive float."""
        model, tokens, _ = self._make_model_and_data()
        ppl = compute_perplexity(model, tokens, device='cpu')
        assert isinstance(ppl, float)
        assert ppl > 0
        assert not torch.isinf(torch.tensor(ppl))
        assert not torch.isnan(torch.tensor(ppl))

    def test_perplexity_is_exp_of_loss(self):
        """Perplexity should equal exp(average cross-entropy loss)."""
        model, tokens, config = self._make_model_and_data()
        ppl = compute_perplexity(model, tokens, device='cpu')

        # Manually compute: average cross-entropy then exp
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for seq in tokens:
                inp = seq[:-1].unsqueeze(0)
                tgt = seq[1:]
                logits = model(inp).squeeze(0)
                loss = torch.nn.functional.cross_entropy(
                    logits, tgt, reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += tgt.numel()
        expected_ppl = torch.exp(
            torch.tensor(total_loss / total_tokens)
        ).item()

        assert abs(ppl - expected_ppl) < 1e-3, (
            f'Perplexity mismatch: {ppl} vs {expected_ppl}'
        )

    def test_random_model_high_perplexity(self):
        """An untrained model should have perplexity near vocab_size."""
        model, tokens, config = self._make_model_and_data(num_seqs=20)
        ppl = compute_perplexity(model, tokens, device='cpu')
        # Random model: PPL should be roughly vocab_size (50257)
        # Allow wide range since it's random
        assert ppl > 1000, f'Untrained model PPL too low: {ppl}'
        assert ppl < 200000, f'Untrained model PPL too high: {ppl}'

    def test_stride_parameter(self):
        """Stride controls sliding window overlap."""
        model, tokens, config = self._make_model_and_data(
            seq_len=32, num_seqs=2
        )
        # Both should produce finite results
        ppl_full = compute_perplexity(model, tokens, device='cpu', stride=32)
        ppl_overlap = compute_perplexity(
            model, tokens, device='cpu', stride=16
        )
        assert ppl_full > 0
        assert ppl_overlap > 0
        # With overlap, more context is available so PPL might differ
        # but both should be valid
        assert isinstance(ppl_full, float)
        assert isinstance(ppl_overlap, float)

    def test_single_sequence(self):
        """Should work with a single sequence."""
        model, _, config = self._make_model_and_data()
        tokens = torch.randint(0, config.vocab_size, (1, 17))
        ppl = compute_perplexity(model, tokens, device='cpu')
        assert ppl > 0
