"""Tests for weight tying (input/output embedding sharing)."""

import torch

from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig
from lmt.models.gpt import GPT


class TestWeightTying:
    """Test weight tying between token embedding and output head."""

    def _make_config(self, tie_weights: bool = True) -> ModelConfig:
        """Create a small config with optional weight tying."""
        return ModelConfig(
            context_length=32,
            vocab_size=256,
            num_layers=2,
            num_heads=4,
            embed_dim=64,
            dropout=0.0,
            tie_weights=tie_weights,
        )

    def test_tied_weights_share_same_tensor(self):
        """When tie_weights=True, tok_embed and out_head share weight."""
        config = self._make_config(tie_weights=True)
        model = GPT(config)
        assert model.tok_embed.weight is model.out_head.weight

    def test_untied_weights_are_different(self):
        """When tie_weights=False, tok_embed and out_head are separate."""
        config = self._make_config(tie_weights=False)
        model = GPT(config)
        assert model.tok_embed.weight is not model.out_head.weight

    def test_tied_reduces_param_count(self):
        """Tying should reduce parameter count by vocab_size * embed_dim."""
        config_tied = self._make_config(tie_weights=True)
        config_untied = self._make_config(tie_weights=False)
        model_tied = GPT(config_tied)
        model_untied = GPT(config_untied)

        tied_params = sum(p.numel() for p in model_tied.parameters())
        untied_params = sum(p.numel() for p in model_untied.parameters())

        expected_saving = config_tied.vocab_size * config_tied.embed_dim
        assert untied_params - tied_params == expected_saving

    def test_tied_forward_pass_works(self):
        """Model with tied weights should produce valid logits."""
        config = self._make_config(tie_weights=True)
        model = GPT(config)
        model.eval()
        x = torch.randint(0, config.vocab_size, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_tied_gradient_flows(self):
        """Gradients should flow through tied weights."""
        config = self._make_config(tie_weights=True)
        model = GPT(config)
        x = torch.randint(0, config.vocab_size, (1, 8))
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        assert model.tok_embed.weight.grad is not None
        # Since weights are tied, the grad should include
        # contributions from both embedding and output head
        assert model.tok_embed.weight.grad.abs().sum() > 0

    def test_default_is_untied(self):
        """Default ModelConfig should have tie_weights=False."""
        config = ModelConfig()
        assert config.tie_weights is False

    def test_base_model_supports_tying(self):
        """BaseModel with default BlockConfig should support tying."""
        config = self._make_config(tie_weights=True)
        model = BaseModel(config)
        assert model.tok_embed.weight is model.out_head.weight
