"""Tests for gradient checkpointing in BaseModel."""

import torch

from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig


class TestGradientCheckpointing:
    """Test gradient checkpointing trades compute for memory."""

    def _make_config(self, **kwargs):
        """Create a small test config."""
        defaults = dict(
            embed_dim=64,
            num_heads=4,
            num_kv_heads=4,
            context_length=32,
            vocab_size=256,
            num_layers=2,
            dropout=0.0,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)

    def test_default_off(self):
        """Gradient checkpointing is off by default."""
        model = BaseModel(self._make_config())
        assert model.gradient_checkpointing is False

    def test_can_enable(self):
        """Can enable gradient checkpointing."""
        model = BaseModel(self._make_config())
        model.gradient_checkpointing = True
        assert model.gradient_checkpointing is True

    def test_forward_works_with_checkpointing(self):
        """Forward pass produces correct shape with checkpointing."""
        model = BaseModel(self._make_config())
        model.gradient_checkpointing = True
        model.train()
        x = torch.randint(0, 256, (2, 8))
        logits = model(x)
        assert logits.shape == (2, 8, 256)

    def test_backward_works_with_checkpointing(self):
        """Backward pass works with checkpointing enabled."""
        model = BaseModel(self._make_config())
        model.gradient_checkpointing = True
        model.train()
        x = torch.randint(0, 256, (2, 8))
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        assert model.tok_embed.weight.grad is not None

    def test_no_checkpointing_in_eval_mode(self):
        """Checkpointing only activates during training, not eval."""
        model = BaseModel(self._make_config())
        model.gradient_checkpointing = True
        model.eval()
        x = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 8, 256)

    def test_output_matches_without_checkpointing(self):
        """Output is identical with and without checkpointing."""
        config = self._make_config()
        model = BaseModel(config)
        model.eval()
        x = torch.randint(0, 256, (1, 8))

        with torch.no_grad():
            out_normal = model(x)

        # Same model, enable checkpointing (won't activate in eval)
        model.gradient_checkpointing = True
        model.train()
        out_ckpt = model(x)

        # Outputs should be the same since it's the same weights
        # (checkpointing doesn't change numerics, just memory)
        # Note: in train mode dropout=0.0 so outputs match
        assert torch.allclose(out_normal, out_ckpt, atol=1e-5)
