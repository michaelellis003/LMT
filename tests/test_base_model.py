"""Tests for BaseModel -- shared language model scaffold."""

import torch
import torch.nn as nn

from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.models.config import ModelConfig


class TestBaseModel:
    """Test the BaseModel shared scaffold."""

    def _make_config(self, **kwargs):
        """Create a ModelConfig with sensible defaults."""
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

    def _make_model(self, **kwargs):
        """Create a BaseModel for testing."""
        from lmt.models.base import BaseModel

        config = self._make_config(**kwargs)
        return BaseModel(config)

    def test_output_shape(self) -> None:
        """Model outputs [B, L, vocab_size] logits."""
        model = self._make_model()
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_single_token(self) -> None:
        """Handles single-token input."""
        model = self._make_model()
        x = torch.randint(0, 256, (1, 1))
        logits = model(x)
        assert logits.shape == (1, 1, 256)

    def test_gradient_flow(self) -> None:
        """Gradients reach the embedding layer."""
        model = self._make_model()
        x = torch.randint(0, 256, (2, 8))
        logits = model(x)
        logits.sum().backward()
        assert model.tok_embed.weight.grad is not None
        assert not torch.all(model.tok_embed.weight.grad == 0)

    def test_weight_init(self) -> None:
        """Weights are initialized (not all zeros)."""
        model = self._make_model(num_layers=4)
        for name, param in model.named_parameters():
            if param.requires_grad and 'bias' not in name:
                assert not torch.all(param == 0), f'{name} is all zeros'

    def test_compatible_with_cross_entropy(self) -> None:
        """Output works with cross-entropy loss."""
        model = self._make_model()
        x = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, 256), targets.view(-1)
        )
        loss.backward()
        assert torch.isfinite(loss)

    def test_custom_block_config(self) -> None:
        """BaseModel accepts a custom BlockConfig."""
        from lmt.models.base import BaseModel

        config = self._make_config()
        block_config = BlockConfig(
            attention='mha', ffn='swiglu', norm='rmsnorm'
        )
        model = BaseModel(config, block_config=block_config)
        x = torch.randint(0, 256, (1, 8))
        logits = model(x)
        assert logits.shape == (1, 8, 256)

    def test_moe_aux_loss(self) -> None:
        """BaseModel collects aux_loss from MoE blocks."""
        from lmt.models.base import BaseModel

        config = self._make_config()
        block_config = BlockConfig(
            attention='gqa',
            ffn='moe',
            norm='rmsnorm',
            moe_num_experts=4,
            moe_top_k=2,
        )
        model = BaseModel(config, block_config=block_config)
        x = torch.randint(0, 256, (1, 8))
        logits = model(x)
        assert logits.shape == (1, 8, 256)
        # MoE models should have non-zero aux_loss
        assert hasattr(model, 'aux_loss')
        assert model.aux_loss.item() >= 0

    def test_moe_expert_w2_scaling(self) -> None:
        """MoE expert w2 weights should use scaled residual init."""
        import math

        from lmt.models.base import BaseModel

        config = self._make_config(num_layers=4)
        block_config = BlockConfig(
            attention='gqa',
            ffn='moe',
            norm='rmsnorm',
            moe_num_experts=4,
            moe_top_k=2,
        )
        model = BaseModel(config, block_config=block_config)

        expected_std = 0.02 / math.sqrt(2 * config.num_layers)
        for block in model.blocks:
            for expert in block.ffn.experts:  # type: ignore[union-attr]
                actual_std = expert.w2.weight.std().item()
                # Allow some variance since these are random samples
                assert actual_std < 0.02, (
                    f'Expert w2 std {actual_std:.4f} not scaled '
                    f'(expected ~{expected_std:.4f})'
                )

    def test_param_count_scales_with_layers(self) -> None:
        """More layers means more parameters."""
        model_2 = self._make_model(num_layers=2)
        model_4 = self._make_model(num_layers=4)
        p2 = sum(p.numel() for p in model_2.parameters())
        p4 = sum(p.numel() for p in model_4.parameters())
        assert p4 > p2


class TestNamedModels:
    """Test that named models (LLaMA, Mixtral, etc.) still work.

    These test that the refactored models produce valid outputs.
    They intentionally don't test exact numerical equivalence with
    the old implementations (weights are re-initialized).
    """

    def _make_config(self, **kwargs):
        """Create a ModelConfig with sensible defaults."""
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

    def test_llama_output_shape(self) -> None:
        """LLaMA produces correct output shape."""
        from lmt.models.llama import LLaMA

        config = self._make_config()
        model = LLaMA(config)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_mixtral_output_shape(self) -> None:
        """Mixtral produces correct output shape."""
        from lmt.models.mixtral import Mixtral

        config = self._make_config()
        model = Mixtral(config, num_experts=4, top_k=2)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_mixtral_has_aux_loss(self) -> None:
        """Mixtral collects MoE auxiliary loss."""
        from lmt.models.mixtral import Mixtral

        config = self._make_config()
        model = Mixtral(config, num_experts=4, top_k=2)
        x = torch.randint(0, 256, (1, 8))
        model(x)
        assert hasattr(model, 'aux_loss')
        assert model.aux_loss.item() >= 0
