"""Tests for architecture comparison experiment."""

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.models.config import ModelConfig
from lmt.models.factory import create_model
from lmt.training.train_config import TrainConfig
from lmt.training.train_loop import TrainState, train_loop


def _tiny_config() -> ModelConfig:
    """Minimal config that works for all architectures."""
    return ModelConfig(
        embed_dim=32,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        ffn_hidden_dim=64,
        vocab_size=64,
        context_length=16,
        tie_weights=True,
        dropout=0.0,
    )


def _loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Standard next-token prediction loss."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    return f.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


class TestArchComparison:
    """Test that architecture comparison pipeline works."""

    def test_all_architectures_train(self):
        """All main architectures should train and reduce loss."""
        torch.manual_seed(42)
        data = torch.randint(0, 64, (32, 17))
        config = TrainConfig(total_steps=5, lr=1e-3)

        for arch in ['gpt', 'llama', 'qwen3', 'gemma']:
            model = create_model(arch, _tiny_config())
            state = train_loop(
                model=model,
                train_data=data,
                loss_fn=_loss_fn,
                config=config,
                batch_size=8,
            )
            assert isinstance(state, TrainState)
            assert state.step == 5
            assert len(state.losses) == 5

    def test_models_have_similar_param_counts(self):
        """Same config should produce similar param counts."""
        cfg = _tiny_config()
        counts = {}
        for arch in ['gpt', 'llama', 'qwen3', 'gemma']:
            model = create_model(arch, cfg)
            counts[arch] = sum(p.numel() for p in model.parameters())

        # All should be within 2x of each other
        min_count = min(counts.values())
        max_count = max(counts.values())
        assert max_count < 2 * min_count, (
            f'Param counts too different: {counts}'
        )

    def test_mixtral_trains(self):
        """Mixtral (MoE) should also train."""
        torch.manual_seed(42)
        data = torch.randint(0, 64, (32, 17))
        config = TrainConfig(total_steps=3, lr=1e-3)

        model = create_model('mixtral', _tiny_config())
        state = train_loop(
            model=model,
            train_data=data,
            loss_fn=_loss_fn,
            config=config,
            batch_size=8,
        )
        assert state.step == 3
