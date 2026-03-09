"""Tests for composable train_loop function."""

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.training.train_config import TrainConfig
from lmt.training.train_loop import TrainState, train_loop


class TestTrainLoop:
    """Test train_loop function."""

    def _make_data(
        self, num_samples: int = 32, seq_len: int = 8, vocab_size: int = 50
    ) -> torch.Tensor:
        """Create random token data."""
        return torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def _make_model(self, vocab_size: int = 50, dim: int = 16) -> nn.Module:
        """Create a tiny model for testing."""
        return nn.Sequential(
            nn.Embedding(vocab_size, dim),
            nn.Linear(dim, vocab_size),
        )

    def _loss_fn(
        self,
        model: nn.Module,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Standard next-token prediction loss."""
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        return f.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

    def test_basic_training(self):
        """Should run training and return state."""
        model = self._make_model()
        data = self._make_data()
        config = TrainConfig(total_steps=5, lr=1e-3)

        state = train_loop(
            model=model,
            train_data=data,
            loss_fn=self._loss_fn,
            config=config,
            batch_size=4,
        )

        assert isinstance(state, TrainState)
        assert state.step == 5
        assert len(state.losses) == 5

    def test_loss_decreases(self):
        """Should show decreasing loss over training."""
        torch.manual_seed(42)
        model = self._make_model()
        data = self._make_data(num_samples=64)
        config = TrainConfig(total_steps=20, lr=1e-2)

        state = train_loop(
            model=model,
            train_data=data,
            loss_fn=self._loss_fn,
            config=config,
            batch_size=8,
        )

        # Average of first 5 losses should be higher than last 5
        early = sum(state.losses[:5]) / 5
        late = sum(state.losses[-5:]) / 5
        assert late < early

    def test_with_eval_fn(self):
        """Should call eval function at intervals."""
        model = self._make_model()
        data = self._make_data()
        config = TrainConfig(total_steps=10, lr=1e-3)

        eval_results: list[float] = []

        def eval_fn(m: nn.Module, step: int) -> dict[str, float]:
            eval_results.append(step)
            return {'val_loss': 1.0}

        state = train_loop(
            model=model,
            train_data=data,
            loss_fn=self._loss_fn,
            config=config,
            batch_size=4,
            eval_fn=eval_fn,
            eval_interval=5,
        )

        # Should have been called at steps 5 and 10
        assert len(eval_results) == 2
        assert 'val_loss' in state.eval_metrics

    def test_with_grad_accumulation(self):
        """Should support gradient accumulation."""
        model = self._make_model()
        data = self._make_data()
        config = TrainConfig(total_steps=4, lr=1e-3)

        state = train_loop(
            model=model,
            train_data=data,
            loss_fn=self._loss_fn,
            config=config,
            batch_size=4,
            accumulation_steps=2,
        )

        assert state.step == 4

    def test_grad_clipping(self):
        """Should clip gradients when configured."""
        model = self._make_model()
        data = self._make_data()
        config = TrainConfig(total_steps=3, lr=1e-3, max_grad_norm=0.5)

        state = train_loop(
            model=model,
            train_data=data,
            loss_fn=self._loss_fn,
            config=config,
            batch_size=4,
        )

        assert state.step == 3

    def test_warmup_schedule(self):
        """Should apply warmup schedule."""
        model = self._make_model()
        data = self._make_data()
        config = TrainConfig(total_steps=10, lr=1e-3, warmup_steps=5)

        state = train_loop(
            model=model,
            train_data=data,
            loss_fn=self._loss_fn,
            config=config,
            batch_size=4,
        )

        assert state.step == 10

    def test_returns_train_state(self):
        """Should return TrainState with all history."""
        model = self._make_model()
        data = self._make_data()
        config = TrainConfig(total_steps=3, lr=1e-3)

        state = train_loop(
            model=model,
            train_data=data,
            loss_fn=self._loss_fn,
            config=config,
            batch_size=4,
        )

        assert hasattr(state, 'step')
        assert hasattr(state, 'losses')
        assert hasattr(state, 'eval_metrics')
        assert hasattr(state, 'lrs')

    def test_lr_tracking(self):
        """Should track learning rates."""
        model = self._make_model()
        data = self._make_data()
        config = TrainConfig(total_steps=5, lr=1e-3, warmup_steps=2)

        state = train_loop(
            model=model,
            train_data=data,
            loss_fn=self._loss_fn,
            config=config,
            batch_size=4,
        )

        assert len(state.lrs) == 5
        # With warmup, first LR should be lower
        assert state.lrs[0] < state.lrs[2]


class TestTrainState:
    """Test TrainState dataclass."""

    def test_creation(self):
        """Should create a train state."""
        state = TrainState(
            step=10,
            losses=[2.0, 1.5],
            lrs=[1e-3, 1e-3],
            eval_metrics={},
        )
        assert state.step == 10

    def test_final_loss(self):
        """Should return last loss."""
        state = TrainState(
            step=3,
            losses=[3.0, 2.0, 1.0],
            lrs=[1e-3] * 3,
            eval_metrics={},
        )
        assert state.final_loss == 1.0

    def test_final_loss_empty(self):
        """Should return None when no losses."""
        state = TrainState(step=0, losses=[], lrs=[], eval_metrics={})
        assert state.final_loss is None
