"""Tests for LR scheduler integration with Trainer."""

from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from lmt.models.config import ModelConfigPresets
from lmt.models.gpt import GPT
from lmt.training.config import BaseTrainingConfig
from lmt.training.trainer import Trainer


class TestSchedulerIntegration:
    """Test PyTorch LR scheduler integration with Trainer."""

    def _make_trainer(self, scheduler_fn=None):
        """Helper to create a trainer with optional scheduler."""
        config = ModelConfigPresets.small_gpt(context_length=8)
        model = GPT(config)
        data = torch.randint(0, config.vocab_size, (4, 8))
        ds = TensorDataset(data, data)
        loader = DataLoader(ds, batch_size=2)

        training_config = BaseTrainingConfig(
            num_epochs=1,
            eval_freq=100,
            eval_iter=1,
            learning_rate=1e-3,
            weight_decay=0.01,
            device='cpu',
        )

        trainer = Trainer(model, loader, loader, training_config)

        scheduler = None
        if scheduler_fn is not None:
            scheduler = scheduler_fn(trainer.optimizer)
            trainer.scheduler = scheduler

        return trainer, scheduler

    def test_trainer_accepts_scheduler(self):
        """Trainer should accept a PyTorch LR scheduler."""
        trainer, scheduler = self._make_trainer(
            lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        )
        assert trainer.scheduler is scheduler

    def test_trainer_without_scheduler(self):
        """Trainer should work without a scheduler."""
        trainer, _ = self._make_trainer()
        assert trainer.scheduler is None
        with patch('builtins.print'):
            trainer.train()
        assert trainer.global_step > -1

    def test_scheduler_steps_during_training(self):
        """Scheduler.step() should be called during training."""
        trainer, scheduler = self._make_trainer(
            lambda opt: torch.optim.lr_scheduler.StepLR(
                opt, step_size=1, gamma=0.5
            )
        )

        initial_lr = trainer.optimizer.param_groups[0]['lr']

        with patch('builtins.print'):
            trainer.train()

        # After training with StepLR(gamma=0.5), LR should have decreased
        final_lr = trainer.optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr, (
            f'LR should decrease: {initial_lr} -> {final_lr}'
        )

    def test_cosine_annealing_scheduler(self):
        """CosineAnnealingLR should integrate correctly."""
        trainer, _ = self._make_trainer(
            lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=10, eta_min=1e-5
            )
        )

        with patch('builtins.print'):
            trainer.train()

        final_lr = trainer.optimizer.param_groups[0]['lr']
        assert final_lr < 1e-3  # Should have decayed from initial 1e-3
