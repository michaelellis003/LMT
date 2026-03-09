"""Tests for unified training configuration."""

import torch
import torch.nn as nn

from lmt.training.train_config import TrainConfig, configure_training


class TestTrainConfig:
    """Test TrainConfig dataclass."""

    def test_defaults(self):
        """Should have sensible defaults."""
        cfg = TrainConfig()
        assert cfg.lr == 3e-4
        assert cfg.weight_decay == 0.1
        assert cfg.max_grad_norm == 1.0
        assert cfg.warmup_steps == 0
        assert cfg.total_steps == 1000

    def test_to_dict(self):
        """Should serialize to dict."""
        cfg = TrainConfig(lr=1e-3, total_steps=500)
        d = cfg.to_dict()
        assert d['lr'] == 1e-3
        assert d['total_steps'] == 500

    def test_from_dict(self):
        """Should deserialize from dict."""
        d = {'lr': 1e-3, 'weight_decay': 0.05, 'unknown_key': True}
        cfg = TrainConfig.from_dict(d)
        assert cfg.lr == 1e-3
        assert cfg.weight_decay == 0.05


class TestConfigureTraining:
    """Test configure_training helper."""

    def test_returns_optimizer_and_scheduler(self):
        """Should return optimizer and optional scheduler."""
        model = nn.Linear(10, 5)
        cfg = TrainConfig(lr=1e-3, total_steps=100, warmup_steps=10)
        optimizer, scheduler = configure_training(model, cfg)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert scheduler is not None

    def test_no_scheduler_when_no_warmup(self):
        """Should skip scheduler if warmup_steps=0."""
        model = nn.Linear(10, 5)
        cfg = TrainConfig(warmup_steps=0)
        optimizer, scheduler = configure_training(model, cfg)
        assert scheduler is None

    def test_uses_param_groups(self):
        """Should separate decay/no-decay parameters."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.LayerNorm(20),
            nn.Linear(20, 5),
        )
        cfg = TrainConfig(weight_decay=0.1)
        optimizer, _ = configure_training(model, cfg)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['weight_decay'] == 0.1
        assert optimizer.param_groups[1]['weight_decay'] == 0.0

    def test_optimizer_lr(self):
        """Should set correct learning rate."""
        model = nn.Linear(10, 5)
        cfg = TrainConfig(lr=5e-4)
        optimizer, _ = configure_training(model, cfg)
        assert optimizer.param_groups[0]['lr'] == 5e-4

    def test_training_step(self):
        """Should work in a basic training step."""
        model = nn.Linear(10, 5)
        cfg = TrainConfig(lr=1e-3, total_steps=10, warmup_steps=2)
        optimizer, scheduler = configure_training(model, cfg)

        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def test_zero_weight_decay(self):
        """Should handle zero weight decay."""
        model = nn.Linear(10, 5)
        cfg = TrainConfig(weight_decay=0.0)
        optimizer, _ = configure_training(model, cfg)
        # All param groups should have weight_decay=0
        for group in optimizer.param_groups:
            assert group['weight_decay'] == 0.0
