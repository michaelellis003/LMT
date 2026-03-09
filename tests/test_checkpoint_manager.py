"""Tests for checkpoint manager."""

import json
from pathlib import Path

import torch
import torch.nn as nn

from lmt.training.checkpoints import CheckpointManager


class _TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, dim: int = 16):
        """Initialize tiny model."""
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x)


class TestCheckpointManager:
    """Test checkpoint save/load with metadata."""

    def test_save_and_load(self, tmp_path: Path):
        """Should save and load model state dict."""
        manager = CheckpointManager(tmp_path)
        model = _TinyModel()

        manager.save(model, step=10, metrics={'loss': 1.5})

        loaded_model = _TinyModel()
        meta = manager.load(loaded_model, step=10)

        # Weights should match
        for key in model.state_dict():
            assert torch.allclose(
                model.state_dict()[key],
                loaded_model.state_dict()[key],
            )
        assert meta['step'] == 10
        assert meta['metrics']['loss'] == 1.5

    def test_save_creates_directory(self, tmp_path: Path):
        """Should create checkpoint directory if needed."""
        ckpt_dir = tmp_path / 'nested' / 'ckpts'
        manager = CheckpointManager(ckpt_dir)
        model = _TinyModel()

        manager.save(model, step=0)
        assert ckpt_dir.exists()

    def test_load_latest(self, tmp_path: Path):
        """Should load the most recent checkpoint."""
        manager = CheckpointManager(tmp_path)
        model = _TinyModel()

        manager.save(model, step=5)
        model.linear.weight.data.fill_(99.0)
        manager.save(model, step=10)

        fresh = _TinyModel()
        meta = manager.load_latest(fresh)

        assert meta is not None
        assert meta['step'] == 10
        assert torch.allclose(
            fresh.state_dict()['linear.weight'],
            torch.full_like(fresh.linear.weight, 99.0),
        )

    def test_load_latest_empty_dir(self, tmp_path: Path):
        """Should return None when no checkpoints exist."""
        manager = CheckpointManager(tmp_path)
        model = _TinyModel()
        meta = manager.load_latest(model)
        assert meta is None

    def test_save_best(self, tmp_path: Path):
        """Should track and save best checkpoint by metric."""
        manager = CheckpointManager(tmp_path)
        model = _TinyModel()

        # First save — becomes best by default
        saved = manager.save_if_best(
            model, step=1, metric_value=2.0, metric_name='loss'
        )
        assert saved is True

        # Worse metric — should not save
        saved = manager.save_if_best(
            model, step=2, metric_value=3.0, metric_name='loss'
        )
        assert saved is False

        # Better metric — should save
        saved = manager.save_if_best(
            model, step=3, metric_value=1.0, metric_name='loss'
        )
        assert saved is True

    def test_save_best_higher_is_better(self, tmp_path: Path):
        """Should support metrics where higher is better."""
        manager = CheckpointManager(tmp_path)
        model = _TinyModel()

        saved = manager.save_if_best(
            model,
            step=1,
            metric_value=0.5,
            metric_name='accuracy',
            higher_is_better=True,
        )
        assert saved is True

        saved = manager.save_if_best(
            model,
            step=2,
            metric_value=0.8,
            metric_name='accuracy',
            higher_is_better=True,
        )
        assert saved is True

    def test_list_checkpoints(self, tmp_path: Path):
        """Should list all saved checkpoints."""
        manager = CheckpointManager(tmp_path)
        model = _TinyModel()

        manager.save(model, step=5)
        manager.save(model, step=10)
        manager.save(model, step=15)

        ckpts = manager.list_checkpoints()
        assert len(ckpts) == 3
        steps = [c['step'] for c in ckpts]
        assert steps == [5, 10, 15]

    def test_metadata_saved_as_json(self, tmp_path: Path):
        """Should save metadata as readable JSON."""
        manager = CheckpointManager(tmp_path)
        model = _TinyModel()

        manager.save(
            model,
            step=42,
            metrics={'loss': 1.5, 'bpb': 0.9},
            extra={'experiment': 'test'},
        )

        meta_path = tmp_path / 'step_42' / 'metadata.json'
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta['step'] == 42
        assert meta['metrics']['bpb'] == 0.9
        assert meta['extra']['experiment'] == 'test'

    def test_save_with_optimizer(self, tmp_path: Path):
        """Should optionally save optimizer state."""
        manager = CheckpointManager(tmp_path)
        model = _TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Run a step to populate optimizer state
        loss = model(torch.randn(1, 16)).sum()
        loss.backward()
        optimizer.step()

        manager.save(model, step=1, optimizer=optimizer)

        # Load into fresh optimizer
        fresh_model = _TinyModel()
        fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        manager.load(fresh_model, step=1, optimizer=fresh_opt)

        # Optimizer state should be populated
        assert len(fresh_opt.state) > 0
