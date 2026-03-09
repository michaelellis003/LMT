"""Tests for autoresearch eval integration.

Tests that the experiment runner can automatically evaluate models
after training, computing BPB on validation data. This is the core
loop of the autoresearch framework.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn


class _TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, vocab_size: int = 32, dim: int = 16):
        """Initialize tiny model."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.linear = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(self.embed(x))


class TestExperimentWithEval:
    """Test experiment runner with post-training evaluation."""

    def test_run_with_val_data(self, tmp_path: Path):
        """Should compute val_bpb when val_data is provided."""
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        model = _TinyModel()
        train_data = torch.randint(0, 32, (10, 8))
        val_data = torch.randint(0, 32, (5, 8))

        config = ExperimentRunConfig(
            name='eval_test',
            train_steps=5,
            eval_interval=5,
            lr=1e-3,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        result = runner.run(model, train_data, config, val_data=val_data)

        # val_bpb should be computed and logged
        bpb_history = result.metrics.get_history('val_bpb')
        assert len(bpb_history) >= 1
        assert isinstance(bpb_history[-1][1], float)
        assert bpb_history[-1][1] > 0

    def test_val_bpb_in_saved_result(self, tmp_path: Path):
        """val_bpb should appear in saved result.json."""
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        model = _TinyModel()
        train_data = torch.randint(0, 32, (10, 8))
        val_data = torch.randint(0, 32, (5, 8))

        config = ExperimentRunConfig(
            name='bpb_save',
            train_steps=3,
            eval_interval=3,
            lr=1e-3,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        runner.run(model, train_data, config, val_data=val_data)

        result_file = tmp_path / 'bpb_save' / 'result.json'
        with open(result_file) as fp:
            saved = json.load(fp)

        assert 'val_bpb' in saved['metrics']

    def test_without_val_data(self, tmp_path: Path):
        """Should still work normally without val_data."""
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        model = _TinyModel()
        train_data = torch.randint(0, 32, (10, 8))

        config = ExperimentRunConfig(
            name='no_val',
            train_steps=3,
            eval_interval=3,
            lr=1e-3,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        result = runner.run(model, train_data, config)

        # No val_bpb should be logged
        assert result.metrics.get_history('val_bpb') == []
        assert result.final_loss is not None

    def test_val_bpb_at_eval_interval(self, tmp_path: Path):
        """Should compute val_bpb at each eval_interval."""
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        model = _TinyModel()
        train_data = torch.randint(0, 32, (10, 8))
        val_data = torch.randint(0, 32, (3, 8))

        config = ExperimentRunConfig(
            name='interval_bpb',
            train_steps=10,
            eval_interval=5,
            lr=1e-3,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        result = runner.run(model, train_data, config, val_data=val_data)

        bpb_history = result.metrics.get_history('val_bpb')
        # Should have 2 eval points
        assert len(bpb_history) == 2

    def test_compare_experiments_by_bpb(self, tmp_path: Path):
        """Should be able to compare experiments by val_bpb."""
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        val_data = torch.randint(0, 32, (5, 8))

        for lr in [1e-2, 1e-4]:
            model = _TinyModel()
            torch.manual_seed(42)
            train_data = torch.randint(0, 32, (10, 8))
            config = ExperimentRunConfig(
                name=f'lr_{lr}',
                train_steps=5,
                eval_interval=5,
                lr=lr,
            )
            runner.run(model, train_data, config, val_data=val_data)

        # Both experiments should have val_bpb
        for result in runner.results:
            bpb = result.metrics.get_best('val_bpb')
            assert bpb is not None
            assert bpb > 0

    def test_best_experiment(self, tmp_path: Path):
        """Should identify the best experiment by val_bpb."""
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        val_data = torch.randint(0, 32, (5, 8))

        for name in ['exp_a', 'exp_b']:
            model = _TinyModel()
            train_data = torch.randint(0, 32, (10, 8))
            config = ExperimentRunConfig(
                name=name,
                train_steps=5,
                eval_interval=5,
                lr=1e-3,
            )
            runner.run(model, train_data, config, val_data=val_data)

        best = runner.best_result('val_bpb')
        assert best is not None
        assert best.config.name in ('exp_a', 'exp_b')
