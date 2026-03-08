"""Tests for the experiment runner (autoresearch framework).

The experiment runner is the core loop for automated research:
1. Define experiment config (model, data, hyperparams)
2. Train for N steps
3. Evaluate metric (BPB, accuracy, etc.)
4. Log results and keep/discard based on improvement

Inspired by Karpathy's autoresearch: iterate fast, measure
everything, keep what works.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn


class _TinyModel(nn.Module):
    """Minimal model for testing experiment runner."""

    def __init__(self, vocab_size: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.linear = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(self.embed(x))


class TestExperimentConfig:
    """Test experiment configuration."""

    def test_create_config(self):
        """ExperimentConfig should hold all experiment parameters."""
        from lmt.research.experiment import ExperimentRunConfig

        config = ExperimentRunConfig(
            name='test_run',
            train_steps=10,
            eval_interval=5,
            lr=1e-4,
        )
        assert config.name == 'test_run'
        assert config.train_steps == 10

    def test_config_to_dict(self):
        """Config should be serializable to dict."""
        from lmt.research.experiment import ExperimentRunConfig

        config = ExperimentRunConfig(name='test')
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d['name'] == 'test'

    def test_config_from_dict(self):
        """Config should be deserializable from dict."""
        from lmt.research.experiment import ExperimentRunConfig

        d = {'name': 'test', 'train_steps': 50, 'lr': 3e-4}
        config = ExperimentRunConfig.from_dict(d)
        assert config.name == 'test'
        assert config.train_steps == 50
        assert config.lr == 3e-4


class TestMetricTracker:
    """Test metric logging and comparison."""

    def test_log_metric(self):
        """Should record metrics with step numbers."""
        from lmt.research.experiment import MetricTracker

        tracker = MetricTracker()
        tracker.log('loss', 2.5, step=0)
        tracker.log('loss', 2.0, step=1)

        history = tracker.get_history('loss')
        assert len(history) == 2
        assert history[0] == (0, 2.5)
        assert history[1] == (1, 2.0)

    def test_get_best(self):
        """Should return the best (lowest) value for a metric."""
        from lmt.research.experiment import MetricTracker

        tracker = MetricTracker()
        tracker.log('bpb', 1.5, step=0)
        tracker.log('bpb', 1.2, step=1)
        tracker.log('bpb', 1.3, step=2)

        assert tracker.get_best('bpb') == 1.2

    def test_get_best_higher_is_better(self):
        """Should support metrics where higher is better."""
        from lmt.research.experiment import MetricTracker

        tracker = MetricTracker()
        tracker.log('accuracy', 0.5, step=0)
        tracker.log('accuracy', 0.8, step=1)
        tracker.log('accuracy', 0.6, step=2)

        assert tracker.get_best('accuracy', higher_is_better=True) == 0.8

    def test_is_improved(self):
        """Should detect whether the latest metric improved."""
        from lmt.research.experiment import MetricTracker

        tracker = MetricTracker()
        tracker.log('bpb', 1.5, step=0)
        tracker.log('bpb', 1.2, step=1)

        assert tracker.is_improved('bpb') is True

        tracker.log('bpb', 1.4, step=2)
        assert tracker.is_improved('bpb') is False

    def test_to_dict(self):
        """Tracker should be serializable for logging."""
        from lmt.research.experiment import MetricTracker

        tracker = MetricTracker()
        tracker.log('loss', 2.0, step=0)
        tracker.log('bpb', 1.5, step=0)

        d = tracker.to_dict()
        assert 'loss' in d
        assert 'bpb' in d

    def test_empty_history(self):
        """Should handle querying metrics that don't exist."""
        from lmt.research.experiment import MetricTracker

        tracker = MetricTracker()
        assert tracker.get_history('nonexistent') == []
        assert tracker.get_best('nonexistent') is None


class TestExperimentRunner:
    """Test the core experiment loop."""

    def test_run_experiment(self, tmp_path: Path):
        """Run a basic training experiment and get results."""
        from lmt.research.experiment import (
            ExperimentResult,
            ExperimentRunConfig,
            ExperimentRunner,
        )

        model = _TinyModel()
        data = torch.randint(0, 32, (20, 8))
        config = ExperimentRunConfig(
            name='test_exp',
            train_steps=5,
            eval_interval=5,
            lr=1e-3,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        result = runner.run(model, data, config)

        assert isinstance(result, ExperimentResult)
        assert result.config.name == 'test_exp'
        assert result.final_loss is not None
        assert isinstance(result.final_loss, float)

    def test_loss_decreases(self, tmp_path: Path):
        """Training should decrease loss on a simple task."""
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        model = _TinyModel()
        # Repeated data so the model can memorize
        data = torch.randint(0, 32, (4, 8)).repeat(10, 1)
        config = ExperimentRunConfig(
            name='memorize',
            train_steps=20,
            eval_interval=20,
            lr=1e-2,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        result = runner.run(model, data, config)

        # Loss should have gone down
        tracker = result.metrics
        history = tracker.get_history('train_loss')
        assert history[-1][1] < history[0][1]

    def test_saves_result(self, tmp_path: Path):
        """Experiment result should be saved to disk."""
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        model = _TinyModel()
        data = torch.randint(0, 32, (10, 8))
        config = ExperimentRunConfig(
            name='save_test',
            train_steps=3,
            eval_interval=3,
            lr=1e-3,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        runner.run(model, data, config)

        result_file = tmp_path / 'save_test' / 'result.json'
        assert result_file.exists()

        with open(result_file) as f:
            saved = json.load(f)
        assert saved['config']['name'] == 'save_test'
        assert 'final_loss' in saved

    def test_saves_checkpoint(self, tmp_path: Path):
        """Experiment should save model checkpoint."""
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        model = _TinyModel()
        data = torch.randint(0, 32, (10, 8))
        config = ExperimentRunConfig(
            name='ckpt_test',
            train_steps=3,
            eval_interval=3,
            lr=1e-3,
            save_checkpoint=True,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        runner.run(model, data, config)

        ckpt_file = tmp_path / 'ckpt_test' / 'checkpoint.pt'
        assert ckpt_file.exists()

    def test_multiple_experiments(self, tmp_path: Path):
        """Runner should handle multiple sequential experiments."""
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))

        for i in range(3):
            model = _TinyModel()
            data = torch.randint(0, 32, (10, 8))
            config = ExperimentRunConfig(
                name=f'exp_{i}',
                train_steps=3,
                eval_interval=3,
                lr=1e-3,
            )
            runner.run(model, data, config)

        assert len(runner.results) == 3
        assert all(r.final_loss is not None for r in runner.results)

    def test_eval_interval_logs_eval_loss(self, tmp_path: Path):
        """Eval loss should be logged at eval_interval steps."""
        from lmt.research.experiment import (
            ExperimentRunConfig,
            ExperimentRunner,
        )

        model = _TinyModel()
        data = torch.randint(0, 32, (10, 8))
        config = ExperimentRunConfig(
            name='eval_test',
            train_steps=10,
            eval_interval=5,
            lr=1e-3,
        )

        runner = ExperimentRunner(output_dir=str(tmp_path))
        result = runner.run(model, data, config)

        eval_history = result.metrics.get_history('eval_loss')
        # Should have 2 eval points: step 4 and step 9 (0-indexed)
        assert len(eval_history) == 2
        assert eval_history[0][0] == 4  # step 5 (0-indexed)
        assert eval_history[1][0] == 9  # step 10 (0-indexed)
