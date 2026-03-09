"""Tests for hyperparameter search in autoresearch framework."""

from pathlib import Path

import torch
import torch.nn as nn

from lmt.research.experiment import ExperimentRunConfig
from lmt.research.search import (
    SearchSpace,
    grid_search,
    random_search,
)


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


class TestSearchSpace:
    """Test search space definition."""

    def test_grid_configs(self):
        """Should generate all combinations for grid search."""
        space = SearchSpace(
            lr=[1e-3, 1e-4],
            batch_size=[4, 8],
        )
        configs = space.grid()
        assert len(configs) == 4
        lrs = {c['lr'] for c in configs}
        assert lrs == {1e-3, 1e-4}

    def test_random_configs(self):
        """Should sample N random configs from the space."""
        space = SearchSpace(
            lr=[1e-3, 1e-4, 1e-5],
            batch_size=[4, 8, 16],
        )
        configs = space.sample(n=5, seed=42)
        assert len(configs) == 5
        for cfg in configs:
            assert cfg['lr'] in [1e-3, 1e-4, 1e-5]
            assert cfg['batch_size'] in [4, 8, 16]

    def test_single_param(self):
        """Should work with a single parameter."""
        space = SearchSpace(lr=[1e-3, 1e-4])
        configs = space.grid()
        assert len(configs) == 2

    def test_to_experiment_configs(self):
        """Should convert to ExperimentRunConfig objects."""
        space = SearchSpace(lr=[1e-3, 1e-4])
        configs = space.to_experiment_configs(
            base_config=ExperimentRunConfig(name='test', train_steps=5),
        )
        assert len(configs) == 2
        assert configs[0].lr == 1e-3
        assert configs[1].lr == 1e-4
        assert all(c.train_steps == 5 for c in configs)
        # Names should be unique
        assert configs[0].name != configs[1].name


class TestGridSearch:
    """Test grid search over hyperparameters."""

    def test_grid_search(self, tmp_path: Path):
        """Should run experiments for all grid combinations."""

        def model_fn():
            return _TinyModel()

        train_data = torch.randint(0, 32, (10, 8))
        space = SearchSpace(lr=[1e-2, 1e-3])

        results = grid_search(
            model_fn=model_fn,
            train_data=train_data,
            search_space=space,
            base_config=ExperimentRunConfig(train_steps=3, eval_interval=3),
            output_dir=str(tmp_path),
        )

        assert len(results) == 2
        assert all(r.final_loss is not None for r in results)

    def test_grid_search_with_val(self, tmp_path: Path):
        """Should compute val_bpb when val_data provided."""

        def model_fn():
            return _TinyModel()

        train_data = torch.randint(0, 32, (10, 8))
        val_data = torch.randint(0, 32, (5, 8))
        space = SearchSpace(lr=[1e-2, 1e-3])

        results = grid_search(
            model_fn=model_fn,
            train_data=train_data,
            search_space=space,
            base_config=ExperimentRunConfig(train_steps=3, eval_interval=3),
            output_dir=str(tmp_path),
            val_data=val_data,
        )

        for result in results:
            bpb = result.metrics.get_best('val_bpb')
            assert bpb is not None


class TestRandomSearch:
    """Test random search over hyperparameters."""

    def test_random_search(self, tmp_path: Path):
        """Should run N random experiments."""

        def model_fn():
            return _TinyModel()

        train_data = torch.randint(0, 32, (10, 8))
        space = SearchSpace(
            lr=[1e-2, 1e-3, 1e-4],
            batch_size=[2, 4],
        )

        results = random_search(
            model_fn=model_fn,
            train_data=train_data,
            search_space=space,
            n_trials=3,
            base_config=ExperimentRunConfig(train_steps=3, eval_interval=3),
            output_dir=str(tmp_path),
            seed=42,
        )

        assert len(results) == 3
