"""Tests for hyperparameter sweep utility."""

import pytest

from lmt.research.sweep import SweepConfig, generate_grid, generate_random


class TestSweepConfig:
    """Test SweepConfig for defining search spaces."""

    def test_init(self):
        """Should define a search space."""
        config = SweepConfig(
            parameters={
                'lr': [1e-3, 1e-4, 1e-5],
                'batch_size': [16, 32],
            }
        )
        assert 'lr' in config.parameters
        assert 'batch_size' in config.parameters

    def test_grid_size(self):
        """Should compute total grid size."""
        config = SweepConfig(
            parameters={
                'lr': [1e-3, 1e-4],
                'batch_size': [16, 32, 64],
            }
        )
        assert config.grid_size == 6  # 2 * 3

    def test_grid_size_single(self):
        """Should handle single parameter."""
        config = SweepConfig(parameters={'lr': [1e-3, 1e-4, 1e-5]})
        assert config.grid_size == 3

    def test_grid_size_empty(self):
        """Should handle empty parameters."""
        config = SweepConfig(parameters={})
        assert config.grid_size == 0


class TestGenerateGrid:
    """Test grid search configuration generation."""

    def test_generates_all_combinations(self):
        """Should produce the full Cartesian product."""
        config = SweepConfig(
            parameters={
                'lr': [1e-3, 1e-4],
                'batch_size': [16, 32],
            }
        )
        configs = generate_grid(config)
        assert len(configs) == 4

    def test_all_values_present(self):
        """Should include all parameter values."""
        config = SweepConfig(
            parameters={
                'lr': [1e-3, 1e-4],
            }
        )
        configs = generate_grid(config)
        lrs = [c['lr'] for c in configs]
        assert 1e-3 in lrs
        assert 1e-4 in lrs

    def test_combinations_correct(self):
        """Should produce correct Cartesian product."""
        config = SweepConfig(
            parameters={
                'a': [1, 2],
                'b': ['x', 'y'],
            }
        )
        configs = generate_grid(config)
        assert {'a': 1, 'b': 'x'} in configs
        assert {'a': 1, 'b': 'y'} in configs
        assert {'a': 2, 'b': 'x'} in configs
        assert {'a': 2, 'b': 'y'} in configs

    def test_empty_parameters(self):
        """Should return empty list for empty parameters."""
        config = SweepConfig(parameters={})
        configs = generate_grid(config)
        assert configs == []

    def test_single_values(self):
        """Should handle single-value parameters."""
        config = SweepConfig(parameters={'lr': [1e-3], 'batch_size': [32]})
        configs = generate_grid(config)
        assert len(configs) == 1
        assert configs[0] == {'lr': 1e-3, 'batch_size': 32}

    def test_with_base_config(self):
        """Should merge with base config."""
        config = SweepConfig(
            parameters={'lr': [1e-3, 1e-4]},
            base_config={'batch_size': 32, 'warmup_steps': 10},
        )
        configs = generate_grid(config)
        for c in configs:
            assert c['batch_size'] == 32
            assert c['warmup_steps'] == 10


class TestGenerateRandom:
    """Test random search configuration generation."""

    def test_generates_n_configs(self):
        """Should produce exactly n configurations."""
        config = SweepConfig(
            parameters={
                'lr': [1e-3, 1e-4, 1e-5],
                'batch_size': [16, 32, 64],
            }
        )
        configs = generate_random(config, n=5, seed=42)
        assert len(configs) == 5

    def test_reproducible(self):
        """Should be reproducible with same seed."""
        config = SweepConfig(
            parameters={
                'lr': [1e-3, 1e-4, 1e-5],
                'batch_size': [16, 32, 64],
            }
        )
        configs1 = generate_random(config, n=3, seed=42)
        configs2 = generate_random(config, n=3, seed=42)
        assert configs1 == configs2

    def test_values_from_options(self):
        """Should only use values from the defined options."""
        config = SweepConfig(parameters={'lr': [1e-3, 1e-4]})
        configs = generate_random(config, n=10, seed=42)
        for c in configs:
            assert c['lr'] in [1e-3, 1e-4]

    def test_with_base_config(self):
        """Should merge with base config."""
        config = SweepConfig(
            parameters={'lr': [1e-3, 1e-4]},
            base_config={'batch_size': 32},
        )
        configs = generate_random(config, n=3, seed=42)
        for c in configs:
            assert c['batch_size'] == 32

    def test_n_zero(self):
        """Should return empty list when n=0."""
        config = SweepConfig(parameters={'lr': [1e-3]})
        configs = generate_random(config, n=0, seed=42)
        assert configs == []

    def test_invalid_n(self):
        """Should reject negative n."""
        config = SweepConfig(parameters={'lr': [1e-3]})
        with pytest.raises(ValueError):
            generate_random(config, n=-1, seed=42)
