"""Tests for reproducibility utilities.

Per research principles: reproducibility is non-negotiable.
These utilities ensure experiments can be exactly replicated.
"""

import torch


class TestSeedSetting:
    """Test deterministic seed setting."""

    def test_set_seed_makes_torch_deterministic(self):
        """Same seed should produce same random tensors."""
        from lmt.training.reproducibility import set_seed

        set_seed(42)
        a = torch.randn(10)

        set_seed(42)
        b = torch.randn(10)

        assert torch.equal(a, b)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different tensors."""
        from lmt.training.reproducibility import set_seed

        set_seed(42)
        a = torch.randn(10)

        set_seed(123)
        b = torch.randn(10)

        assert not torch.equal(a, b)

    def test_set_seed_affects_python_random(self):
        """set_seed should also seed Python's random module."""
        import random

        from lmt.training.reproducibility import set_seed

        set_seed(42)
        a = random.random()

        set_seed(42)
        b = random.random()

        assert a == b


class TestExperimentConfig:
    """Test experiment configuration logging."""

    def test_create_experiment_config(self):
        """ExperimentConfig captures all reproducibility info."""
        from lmt.training.reproducibility import ExperimentConfig

        config = ExperimentConfig(
            seed=42,
            model_type='qwen3',
            description='Test experiment',
        )
        assert config.seed == 42
        assert config.model_type == 'qwen3'

    def test_experiment_config_to_dict(self):
        """Config should be serializable to dict."""
        from lmt.training.reproducibility import ExperimentConfig

        config = ExperimentConfig(
            seed=42,
            model_type='qwen3',
            description='Test',
        )
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d['seed'] == 42
        assert 'git_hash' in d
        assert 'python_version' in d
