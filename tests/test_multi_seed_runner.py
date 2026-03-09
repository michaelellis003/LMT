"""Tests for multi-seed experiment runner."""

import pytest
import torch

from lmt.research.multi_seed import (
    MultiSeedConfig,
    MultiSeedResult,
    SeedResult,
    run_multi_seed,
)
from lmt.research.stats import ExperimentSamples, bayesian_compare


class TestMultiSeedConfig:
    """Test MultiSeedConfig dataclass."""

    def test_default_seeds(self):
        """Should have sensible default seeds."""
        config = MultiSeedConfig(n_seeds=5)
        assert len(config.seeds) == 5
        assert len(set(config.seeds)) == 5  # All unique

    def test_explicit_seeds(self):
        """Should accept explicit seed list."""
        config = MultiSeedConfig(n_seeds=3, seeds=[10, 20, 30])
        assert config.seeds == [10, 20, 30]

    def test_seed_count_mismatch_raises(self):
        """Should raise if n_seeds != len(seeds)."""
        with pytest.raises(ValueError, match='n_seeds'):
            MultiSeedConfig(n_seeds=5, seeds=[1, 2, 3])


class TestSeedResult:
    """Test SeedResult dataclass."""

    def test_basic_fields(self):
        """Should store seed, metric, and elapsed time."""
        result = SeedResult(seed=42, best_bpb=3.14, elapsed=10.5)
        assert result.seed == 42
        assert result.best_bpb == pytest.approx(3.14)
        assert result.elapsed == pytest.approx(10.5)


class TestMultiSeedResult:
    """Test MultiSeedResult dataclass."""

    def test_to_samples(self):
        """Should convert to ExperimentSamples."""
        result = MultiSeedResult(
            variant_name='test',
            seed_results=[
                SeedResult(seed=42, best_bpb=3.0, elapsed=1.0),
                SeedResult(seed=43, best_bpb=3.1, elapsed=1.0),
                SeedResult(seed=44, best_bpb=3.2, elapsed=1.0),
            ],
        )
        samples = result.to_samples()
        assert isinstance(samples, ExperimentSamples)
        assert samples.name == 'test'
        assert samples.values == [3.0, 3.1, 3.2]
        assert samples.n == 3

    def test_summary_stats(self):
        """Should compute mean and std across seeds."""
        result = MultiSeedResult(
            variant_name='test',
            seed_results=[
                SeedResult(seed=42, best_bpb=3.0, elapsed=1.0),
                SeedResult(seed=43, best_bpb=3.1, elapsed=1.0),
                SeedResult(seed=44, best_bpb=3.2, elapsed=1.0),
            ],
        )
        assert result.mean_bpb == pytest.approx(3.1)
        assert result.std_bpb == pytest.approx(0.1, rel=1e-3)


class TestRunMultiSeed:
    """Test the multi-seed runner with a tiny model."""

    def test_runs_multiple_seeds(self):
        """Should run experiment with multiple seeds and collect results."""

        # Define a trivial train function that returns a BPB based on seed
        def train_fn(seed: int) -> float:
            """Fake training that returns deterministic BPB."""
            # Simulate slight variation across seeds
            torch.manual_seed(seed)
            return 3.0 + torch.randn(1).item() * 0.1

        config = MultiSeedConfig(n_seeds=3, seeds=[42, 43, 44])
        result = run_multi_seed(
            variant_name='test_model',
            train_fn=train_fn,
            config=config,
        )

        assert isinstance(result, MultiSeedResult)
        assert result.variant_name == 'test_model'
        assert len(result.seed_results) == 3
        # Each seed should produce a different result
        bpbs = [r.best_bpb for r in result.seed_results]
        assert len(set(bpbs)) == 3  # All different due to different seeds

    def test_results_are_deterministic(self):
        """Same seeds should give same results."""

        def train_fn(seed: int) -> float:
            torch.manual_seed(seed)
            return 3.0 + torch.randn(1).item() * 0.1

        config = MultiSeedConfig(n_seeds=3, seeds=[42, 43, 44])
        result1 = run_multi_seed('test', train_fn, config)
        result2 = run_multi_seed('test', train_fn, config)

        bpbs1 = [r.best_bpb for r in result1.seed_results]
        bpbs2 = [r.best_bpb for r in result2.seed_results]
        for a, b in zip(bpbs1, bpbs2, strict=True):
            assert a == pytest.approx(b)

    def test_bayesian_comparison_integration(self):
        """Should work end-to-end with bayesian_compare."""

        def good_model(seed: int) -> float:
            torch.manual_seed(seed)
            return 3.0 + torch.randn(1).item() * 0.05

        def bad_model(seed: int) -> float:
            torch.manual_seed(seed)
            return 3.5 + torch.randn(1).item() * 0.05

        config = MultiSeedConfig(n_seeds=5, seeds=[42, 43, 44, 45, 46])
        result_a = run_multi_seed('good', good_model, config)
        result_b = run_multi_seed('bad', bad_model, config)

        comparison = bayesian_compare(
            result_a.to_samples(), result_b.to_samples()
        )
        # Good model should be clearly better (lower BPB)
        assert comparison.prob_a_better > 0.9
        assert comparison.mean_difference < 0
