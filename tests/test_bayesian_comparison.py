"""Tests for Bayesian statistical comparison framework."""

import math

import pytest

from lmt.research.stats import (
    BayesianComparison,
    ExperimentSamples,
    bayesian_compare,
    cohens_d,
    credible_interval,
    multi_seed_summary,
)


class TestExperimentSamples:
    """Test ExperimentSamples dataclass."""

    def test_basic_stats(self):
        """Should compute mean and std from samples."""
        samples = ExperimentSamples(
            name='test', values=[1.0, 2.0, 3.0, 4.0, 5.0]
        )
        assert samples.mean == pytest.approx(3.0)
        assert samples.std == pytest.approx(math.sqrt(2.5), rel=1e-5)
        assert samples.n == 5

    def test_single_sample(self):
        """Should handle single sample (std=0)."""
        samples = ExperimentSamples(name='test', values=[3.14])
        assert samples.mean == pytest.approx(3.14)
        assert samples.std == 0.0
        assert samples.n == 1

    def test_sem(self):
        """Should compute standard error of the mean."""
        samples = ExperimentSamples(
            name='test', values=[1.0, 2.0, 3.0, 4.0, 5.0]
        )
        expected_sem = samples.std / math.sqrt(5)
        assert samples.sem == pytest.approx(expected_sem, rel=1e-5)


class TestCredibleInterval:
    """Test credible interval computation."""

    def test_basic_interval(self):
        """Should return lower and upper bounds."""
        values = [3.0, 3.1, 3.2, 3.3, 3.4]
        lo, hi = credible_interval(values, level=0.95)
        assert lo < hi
        assert lo >= min(values)
        assert hi <= max(values)

    def test_wider_with_more_variance(self):
        """Higher variance should give wider interval."""
        tight = [3.0, 3.01, 3.02, 3.03, 3.04]
        wide = [2.0, 3.0, 4.0, 5.0, 6.0]
        lo_t, hi_t = credible_interval(tight, level=0.95)
        lo_w, hi_w = credible_interval(wide, level=0.95)
        assert (hi_w - lo_w) > (hi_t - lo_t)

    def test_single_value(self):
        """Single value should return point interval."""
        lo, hi = credible_interval([3.0], level=0.95)
        assert lo == pytest.approx(3.0)
        assert hi == pytest.approx(3.0)


class TestCohensD:
    """Test Cohen's d effect size."""

    def test_identical_groups(self):
        """Identical groups should have d=0."""
        a = [3.0, 3.1, 3.2, 3.3, 3.4]
        d = cohens_d(a, a)
        assert d == pytest.approx(0.0)

    def test_large_effect(self):
        """Well-separated groups should have large d."""
        a = [1.0, 1.1, 1.2, 1.3, 1.4]
        b = [5.0, 5.1, 5.2, 5.3, 5.4]
        d = cohens_d(a, b)
        assert abs(d) > 2.0  # Very large effect

    def test_sign_convention(self):
        """Positive d means a < b (a is better for BPB)."""
        better = [3.0, 3.1, 3.2]
        worse = [4.0, 4.1, 4.2]
        d = cohens_d(better, worse)
        assert d > 0  # Positive = first group is lower


class TestBayesianCompare:
    """Test Bayesian comparison function."""

    def test_clear_winner(self):
        """Should detect a clear winner."""
        a = ExperimentSamples(
            name='better', values=[3.0, 3.1, 3.05, 3.08, 3.02]
        )
        b = ExperimentSamples(
            name='worse', values=[3.5, 3.6, 3.55, 3.58, 3.52]
        )
        result = bayesian_compare(a, b)
        assert isinstance(result, BayesianComparison)
        assert result.prob_a_better > 0.9
        assert result.mean_difference < 0  # a has lower BPB

    def test_no_difference(self):
        """Nearly identical groups should have prob ~0.5."""
        a = ExperimentSamples(name='group_a', values=[3.0, 3.1, 3.2, 3.3, 3.4])
        b = ExperimentSamples(
            name='group_b', values=[3.05, 3.15, 3.25, 3.35, 3.45]
        )
        result = bayesian_compare(a, b)
        # With this much overlap, probability should be near 0.5
        assert 0.2 < result.prob_a_better < 0.8

    def test_credible_interval_in_result(self):
        """Should include credible interval for the difference."""
        a = ExperimentSamples(name='a', values=[3.0, 3.1, 3.2])
        b = ExperimentSamples(name='b', values=[3.5, 3.6, 3.7])
        result = bayesian_compare(a, b)
        assert result.ci_lower < result.ci_upper
        assert result.ci_lower < 0  # a - b should be negative

    def test_effect_size_in_result(self):
        """Should include Cohen's d."""
        a = ExperimentSamples(name='a', values=[3.0, 3.1, 3.2])
        b = ExperimentSamples(name='b', values=[3.5, 3.6, 3.7])
        result = bayesian_compare(a, b)
        assert result.cohens_d != 0.0

    def test_three_way_probabilities_sum_to_one(self):
        """P(A better) + P(equiv) + P(B better) should equal 1."""
        a = ExperimentSamples(name='a', values=[3.0, 3.1, 3.2])
        b = ExperimentSamples(name='b', values=[3.5, 3.6, 3.7])
        result = bayesian_compare(a, b, rope=0.01)
        total = result.prob_a_better + result.prob_rope + result.prob_b_better
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_rope_zero_degenerates_to_two_way(self):
        """With rope=0, prob_rope should be 0 and probs sum to 1."""
        a = ExperimentSamples(name='a', values=[3.0, 3.1, 3.2])
        b = ExperimentSamples(name='b', values=[3.5, 3.6, 3.7])
        result = bayesian_compare(a, b, rope=0.0)
        assert result.prob_rope == pytest.approx(0.0, abs=1e-6)
        assert result.prob_a_better + result.prob_b_better == pytest.approx(
            1.0, abs=1e-6
        )

    def test_large_rope_gives_high_equiv_prob(self):
        """Very large ROPE should make most differences 'equivalent'."""
        a = ExperimentSamples(name='a', values=[3.0, 3.1, 3.2])
        b = ExperimentSamples(name='b', values=[3.5, 3.6, 3.7])
        result = bayesian_compare(a, b, rope=10.0)  # Huge ROPE
        assert result.prob_rope > 0.99

    def test_rope_with_clear_difference(self):
        """Small ROPE with big difference should still detect winner."""
        a = ExperimentSamples(name='a', values=[3.0, 3.1, 3.05, 3.08, 3.02])
        b = ExperimentSamples(name='b', values=[3.5, 3.6, 3.55, 3.58, 3.52])
        result = bayesian_compare(a, b, rope=0.01)
        assert result.prob_a_better > 0.9
        assert result.prob_rope < 0.05


class TestMultiSeedSummary:
    """Test multi-seed summary generation."""

    def test_basic_summary(self):
        """Should produce a summary dict."""
        samples = ExperimentSamples(
            name='test_model', values=[3.0, 3.1, 3.2, 3.3, 3.4]
        )
        summary = multi_seed_summary(samples)
        assert summary['name'] == 'test_model'
        assert summary['n_seeds'] == 5
        assert 'mean' in summary
        assert 'std' in summary
        assert 'ci_lower' in summary
        assert 'ci_upper' in summary
        assert 'sem' in summary

    def test_summary_values_reasonable(self):
        """Summary values should be consistent."""
        samples = ExperimentSamples(name='test', values=[3.0, 3.1, 3.2])
        summary = multi_seed_summary(samples)
        assert summary['ci_lower'] <= summary['mean']
        assert summary['ci_upper'] >= summary['mean']
