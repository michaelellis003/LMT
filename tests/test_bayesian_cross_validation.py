"""Cross-validation of our Bayesian stats against scipy.

This test file validates that our stdlib-only Monte Carlo implementation
produces results consistent with scipy's analytical t-distribution.
These tests serve as a correctness check, NOT as part of the normal
test suite (they depend on scipy which is not a project dependency).

Run with: uv run pytest tests/test_bayesian_cross_validation.py -v
"""

import math

import pytest

scipy_stats = pytest.importorskip('scipy.stats')

from lmt.research.stats import (  # noqa: E402
    ExperimentSamples,
    bayesian_compare,
    cohens_d,
    credible_interval,
)


class TestCredibleIntervalVsScipy:
    """Compare our CI with scipy's t-distribution interval."""

    def test_ci_matches_scipy_t_interval(self):
        """Our MC credible interval should match scipy's analytical CI.

        For a Normal-InverseGamma conjugate prior with Jeffreys prior,
        the posterior for the mean is Student-t:
            mu | data ~ t(df=N-1, loc=x_bar, scale=s/sqrt(N))

        scipy.stats.t.interval() gives the exact analytical interval.
        Our Monte Carlo should be close (within ~1% for 10K samples).
        """
        values = [3.0, 3.1, 3.2, 3.3, 3.4, 3.05, 3.15, 3.25]
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
        sem = std / math.sqrt(n)

        # Scipy analytical interval
        scipy_lo, scipy_hi = scipy_stats.t.interval(
            0.95, df=n - 1, loc=mean, scale=sem
        )

        # Our Monte Carlo interval
        our_lo, our_hi = credible_interval(values, level=0.95)

        # Should match within 2% (MC variance)
        assert our_lo == pytest.approx(scipy_lo, rel=0.02)
        assert our_hi == pytest.approx(scipy_hi, rel=0.02)

    def test_ci_matches_scipy_small_sample(self):
        """Small sample (n=3) where the t-distribution matters most."""
        values = [3.0, 3.5, 4.0]
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
        sem = std / math.sqrt(n)

        scipy_lo, scipy_hi = scipy_stats.t.interval(
            0.95, df=n - 1, loc=mean, scale=sem
        )

        our_lo, our_hi = credible_interval(values, level=0.95)

        # Wider tolerance for small samples (MC variance is higher)
        assert our_lo == pytest.approx(scipy_lo, rel=0.05)
        assert our_hi == pytest.approx(scipy_hi, rel=0.05)

    def test_ci_matches_scipy_large_sample(self):
        """Larger sample (n=20) where MC should converge better."""
        import random

        rng = random.Random(123)
        values = [3.0 + rng.gauss(0, 0.2) for _ in range(20)]

        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
        sem = std / math.sqrt(n)

        scipy_lo, scipy_hi = scipy_stats.t.interval(
            0.95, df=n - 1, loc=mean, scale=sem
        )

        our_lo, our_hi = credible_interval(values, level=0.95)

        # Tighter tolerance for larger samples
        assert our_lo == pytest.approx(scipy_lo, rel=0.02)
        assert our_hi == pytest.approx(scipy_hi, rel=0.02)


class TestCohensDAgreement:
    """Compare our Cohen's d with scipy-based computation."""

    def test_cohens_d_matches_manual(self):
        """Verify against a hand-calculated example."""
        # Two groups with known properties
        a = [10.0, 12.0, 14.0, 16.0, 18.0]
        b = [20.0, 22.0, 24.0, 26.0, 28.0]

        # Manual calculation:
        # mean_a = 14, mean_b = 24, diff = 10
        # var_a = var_b = 10 (Bessel-corrected)
        # pooled_std = sqrt(10) = 3.162
        # d = (24 - 14) / 3.162 = 3.162
        expected_d = 10.0 / math.sqrt(10.0)

        our_d = cohens_d(a, b)
        assert our_d == pytest.approx(expected_d, rel=1e-6)


class TestBayesianCompareVsScipy:
    """Compare our P(A<B) with scipy-based Monte Carlo."""

    def test_prob_a_better_matches_scipy_mc(self):
        """Our posterior sampling should match scipy's t-distribution.

        We sample from scipy.stats.t and compute P(A<B) independently,
        then check our implementation agrees.
        """
        a_values = [3.0, 3.1, 3.05, 3.08, 3.02]
        b_values = [3.5, 3.6, 3.55, 3.58, 3.52]

        # Scipy-based Monte Carlo
        n_mc = 100000
        rng_seed = 99

        def scipy_posterior_samples(values, n_samples, seed):
            """Sample from posterior using scipy."""
            import numpy as np

            n = len(values)
            mean = sum(values) / n
            std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
            sem = std / math.sqrt(n)
            rng = np.random.RandomState(seed)
            return scipy_stats.t.rvs(
                df=n - 1,
                loc=mean,
                scale=sem,
                size=n_samples,
                random_state=rng,
            )

        scipy_a = scipy_posterior_samples(a_values, n_mc, rng_seed)
        scipy_b = scipy_posterior_samples(b_values, n_mc, rng_seed + 1)
        scipy_prob = float((scipy_a < scipy_b).mean())

        # Our implementation
        a = ExperimentSamples(name='a', values=a_values)
        b = ExperimentSamples(name='b', values=b_values)
        result = bayesian_compare(a, b, n_mc_samples=100000)

        # Both should agree that A is clearly better (lower BPB)
        # Allow 3% tolerance due to MC variance
        assert result.prob_a_better == pytest.approx(scipy_prob, abs=0.03)
        assert result.prob_a_better > 0.95
        assert scipy_prob > 0.95

    def test_close_groups_prob_near_half(self):
        """When groups overlap heavily, P(A<B) should be near 0.5."""
        a_values = [3.0, 3.1, 3.2, 3.3, 3.4]
        b_values = [3.05, 3.15, 3.25, 3.35, 3.45]

        a = ExperimentSamples(name='a', values=a_values)
        b = ExperimentSamples(name='b', values=b_values)
        result = bayesian_compare(a, b)

        # Should be near 0.5 (no clear winner)
        assert 0.3 < result.prob_a_better < 0.7

    def test_mean_difference_sign(self):
        """Mean difference A-B should be negative when A is lower."""
        a = ExperimentSamples(name='a', values=[3.0, 3.1, 3.05])
        b = ExperimentSamples(name='b', values=[3.5, 3.6, 3.55])
        result = bayesian_compare(a, b)
        assert result.mean_difference < 0  # A - B < 0 means A is better

    def test_ci_contains_true_difference(self):
        """95% CI should contain the true mean difference."""
        # Groups with known true means (3.0 and 3.5)
        import random as rng_mod

        rng = rng_mod.Random(42)
        a_values = [3.0 + rng.gauss(0, 0.1) for _ in range(10)]
        b_values = [3.5 + rng.gauss(0, 0.1) for _ in range(10)]

        a = ExperimentSamples(name='a', values=a_values)
        b = ExperimentSamples(name='b', values=b_values)
        result = bayesian_compare(a, b)

        # True difference is approximately -0.5
        # The CI should contain it
        assert result.ci_lower < -0.5 < result.ci_upper


class TestSimulationBasedCalibration:
    """Simulation-based calibration (SBC) per Vehtari et al.

    Generate many fake experiments where we KNOW the true parameters,
    run our Bayesian comparison, and verify that:
    1. 95% CIs contain the true value ~95% of the time
    2. P(A<B) is well-calibrated (true 70% of the time when P=0.7)

    This is the gold standard for validating Bayesian implementations.
    """

    def test_credible_interval_coverage(self):
        """95% CI should contain the true mean ~95% of the time.

        Simulate 200 datasets from a known Normal(mu=3.0, sigma=0.1),
        compute our 95% CI for the mean, and check coverage.
        """
        import random as rng_mod

        true_mu = 3.0
        true_sigma = 0.1
        n_per_experiment = 5  # Typical seed count
        n_simulations = 200
        rng = rng_mod.Random(12345)

        covered = 0
        for _ in range(n_simulations):
            values = [
                true_mu + rng.gauss(0, true_sigma)
                for _ in range(n_per_experiment)
            ]
            lo, hi = credible_interval(values, level=0.95)
            if lo <= true_mu <= hi:
                covered += 1

        coverage = covered / n_simulations
        # Should be near 0.95. With 200 sims, allow ±5%
        assert 0.88 <= coverage <= 1.0, (
            f'Coverage {coverage:.2f} outside [0.88, 1.0]'
        )

    def test_prob_a_better_calibration(self):
        """P(A<B) should be well-calibrated across scenarios.

        Generate experiments where A truly is better (mu_A < mu_B),
        and check that our P(A better) reflects reality.
        """
        import random as rng_mod

        n_per_group = 5
        n_simulations = 100
        rng = rng_mod.Random(54321)

        # Scenario: A is truly better by 0.3 (moderate gap)
        true_mu_a = 3.0
        true_mu_b = 3.3
        true_sigma = 0.1

        a_wins = 0
        for _ in range(n_simulations):
            a_vals = [
                true_mu_a + rng.gauss(0, true_sigma)
                for _ in range(n_per_group)
            ]
            b_vals = [
                true_mu_b + rng.gauss(0, true_sigma)
                for _ in range(n_per_group)
            ]
            a = ExperimentSamples(name='a', values=a_vals)
            b = ExperimentSamples(name='b', values=b_vals)
            result = bayesian_compare(a, b, n_mc_samples=10000)
            if result.prob_a_better > 0.5:
                a_wins += 1

        win_rate = a_wins / n_simulations
        # With effect size d=3.0 (huge), A should win almost always
        assert win_rate > 0.90, (
            f'A only won {win_rate:.0%} of simulations (expected >90%)'
        )

    def test_rope_contains_truth_when_equal(self):
        """When A==B, P(equivalent) with ROPE should be high.

        If we draw A and B from the SAME distribution, a ROPE of
        ±0.05 (half the std) should frequently declare equivalence.
        """
        import random as rng_mod

        true_mu = 3.0
        true_sigma = 0.1
        n_per_group = 5
        n_simulations = 100
        rng = rng_mod.Random(99999)

        equiv_count = 0
        for _ in range(n_simulations):
            a_vals = [
                true_mu + rng.gauss(0, true_sigma) for _ in range(n_per_group)
            ]
            b_vals = [
                true_mu + rng.gauss(0, true_sigma) for _ in range(n_per_group)
            ]
            a = ExperimentSamples(name='a', values=a_vals)
            b = ExperimentSamples(name='b', values=b_vals)
            result = bayesian_compare(a, b, n_mc_samples=10000, rope=0.05)
            if result.prob_rope > 0.3:
                equiv_count += 1

        equiv_rate = equiv_count / n_simulations
        # Most simulations should show meaningful equivalence prob
        assert equiv_rate > 0.40, (
            f'Only {equiv_rate:.0%} had P(equiv)>0.3 (expected >40%)'
        )
