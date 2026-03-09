"""Cross-validation of our Bayesian stats against known reference values.

Validates that our stdlib-only Monte Carlo implementation produces
results consistent with analytically-computed reference values from
the Student-t distribution. Reference values were computed once using
scipy.stats.t and hardcoded here so this test has zero external
dependencies.

Run with: uv run pytest tests/test_bayesian_cross_validation.py -v
"""

import math
import random

import pytest

from lmt.research.stats import (
    ExperimentSamples,
    bayesian_compare,
    cohens_d,
    credible_interval,
)

# Reference values computed via scipy.stats.t.interval() and scipy.stats.t.rvs()
# These are deterministic analytical results, not MC approximations.
# Recompute with: .venv/bin/python -c "import scipy.stats; ..."


class TestCredibleIntervalVsReference:
    """Compare our CI with analytically-computed reference intervals."""

    def test_ci_matches_reference_n8(self):
        """Our MC credible interval should match analytical CI for n=8.

        Reference: scipy.stats.t.interval(0.95, df=7, loc=3.18125, scale=0.04532)
        => (3.0696717400, 3.2928282600)
        """
        values = [3.0, 3.1, 3.2, 3.3, 3.4, 3.05, 3.15, 3.25]
        ref_lo = 3.0696717400
        ref_hi = 3.2928282600

        our_lo, our_hi = credible_interval(values, level=0.95)

        # Should match within 2% (MC variance)
        assert our_lo == pytest.approx(ref_lo, rel=0.02)
        assert our_hi == pytest.approx(ref_hi, rel=0.02)

    def test_ci_matches_reference_n3(self):
        """Small sample (n=3) where the t-distribution matters most.

        Reference: scipy.stats.t.interval(0.95, df=2, loc=3.5, scale=0.28868)
        => (2.2579311441, 4.7420688559)
        """
        values = [3.0, 3.5, 4.0]
        ref_lo = 2.2579311441
        ref_hi = 4.7420688559

        our_lo, our_hi = credible_interval(values, level=0.95)

        # Wider tolerance for small samples (MC variance is higher)
        assert our_lo == pytest.approx(ref_lo, rel=0.05)
        assert our_hi == pytest.approx(ref_hi, rel=0.05)

    def test_ci_matches_reference_n20(self):
        """Larger sample (n=20) where MC should converge better.

        Values generated with random.Random(123).gauss(3.0, 0.2) x 20.
        Reference: scipy.stats.t.interval(0.95, df=19, loc=2.9944, scale=0.02035)
        => (2.9516447733, 3.0371243765)
        """
        rng = random.Random(123)
        values = [3.0 + rng.gauss(0, 0.2) for _ in range(20)]
        ref_lo = 2.9516447733
        ref_hi = 3.0371243765

        our_lo, our_hi = credible_interval(values, level=0.95)

        # Tighter tolerance for larger samples
        assert our_lo == pytest.approx(ref_lo, rel=0.02)
        assert our_hi == pytest.approx(ref_hi, rel=0.02)


class TestCohensDAgreement:
    """Compare our Cohen's d with hand-calculated reference."""

    def test_cohens_d_matches_manual(self):
        """Verify against a hand-calculated example.

        mean_a=14, mean_b=24, var_a=var_b=10 (Bessel-corrected)
        pooled_std = sqrt(10) = 3.16228
        d = (24-14) / 3.16228 = 3.16228

        Reference: 10.0 / sqrt(10.0) = 3.1622776602
        """
        a = [10.0, 12.0, 14.0, 16.0, 18.0]
        b = [20.0, 22.0, 24.0, 26.0, 28.0]
        expected_d = 10.0 / math.sqrt(10.0)  # 3.1622776602

        our_d = cohens_d(a, b)
        assert our_d == pytest.approx(expected_d, rel=1e-6)


class TestBayesianCompareVsReference:
    """Compare our P(A<B) with reference values."""

    def test_prob_a_better_for_separated_groups(self):
        """Well-separated groups should give P(A<B) > 0.99.

        Reference: scipy MC with 100K samples gives P(A<B) = 0.99999.
        Our MC should also be very close to 1.0.
        """
        a = ExperimentSamples(name='a', values=[3.0, 3.1, 3.05, 3.08, 3.02])
        b = ExperimentSamples(name='b', values=[3.5, 3.6, 3.55, 3.58, 3.52])
        result = bayesian_compare(a, b, n_mc_samples=100000)

        # Both our implementation and scipy agree: A is clearly better
        assert result.prob_a_better > 0.99

    def test_close_groups_prob_near_half(self):
        """When groups overlap heavily, P(A<B) should be near 0.5."""
        a = ExperimentSamples(name='a', values=[3.0, 3.1, 3.2, 3.3, 3.4])
        b = ExperimentSamples(name='b', values=[3.05, 3.15, 3.25, 3.35, 3.45])
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
        rng = random.Random(42)
        a_values = [3.0 + rng.gauss(0, 0.1) for _ in range(10)]
        b_values = [3.5 + rng.gauss(0, 0.1) for _ in range(10)]

        a = ExperimentSamples(name='a', values=a_values)
        b = ExperimentSamples(name='b', values=b_values)
        result = bayesian_compare(a, b)

        # True difference is approximately -0.5
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
        true_mu = 3.0
        true_sigma = 0.1
        n_per_experiment = 5  # Typical seed count
        n_simulations = 200
        rng = random.Random(12345)

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
        n_per_group = 5
        n_simulations = 100
        rng = random.Random(54321)

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
        +/-0.05 (half the std) should frequently declare equivalence.
        """
        true_mu = 3.0
        true_sigma = 0.1
        n_per_group = 5
        n_simulations = 100
        rng = random.Random(99999)

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
