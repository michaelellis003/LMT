"""Bayesian statistical comparison framework for experiments.

Provides tools for rigorous comparison of experiment results across
multiple random seeds. Uses Bayesian inference with conjugate priors
to compute credible intervals and posterior probabilities.

The key insight for toy-scale ML experiments: with only 5-10 seeds,
frequentist tests (t-tests, p-values) have very low power. Bayesian
methods work better because they:

1. Naturally handle small sample sizes via the posterior distribution
2. Give direct probability statements ("P(A < B) = 0.93")
3. Produce credible intervals (not confidence intervals)
4. Can incorporate prior knowledge

Mathematical foundation:

For N samples x_1, ..., x_N with unknown mean mu and variance sigma^2,
using the conjugate Normal-InverseGamma prior (non-informative):

    posterior for mu | x ~ Student-t(df=N-1, loc=x_bar, scale=s/sqrt(N))

This is the standard Bayesian result for a Normal likelihood with
unknown mean and variance using Jeffreys prior.

For comparing two groups A and B, we compute:

    P(mu_A < mu_B) = integral over posterior samples

using Monte Carlo sampling from both posteriors.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class ExperimentSamples:
    """Collection of metric values from multiple seeds.

    Attributes:
        name: Experiment or variant name.
        values: List of metric values (one per seed).
    """

    name: str
    values: list[float]

    @property
    def n(self) -> int:
        """Number of samples (seeds)."""
        return len(self.values)

    @property
    def mean(self) -> float:
        """Sample mean."""
        return sum(self.values) / self.n

    @property
    def std(self) -> float:
        """Sample standard deviation (Bessel-corrected)."""
        if self.n < 2:
            return 0.0
        m = self.mean
        return math.sqrt(sum((x - m) ** 2 for x in self.values) / (self.n - 1))

    @property
    def sem(self) -> float:
        """Standard error of the mean."""
        if self.n < 2:
            return 0.0
        return self.std / math.sqrt(self.n)


@dataclass
class BayesianComparison:
    """Result of a Bayesian comparison between two experiments.

    Attributes:
        name_a: Name of first experiment.
        name_b: Name of second experiment.
        mean_a: Mean of first experiment.
        mean_b: Mean of second experiment.
        mean_difference: Mean of (A - B). Negative means A is better
            (lower BPB).
        prob_a_better: P(mu_A < mu_B) -- probability that A has lower
            BPB than B.
        ci_lower: Lower bound of 95% credible interval for (A - B).
        ci_upper: Upper bound of 95% credible interval for (A - B).
        cohens_d: Cohen's d effect size (positive = A is better).
        n_a: Number of seeds for A.
        n_b: Number of seeds for B.
    """

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    mean_difference: float
    prob_a_better: float
    ci_lower: float
    ci_upper: float
    cohens_d: float
    n_a: int
    n_b: int
    interpretation: str = field(default='')

    def __post_init__(self) -> None:
        """Generate human-readable interpretation."""
        if not self.interpretation:
            self.interpretation = _interpret(self)


def credible_interval(
    values: list[float], level: float = 0.95
) -> tuple[float, float]:
    """Compute credible interval using the posterior t-distribution.

    For small samples, we use the Student-t posterior which naturally
    accounts for uncertainty in both the mean and variance.

    For N samples, the posterior for the mean is:
        mu | data ~ t(df=N-1, loc=x_bar, scale=s/sqrt(N))

    We approximate the CI using Monte Carlo sampling from this
    posterior.

    Args:
        values: Observed metric values.
        level: Credible level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    n = len(values)
    if n < 2:
        v = values[0] if values else 0.0
        return (v, v)

    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    sem = std / math.sqrt(n)

    # Sample from Student-t posterior using the standard trick:
    # t = Z / sqrt(V/df) where Z ~ N(0,1), V ~ Chi2(df)
    # Then mu_posterior = mean + sem * t
    n_samples = 10000
    rng = random.Random(42)  # Deterministic for reproducibility
    df = n - 1

    samples = []
    for _ in range(n_samples):
        # Generate t-distributed sample via ratio of normal/chi2
        z = _standard_normal(rng)
        # Chi2(df) = sum of df standard normals squared
        chi2 = sum(_standard_normal(rng) ** 2 for _ in range(df))
        if chi2 < 1e-10:
            continue
        t_sample = z / math.sqrt(chi2 / df)
        samples.append(mean + sem * t_sample)

    samples.sort()
    alpha = 1 - level
    lo_idx = int(alpha / 2 * len(samples))
    hi_idx = int((1 - alpha / 2) * len(samples))
    lo_idx = max(0, min(lo_idx, len(samples) - 1))
    hi_idx = max(0, min(hi_idx, len(samples) - 1))

    return (samples[lo_idx], samples[hi_idx])


def cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size.

    Positive d means group A has lower values (better for BPB).
    Uses pooled standard deviation.

    Interpretation thresholds (Cohen, 1988):
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Args:
        a: Values from group A.
        b: Values from group B.

    Returns:
        Cohen's d effect size.
    """
    n_a = len(a)
    n_b = len(b)
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b

    if n_a < 2 or n_b < 2:
        return 0.0

    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = math.sqrt(pooled_var)

    if pooled_std < 1e-10:
        return 0.0

    # Positive d = A is lower (better for BPB)
    return (mean_b - mean_a) / pooled_std


def bayesian_compare(
    a: ExperimentSamples,
    b: ExperimentSamples,
    n_mc_samples: int = 50000,
) -> BayesianComparison:
    """Compare two experiments using Bayesian posterior sampling.

    Samples from the posterior t-distributions for each group's mean,
    then computes P(mu_A < mu_B) by Monte Carlo.

    Args:
        a: Samples from experiment A.
        b: Samples from experiment B.
        n_mc_samples: Number of Monte Carlo samples.

    Returns:
        BayesianComparison with posterior probabilities and intervals.
    """
    rng = random.Random(42)

    # Sample from posterior of A
    samples_a = _posterior_samples(a.values, n_mc_samples, rng)
    # Sample from posterior of B
    samples_b = _posterior_samples(b.values, n_mc_samples, rng)

    # Compute difference distribution: A - B
    diffs = [sa - sb for sa, sb in zip(samples_a, samples_b, strict=True)]

    # P(A < B) = P(diff < 0)
    prob_a_better = sum(1 for d in diffs if d < 0) / len(diffs)

    # Credible interval for the difference
    diffs.sort()
    lo_idx = int(0.025 * len(diffs))
    hi_idx = int(0.975 * len(diffs))
    ci_lower = diffs[lo_idx]
    ci_upper = diffs[hi_idx]

    mean_diff = sum(diffs) / len(diffs)

    return BayesianComparison(
        name_a=a.name,
        name_b=b.name,
        mean_a=a.mean,
        mean_b=b.mean,
        mean_difference=mean_diff,
        prob_a_better=prob_a_better,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        cohens_d=cohens_d(a.values, b.values),
        n_a=a.n,
        n_b=b.n,
    )


def multi_seed_summary(
    samples: ExperimentSamples,
) -> dict[str, float | str | int]:
    """Generate summary statistics for a multi-seed experiment.

    Args:
        samples: Experiment samples from multiple seeds.

    Returns:
        Dictionary with name, n_seeds, mean, std, sem, ci_lower,
        ci_upper.
    """
    ci_lo, ci_hi = credible_interval(samples.values)
    return {
        'name': samples.name,
        'n_seeds': samples.n,
        'mean': samples.mean,
        'std': samples.std,
        'sem': samples.sem,
        'ci_lower': ci_lo,
        'ci_upper': ci_hi,
    }


def _standard_normal(rng: random.Random) -> float:
    """Generate a standard normal sample using Box-Muller."""
    u1 = rng.random()
    u2 = rng.random()
    # Avoid log(0)
    u1 = max(u1, 1e-10)
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _posterior_samples(
    values: list[float],
    n_samples: int,
    rng: random.Random,
) -> list[float]:
    """Sample from the posterior distribution of the mean.

    Uses Student-t posterior: mu | data ~ t(N-1, x_bar, s/sqrt(N))
    """
    n = len(values)
    mean = sum(values) / n

    if n < 2:
        return [mean] * n_samples

    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    sem = std / math.sqrt(n)
    df = n - 1

    samples = []
    for _ in range(n_samples):
        z = _standard_normal(rng)
        chi2 = sum(_standard_normal(rng) ** 2 for _ in range(df))
        if chi2 < 1e-10:
            samples.append(mean)
            continue
        t_sample = z / math.sqrt(chi2 / df)
        samples.append(mean + sem * t_sample)

    return samples


def _interpret(result: BayesianComparison) -> str:
    """Generate human-readable interpretation of comparison."""
    p = result.prob_a_better
    d = abs(result.cohens_d)
    a = result.name_a
    b = result.name_b

    # Effect size interpretation
    if d < 0.2:
        size_str = 'negligible'
    elif d < 0.5:
        size_str = 'small'
    elif d < 0.8:
        size_str = 'medium'
    else:
        size_str = 'large'

    # Probability interpretation
    if p > 0.95:
        confidence = 'strong evidence'
        winner = a
    elif p > 0.8:
        confidence = 'moderate evidence'
        winner = a
    elif p < 0.05:
        confidence = 'strong evidence'
        winner = b
    elif p < 0.2:
        confidence = 'moderate evidence'
        winner = b
    else:
        return (
            f'No clear winner ({a} vs {b}). '
            f'P({a} better) = {p:.2f}, '
            f'effect size: {size_str} (d={result.cohens_d:.2f}). '
            f'Need more seeds to distinguish.'
        )

    return (
        f'{confidence} that {winner} is better. '
        f'P({a} better) = {p:.2f}, '
        f'effect size: {size_str} (d={result.cohens_d:.2f}). '
        f'Mean diff: {result.mean_difference:.4f} '
        f'[{result.ci_lower:.4f}, {result.ci_upper:.4f}].'
    )
