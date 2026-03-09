r"""Multi-seed experiment runner for statistically rigorous comparisons.

Runs the same experiment configuration across multiple random seeds,
collects per-seed metrics, and integrates with the Bayesian comparison
framework to determine statistical significance.

The key insight: a single-seed result tells you nothing about whether
a difference is real or just due to random initialization. With 5+
seeds, you can compute posterior probabilities like P(A < B) = 0.95.

Two-stage protocol (per Colas et al. 2018):
    Phase 1 -- Screening (5 seeds): quick keep/discard
    Phase 2 -- Confirmation (10 seeds): Bayesian comparison with ROPE

Usage::

    from lmt.research.multi_seed import (
        MultiSeedConfig,
        run_multi_seed,
        compare_variants,
    )

    config = MultiSeedConfig(n_seeds=10)

    result_a = run_multi_seed('llama', train_llama, config)
    result_b = run_multi_seed('gpt', train_gpt, config)

    comparison = compare_variants(result_a, result_b, rope=0.01)
    print(comparison.interpretation)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from lmt.research.stats import (
    BayesianComparison,
    ExperimentSamples,
    bayesian_compare,
)

# Default seeds chosen to be spread across the int range
# to minimize correlation between initializations
_DEFAULT_SEEDS = [42, 137, 256, 1337, 2024, 7, 99, 314, 512, 1024]


@dataclass
class MultiSeedConfig:
    """Configuration for multi-seed experiment runs.

    Attributes:
        n_seeds: Number of seeds to run.
        seeds: Explicit seed values. If not provided, uses first
            n_seeds from the default list.
    """

    n_seeds: int = 5
    seeds: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and fill in default seeds."""
        if not self.seeds:
            self.seeds = _DEFAULT_SEEDS[: self.n_seeds]
        if len(self.seeds) != self.n_seeds:
            msg = f'n_seeds={self.n_seeds} but got {len(self.seeds)} seeds'
            raise ValueError(msg)


@dataclass
class SeedResult:
    """Result from a single seed run.

    Attributes:
        seed: Random seed used.
        best_bpb: Best BPB achieved during training.
        elapsed: Wall-clock time in seconds.
    """

    seed: int
    best_bpb: float
    elapsed: float


@dataclass
class MultiSeedResult:
    """Aggregated results from running an experiment across seeds.

    Attributes:
        variant_name: Name of the experiment variant.
        seed_results: Per-seed results.
    """

    variant_name: str
    seed_results: list[SeedResult]

    @property
    def mean_bpb(self) -> float:
        """Mean BPB across seeds."""
        values = [r.best_bpb for r in self.seed_results]
        return sum(values) / len(values)

    @property
    def std_bpb(self) -> float:
        """Standard deviation of BPB across seeds (Bessel-corrected)."""
        values = [r.best_bpb for r in self.seed_results]
        n = len(values)
        if n < 2:
            return 0.0
        mean = self.mean_bpb
        return (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5

    def to_samples(self) -> ExperimentSamples:
        """Convert to ExperimentSamples for Bayesian comparison."""
        return ExperimentSamples(
            name=self.variant_name,
            values=[r.best_bpb for r in self.seed_results],
        )


def run_multi_seed(
    variant_name: str,
    train_fn: Callable[[int], float],
    config: MultiSeedConfig | None = None,
) -> MultiSeedResult:
    """Run an experiment across multiple seeds.

    The train_fn receives a seed and returns the best BPB achieved.
    This function handles timing and result collection.

    Args:
        variant_name: Name for this experiment variant.
        train_fn: Function that takes a seed (int) and returns
            the best BPB (float) achieved during training.
        config: Multi-seed configuration. Uses defaults if None.

    Returns:
        MultiSeedResult with per-seed results.
    """
    if config is None:
        config = MultiSeedConfig()

    seed_results: list[SeedResult] = []

    for i, seed in enumerate(config.seeds):
        print(f'  [{variant_name}] Seed {seed} ({i + 1}/{config.n_seeds})...')

        start = time.time()
        best_bpb = train_fn(seed)
        elapsed = time.time() - start

        seed_results.append(
            SeedResult(seed=seed, best_bpb=best_bpb, elapsed=elapsed)
        )

        print(
            f'  [{variant_name}] Seed {seed}: '
            f'BPB={best_bpb:.4f} ({elapsed:.1f}s)'
        )

    result = MultiSeedResult(
        variant_name=variant_name,
        seed_results=seed_results,
    )

    print(
        f'  [{variant_name}] Mean BPB: {result.mean_bpb:.4f} '
        f'± {result.std_bpb:.4f} (n={config.n_seeds})'
    )

    return result


def compare_variants(
    a: MultiSeedResult,
    b: MultiSeedResult,
    rope: float = 0.0,
    n_mc_samples: int = 50000,
) -> BayesianComparison:
    """Compare two multi-seed results using Bayesian inference.

    Convenience wrapper that converts MultiSeedResults to
    ExperimentSamples and runs bayesian_compare.

    Args:
        a: Results from variant A.
        b: Results from variant B.
        rope: Region of practical equivalence half-width.
        n_mc_samples: Monte Carlo samples for posterior.

    Returns:
        BayesianComparison with probabilities and intervals.
    """
    return bayesian_compare(
        a.to_samples(),
        b.to_samples(),
        n_mc_samples=n_mc_samples,
        rope=rope,
    )


def screen_then_confirm(
    name_a: str,
    name_b: str,
    train_fn_a: Callable[[int], float],
    train_fn_b: Callable[[int], float],
    rope: float = 0.01,
    screen_seeds: int = 5,
    confirm_seeds: int = 10,
) -> BayesianComparison | None:
    """Two-stage protocol: screen with few seeds, confirm if promising.

    Phase 1 (screening): Run both variants with screen_seeds seeds.
    If the mean difference is less than the pooled std, declare
    "no detectable difference" and return None (saves compute).

    Phase 2 (confirmation): Run additional seeds up to confirm_seeds
    total. Run full Bayesian comparison with ROPE.

    Based on Colas et al. (2018) power analysis recommendations.

    Args:
        name_a: Name of variant A.
        name_b: Name of variant B.
        train_fn_a: Training function for variant A.
        train_fn_b: Training function for variant B.
        rope: ROPE half-width for Bayesian comparison.
        screen_seeds: Seeds for Phase 1 screening.
        confirm_seeds: Total seeds for Phase 2 confirmation.

    Returns:
        BayesianComparison if the effect survived screening,
        None if the effect was too small to detect.
    """
    # Phase 1: Screening
    print(f'\n=== Phase 1: Screening ({screen_seeds} seeds) ===')
    screen_config = MultiSeedConfig(n_seeds=screen_seeds)
    result_a = run_multi_seed(name_a, train_fn_a, screen_config)
    result_b = run_multi_seed(name_b, train_fn_b, screen_config)

    # Check if effect is detectable above noise
    mean_diff = abs(result_a.mean_bpb - result_b.mean_bpb)
    pooled_std = ((result_a.std_bpb**2 + result_b.std_bpb**2) / 2) ** 0.5

    if pooled_std > 0 and mean_diff < pooled_std:
        print(
            f'\n  Screening: |diff|={mean_diff:.4f} < '
            f'pooled_std={pooled_std:.4f}'
        )
        print('  Effect too small to detect. Stopping.')
        return None

    # Phase 2: Confirmation with more seeds
    extra_seeds = confirm_seeds - screen_seeds
    if extra_seeds > 0:
        print(f'\n=== Phase 2: Confirmation (+{extra_seeds} seeds) ===')
        extra_config = MultiSeedConfig(
            n_seeds=extra_seeds,
            seeds=_DEFAULT_SEEDS[screen_seeds:confirm_seeds],
        )
        extra_a = run_multi_seed(name_a, train_fn_a, extra_config)
        extra_b = run_multi_seed(name_b, train_fn_b, extra_config)

        # Merge results
        all_a = MultiSeedResult(
            variant_name=name_a,
            seed_results=result_a.seed_results + extra_a.seed_results,
        )
        all_b = MultiSeedResult(
            variant_name=name_b,
            seed_results=result_b.seed_results + extra_b.seed_results,
        )
    else:
        all_a = result_a
        all_b = result_b

    comparison = compare_variants(all_a, all_b, rope=rope)
    print(f'\n  Result: {comparison.interpretation}')
    return comparison
