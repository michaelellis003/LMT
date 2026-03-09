"""Multi-seed experiment runner for statistically rigorous comparisons.

Runs the same experiment configuration across multiple random seeds,
collects per-seed metrics, and integrates with the Bayesian comparison
framework to determine statistical significance.

The key insight: a single-seed result tells you nothing about whether
a difference is real or just due to random initialization. With 5+
seeds, you can compute posterior probabilities like P(A < B) = 0.95.

Usage::

    from lmt.research.multi_seed import (
        MultiSeedConfig,
        run_multi_seed,
    )
    from lmt.research.stats import bayesian_compare

    config = MultiSeedConfig(n_seeds=5)

    result_a = run_multi_seed('llama', train_llama, config)
    result_b = run_multi_seed('gpt', train_gpt, config)

    comparison = bayesian_compare(
        result_a.to_samples(),
        result_b.to_samples(),
    )
    print(comparison.interpretation)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from lmt.research.stats import ExperimentSamples

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
