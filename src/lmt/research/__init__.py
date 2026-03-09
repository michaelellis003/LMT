"""Automated research and experiment management."""

from lmt.research.multi_seed import (
    MultiSeedConfig,
    MultiSeedResult,
    SeedResult,
    compare_variants,
    run_multi_seed,
    screen_then_confirm,
)
from lmt.research.stats import (
    BayesianComparison,
    ExperimentSamples,
    bayesian_compare,
    cohens_d,
    credible_interval,
    multi_seed_summary,
)

__all__ = [
    'BayesianComparison',
    'ExperimentSamples',
    'MultiSeedConfig',
    'MultiSeedResult',
    'SeedResult',
    'bayesian_compare',
    'cohens_d',
    'compare_variants',
    'credible_interval',
    'multi_seed_summary',
    'run_multi_seed',
    'screen_then_confirm',
]
