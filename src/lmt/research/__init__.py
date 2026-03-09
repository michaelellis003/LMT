"""Automated research and experiment management."""

from lmt.research.multi_seed import (
    MultiSeedConfig,
    MultiSeedResult,
    SeedResult,
    run_multi_seed,
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
    'credible_interval',
    'multi_seed_summary',
    'run_multi_seed',
]
