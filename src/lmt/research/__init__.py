"""Automated research and experiment management."""

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
    'bayesian_compare',
    'cohens_d',
    'credible_interval',
    'multi_seed_summary',
]
