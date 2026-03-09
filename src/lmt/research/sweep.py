"""Hyperparameter sweep utility for automated experiment search.

Generates hyperparameter configurations using grid search or random
search, for use with the autoresearch experiment pipeline.

Usage::

    from lmt.research.sweep import SweepConfig, generate_grid

    config = SweepConfig(
        parameters={
            'lr': [1e-3, 1e-4, 1e-5],
            'batch_size': [16, 32, 64],
        },
        base_config={'warmup_steps': 50, 'total_steps': 500},
    )

    # Grid search: all combinations
    for hp_config in generate_grid(config):
        print(hp_config)
        # {'lr': 1e-3, 'batch_size': 16, 'warmup_steps': 50, ...}

    # Random search: sample 10 configs
    for hp_config in generate_random(config, n=10, seed=42):
        run_experiment(hp_config)
"""

from __future__ import annotations

import itertools
import random as random_module
from dataclasses import dataclass, field
from functools import reduce
from typing import Any


@dataclass
class SweepConfig:
    """Defines a hyperparameter search space.

    Each parameter maps to a list of values to try. Grid search
    produces the full Cartesian product; random search samples
    from the options.

    Args:
        parameters: Dict mapping parameter names to lists of values.
        base_config: Fixed parameters merged into every configuration.
    """

    parameters: dict[str, list[Any]]
    base_config: dict[str, Any] = field(default_factory=dict)

    @property
    def grid_size(self) -> int:
        """Total number of configurations in full grid search."""
        if not self.parameters:
            return 0
        return reduce(
            lambda a, b: a * b,
            (len(v) for v in self.parameters.values()),
        )


def generate_grid(config: SweepConfig) -> list[dict[str, Any]]:
    """Generate all configurations via Cartesian product.

    Args:
        config: Sweep configuration with parameter options.

    Returns:
        List of config dicts, one per grid point. Each dict
        contains the swept parameters merged with base_config.
    """
    if not config.parameters:
        return []

    keys = list(config.parameters.keys())
    value_lists = [config.parameters[k] for k in keys]

    configs: list[dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        d = dict(config.base_config)
        d.update(dict(zip(keys, combo, strict=True)))
        configs.append(d)

    return configs


def generate_random(
    config: SweepConfig,
    n: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate random configurations by sampling from options.

    Args:
        config: Sweep configuration with parameter options.
        n: Number of configurations to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of n config dicts, each with randomly sampled
        parameter values merged with base_config.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError(f'n must be >= 0, got {n}')
    if n == 0 or not config.parameters:
        return []

    rng = random_module.Random(seed)
    keys = list(config.parameters.keys())

    configs: list[dict[str, Any]] = []
    for _ in range(n):
        d = dict(config.base_config)
        for key in keys:
            d[key] = rng.choice(config.parameters[key])
        configs.append(d)

    return configs
