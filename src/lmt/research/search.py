"""Hyperparameter search for the autoresearch framework.

Provides grid search and random search over experiment
configurations. Each trial creates a fresh model, trains it,
and evaluates BPB, making it easy to find optimal hyperparameters.

Usage::

    from lmt.research.search import SearchSpace, grid_search

    space = SearchSpace(lr=[1e-3, 1e-4], batch_size=[4, 8])
    results = grid_search(
        model_fn=lambda: MyModel(config),
        train_data=train_data,
        search_space=space,
        base_config=ExperimentRunConfig(train_steps=100),
        output_dir='runs/grid',
        val_data=val_data,
    )
    best = min(results, key=lambda r: r.metrics.get_best('val_bpb'))
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn

from lmt.research.experiment import (
    ExperimentResult,
    ExperimentRunConfig,
    ExperimentRunner,
)


class SearchSpace:
    """Defines a hyperparameter search space.

    Each keyword argument maps a parameter name to a list of
    values to try. The parameter names must match fields in
    ``ExperimentRunConfig``.

    Args:
        **params: Parameter names mapped to lists of values.
    """

    def __init__(self, **params: list[Any]) -> None:
        """Initialize search space.

        Args:
            **params: Parameter names to lists of candidate values.
        """
        self.params = params

    def grid(self) -> list[dict[str, Any]]:
        """Generate all combinations (Cartesian product).

        Returns:
            List of dicts, each mapping param names to values.
        """
        keys = list(self.params.keys())
        values = list(self.params.values())
        return [
            dict(zip(keys, combo, strict=True))
            for combo in itertools.product(*values)
        ]

    def sample(
        self,
        n: int,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Sample N random configurations.

        Args:
            n: Number of random configs to sample.
            seed: Optional random seed for reproducibility.

        Returns:
            List of N dicts with randomly sampled values.
        """
        rng = random.Random(seed)
        configs = []
        for _ in range(n):
            cfg = {k: rng.choice(v) for k, v in self.params.items()}
            configs.append(cfg)
        return configs

    def to_experiment_configs(
        self,
        base_config: ExperimentRunConfig,
    ) -> list[ExperimentRunConfig]:
        """Convert grid to ExperimentRunConfig objects.

        Each config gets a unique name based on its parameters.

        Args:
            base_config: Base config to override with search values.

        Returns:
            List of ExperimentRunConfig with unique names.
        """
        configs = []
        for param_dict in self.grid():
            name_parts = [f'{k}={v}' for k, v in sorted(param_dict.items())]
            name = f'{base_config.name}_{"_".join(name_parts)}'
            cfg = replace(base_config, name=name, **param_dict)
            configs.append(cfg)
        return configs


def grid_search(
    model_fn: Callable[[], nn.Module],
    train_data: torch.Tensor,
    search_space: SearchSpace,
    base_config: ExperimentRunConfig,
    output_dir: str = 'runs/grid',
    val_data: torch.Tensor | None = None,
) -> list[ExperimentResult]:
    """Run grid search over hyperparameters.

    For each combination in the search space, creates a fresh
    model and runs a training experiment.

    Args:
        model_fn: Factory function that creates a fresh model.
        train_data: Training data ``[num_samples, seq_len]``.
        search_space: Hyperparameter search space.
        base_config: Base experiment config (overridden by search).
        output_dir: Directory for saving results.
        val_data: Optional validation data for BPB.

    Returns:
        List of ExperimentResult, one per grid combination.
    """
    configs = search_space.to_experiment_configs(base_config)
    runner = ExperimentRunner(output_dir=output_dir)

    results = []
    for cfg in configs:
        model = model_fn()
        result = runner.run(model, train_data, cfg, val_data=val_data)
        results.append(result)

    return results


def random_search(
    model_fn: Callable[[], nn.Module],
    train_data: torch.Tensor,
    search_space: SearchSpace,
    n_trials: int,
    base_config: ExperimentRunConfig,
    output_dir: str = 'runs/random',
    val_data: torch.Tensor | None = None,
    seed: int | None = None,
) -> list[ExperimentResult]:
    """Run random search over hyperparameters.

    Samples N random configurations from the search space and
    runs a training experiment for each.

    Args:
        model_fn: Factory function that creates a fresh model.
        train_data: Training data ``[num_samples, seq_len]``.
        search_space: Hyperparameter search space.
        n_trials: Number of random trials to run.
        base_config: Base experiment config.
        output_dir: Directory for saving results.
        val_data: Optional validation data for BPB.
        seed: Random seed for reproducibility.

    Returns:
        List of ExperimentResult, one per trial.
    """
    sampled = search_space.sample(n=n_trials, seed=seed)
    runner = ExperimentRunner(output_dir=output_dir)

    results = []
    for i, param_dict in enumerate(sampled):
        name_parts = [f'{k}={v}' for k, v in sorted(param_dict.items())]
        name = f'{base_config.name}_trial{i}_{"_".join(name_parts)}'
        cfg = replace(base_config, name=name, **param_dict)
        model = model_fn()
        result = runner.run(model, train_data, cfg, val_data=val_data)
        results.append(result)

    return results
