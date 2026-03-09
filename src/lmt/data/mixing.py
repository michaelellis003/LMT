"""Data mixing for multi-domain training.

Combines multiple TokenDatasets with controllable sampling weights,
enabling training on mixed domains (code, math, text) with specified
proportions.

Usage::

    from lmt.data.mixing import MixedDataset
    from torch.utils.data import DataLoader

    mixed = MixedDataset(
        datasets=[code_ds, math_ds, text_ds],
        weights=[0.5, 0.3, 0.2],
    )

    # Weighted random sampling
    sampler = mixed.weighted_sampler(num_samples=10000)
    loader = DataLoader(mixed, batch_size=8, sampler=sampler)

    # Or simple sequential access
    loader = DataLoader(mixed, batch_size=8, shuffle=True)
"""

from __future__ import annotations

import random as stdlib_random

import torch
from torch.utils.data import Dataset, Sampler

from lmt.data.token_dataset import TokenDataset


class MixedDataset(Dataset):
    """Concatenates multiple TokenDatasets with weighted sampling.

    Each dataset is indexed into a global index space. The
    ``weighted_sampler`` method creates a sampler that draws
    from each domain according to the specified weights.

    Args:
        datasets: List of TokenDatasets to mix.
        weights: Optional sampling weights per dataset.
            Defaults to uniform. Normalized internally.
    """

    def __init__(
        self,
        datasets: list[TokenDataset],
        weights: list[float] | None = None,
    ) -> None:
        """Initialize mixed dataset.

        Args:
            datasets: List of TokenDatasets to combine.
            weights: Sampling weights per dataset. If None,
                uses uniform weights.

        Raises:
            ValueError: If datasets have different context lengths.
        """
        if not datasets:
            raise ValueError('At least one dataset required')

        # Validate same context length
        ctx_len = datasets[0].context_length
        for ds in datasets[1:]:
            if ds.context_length != ctx_len:
                raise ValueError(
                    f'All datasets must have the same context '
                    f'length. Got {ctx_len} and '
                    f'{ds.context_length}'
                )

        self.datasets = datasets
        self._offsets: list[int] = []
        total = 0
        for ds in datasets:
            self._offsets.append(total)
            total += len(ds)
        self._total_len = total

        if weights is None:
            weights = [1.0] * len(datasets)
        total_w = sum(weights)
        self.weights = [w / total_w for w in weights]

    def __len__(self) -> int:
        """Return total number of samples across all datasets."""
        return self._total_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample by global index.

        Args:
            idx: Global index across all datasets.

        Returns:
            Token sequence from the appropriate dataset.
        """
        if idx < 0:
            idx += self._total_len
        # Find which dataset this index belongs to
        for i in range(len(self.datasets) - 1, -1, -1):
            if idx >= self._offsets[i]:
                local_idx = idx - self._offsets[i]
                return self.datasets[i][local_idx]
        raise IndexError(f'Index {idx} out of range')

    def weighted_sampler(
        self,
        num_samples: int,
        seed: int | None = None,
    ) -> WeightedDomainSampler:
        """Create a sampler that respects domain weights.

        Args:
            num_samples: Number of samples to draw.
            seed: Random seed for reproducibility.

        Returns:
            A Sampler that draws from domains proportionally.
        """
        return WeightedDomainSampler(
            dataset=self,
            num_samples=num_samples,
            seed=seed,
        )


class WeightedDomainSampler(Sampler):
    """Samples indices according to domain weights.

    Draws from each domain proportionally to its weight,
    producing indices into the MixedDataset's global index space.

    Args:
        dataset: The MixedDataset to sample from.
        num_samples: Total number of indices to yield.
        seed: Optional random seed.
    """

    def __init__(
        self,
        dataset: MixedDataset,
        num_samples: int,
        seed: int | None = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            dataset: The MixedDataset to sample from.
            num_samples: Total number of indices to yield.
            seed: Optional random seed.
        """
        self.dataset = dataset
        self._num_samples = num_samples
        self.seed = seed

    def __iter__(self):
        """Yield sampled indices."""
        rng = stdlib_random.Random(self.seed)
        ds = self.dataset

        for _ in range(self._num_samples):
            # Pick a domain based on weights
            domain_idx = _weighted_choice(ds.weights, rng)
            # Pick a random sample within that domain
            domain_len = len(ds.datasets[domain_idx])
            local_idx = rng.randint(0, domain_len - 1)
            global_idx = ds._offsets[domain_idx] + local_idx
            yield global_idx

    def __len__(self) -> int:
        """Return number of samples."""
        return self._num_samples


def _weighted_choice(
    weights: list[float],
    rng: stdlib_random.Random,
) -> int:
    """Choose an index proportional to weights.

    Args:
        weights: Normalized probability weights.
        rng: Random number generator.

    Returns:
        Selected index.
    """
    r = rng.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r < cumulative:
            return i
    return len(weights) - 1
