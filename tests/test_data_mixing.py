"""Tests for data mixing utility."""

import torch
from torch.utils.data import DataLoader

from lmt.data.mixing import MixedDataset
from lmt.data.token_dataset import TokenDataset


def _make_dataset(value: int, n_tokens: int, ctx_len: int) -> TokenDataset:
    """Create a TokenDataset filled with a specific value."""
    tokens = torch.full((n_tokens,), value, dtype=torch.long)
    return TokenDataset(tokens, ctx_len)


class TestMixedDataset:
    """Test weighted mixing of multiple datasets."""

    def test_basic_mixing(self):
        """Should sample from multiple datasets."""
        ds_a = _make_dataset(1, 100, 4)
        ds_b = _make_dataset(2, 100, 4)

        mixed = MixedDataset(
            datasets=[ds_a, ds_b],
            weights=[0.5, 0.5],
        )

        assert len(mixed) == len(ds_a) + len(ds_b)
        sample = mixed[0]
        assert sample.shape == (5,)  # ctx_len + 1

    def test_weighted_sampling(self):
        """Should respect weights when sampling with DataLoader."""
        ds_a = _make_dataset(1, 50, 4)
        ds_b = _make_dataset(2, 50, 4)

        mixed = MixedDataset(
            datasets=[ds_a, ds_b],
            weights=[0.8, 0.2],
        )

        # Use the sampler for weighted random access
        sampler = mixed.weighted_sampler(num_samples=100, seed=42)
        loader = DataLoader(mixed, batch_size=10, sampler=sampler)

        # Count samples from each dataset
        total_a = 0
        total_b = 0
        for batch in loader:
            total_a += (batch[:, 0] == 1).sum().item()
            total_b += (batch[:, 0] == 2).sum().item()

        # With 0.8/0.2 weights, expect ~80% from A
        ratio_a = total_a / (total_a + total_b)
        assert 0.6 < ratio_a < 0.95  # Loose bounds for randomness

    def test_uniform_weights(self):
        """Should default to uniform weights."""
        ds_a = _make_dataset(1, 50, 4)
        ds_b = _make_dataset(2, 50, 4)

        mixed = MixedDataset(datasets=[ds_a, ds_b])

        assert len(mixed) == len(ds_a) + len(ds_b)

    def test_single_dataset(self):
        """Should work with a single dataset."""
        ds = _make_dataset(1, 50, 4)
        mixed = MixedDataset(datasets=[ds], weights=[1.0])
        assert len(mixed) == len(ds)

    def test_mismatched_context_lengths(self):
        """Should raise if datasets have different shapes."""
        ds_a = _make_dataset(1, 50, 4)
        ds_b = _make_dataset(2, 50, 8)

        import pytest

        with pytest.raises(ValueError, match='context'):
            MixedDataset(datasets=[ds_a, ds_b])

    def test_three_domains(self):
        """Should handle three or more domains."""
        ds_code = _make_dataset(1, 60, 4)
        ds_math = _make_dataset(2, 40, 4)
        ds_text = _make_dataset(3, 100, 4)

        mixed = MixedDataset(
            datasets=[ds_code, ds_math, ds_text],
            weights=[0.5, 0.3, 0.2],
        )

        total = len(ds_code) + len(ds_math) + len(ds_text)
        assert len(mixed) == total

    def test_works_with_dataloader(self):
        """Should work with standard DataLoader."""
        ds_a = _make_dataset(1, 50, 4)
        ds_b = _make_dataset(2, 50, 4)

        mixed = MixedDataset(
            datasets=[ds_a, ds_b],
            weights=[0.5, 0.5],
        )

        loader = DataLoader(mixed, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        assert batch.shape == (4, 5)
