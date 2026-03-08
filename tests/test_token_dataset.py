"""Tests for token dataset with sequence packing.

Tests the TokenDataset class which provides efficient packed
sequences for language model pretraining.
"""

import torch


class TestTokenDataset:
    """Test the TokenDataset for pretraining data."""

    def test_create_from_tokens(self):
        """Should create dataset from a 1D tensor of token IDs."""
        from lmt.data.token_dataset import TokenDataset

        tokens = torch.randint(0, 100, (1000,))
        ds = TokenDataset(tokens, context_length=64)

        assert ds.context_length == 64
        assert len(ds) > 0

    def test_len_is_correct(self):
        """Length should be number of complete sequences."""
        from lmt.data.token_dataset import TokenDataset

        # 1000 tokens, context_length=64 → need 65 tokens per sample
        # (64 input + 1 target shift) → floor(1000 / 65) = 15
        tokens = torch.randint(0, 100, (1000,))
        ds = TokenDataset(tokens, context_length=64)

        # Each sample needs context_length + 1 tokens
        expected = 1000 // (64 + 1)
        assert len(ds) == expected

    def test_getitem_returns_correct_shape(self):
        """Each item should be [context_length + 1] for input/target."""
        from lmt.data.token_dataset import TokenDataset

        tokens = torch.randint(0, 100, (500,))
        ds = TokenDataset(tokens, context_length=32)

        item = ds[0]
        assert item.shape == (33,)  # 32 + 1

    def test_getitem_content_is_correct(self):
        """Items should be contiguous slices of the original tokens."""
        from lmt.data.token_dataset import TokenDataset

        tokens = torch.arange(100)
        ds = TokenDataset(tokens, context_length=10)

        # First item should be tokens 0..10
        item0 = ds[0]
        assert torch.equal(item0, torch.arange(11))

        # Second item should be tokens 11..21
        item1 = ds[1]
        assert torch.equal(item1, torch.arange(11, 22))

    def test_getitem_negative_index(self):
        """Should support negative indexing."""
        from lmt.data.token_dataset import TokenDataset

        tokens = torch.arange(100)
        ds = TokenDataset(tokens, context_length=10)

        last = ds[-1]
        also_last = ds[len(ds) - 1]
        assert torch.equal(last, also_last)

    def test_getitem_out_of_range(self):
        """Should raise IndexError for out-of-range indices."""
        from lmt.data.token_dataset import TokenDataset

        tokens = torch.arange(100)
        ds = TokenDataset(tokens, context_length=10)

        import pytest

        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_works_with_dataloader(self):
        """Should be compatible with torch DataLoader."""
        from torch.utils.data import DataLoader

        from lmt.data.token_dataset import TokenDataset

        tokens = torch.randint(0, 100, (500,))
        ds = TokenDataset(tokens, context_length=32)

        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        assert batch.shape == (4, 33)

    def test_input_target_split(self):
        """Convenience: item[:-1] is input, item[1:] is target."""
        from lmt.data.token_dataset import TokenDataset

        tokens = torch.arange(100)
        ds = TokenDataset(tokens, context_length=10)

        item = ds[0]
        inputs = item[:-1]
        targets = item[1:]

        assert inputs.shape == (10,)
        assert targets.shape == (10,)
        # targets should be shifted by 1
        assert torch.equal(targets, inputs + 1)


class TestPackDocuments:
    """Test document packing utility."""

    def test_pack_single_document(self):
        """Single document should produce packed token tensor."""
        from lmt.data.token_dataset import pack_documents

        docs = [torch.tensor([1, 2, 3, 4, 5])]
        packed = pack_documents(docs)

        assert torch.equal(packed, torch.tensor([1, 2, 3, 4, 5]))

    def test_pack_multiple_documents(self):
        """Multiple documents concatenated with separator."""
        from lmt.data.token_dataset import pack_documents

        docs = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
        ]
        packed = pack_documents(docs, sep_token_id=0)

        # Should be [1, 2, 3, 0, 4, 5, 6]
        assert torch.equal(packed, torch.tensor([1, 2, 3, 0, 4, 5, 6]))

    def test_pack_no_separator(self):
        """Without separator, documents are just concatenated."""
        from lmt.data.token_dataset import pack_documents

        docs = [
            torch.tensor([1, 2]),
            torch.tensor([3, 4]),
        ]
        packed = pack_documents(docs, sep_token_id=None)

        assert torch.equal(packed, torch.tensor([1, 2, 3, 4]))

    def test_pack_empty_list(self):
        """Empty list should return empty tensor."""
        from lmt.data.token_dataset import pack_documents

        packed = pack_documents([])
        assert packed.shape == (0,)

    def test_pack_preserves_dtype(self):
        """Should preserve the dtype of input tensors."""
        from lmt.data.token_dataset import pack_documents

        docs = [torch.tensor([1, 2, 3], dtype=torch.int32)]
        packed = pack_documents(docs)

        assert packed.dtype == torch.int32
