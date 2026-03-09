"""Tests for streaming data pipeline."""

from torch.utils.data import DataLoader

from lmt.data.token_dataset import TokenDataset


class TestStreamingTokenize:
    """Test streaming tokenization from iterables."""

    def test_tokenize_and_pack_from_iterable(self):
        """Should tokenize text strings and pack into a TokenDataset."""
        from lmt.data.streaming import tokenize_and_pack

        texts = [
            'hello world',
            'this is a test',
            'streaming data',
        ]

        # Simple word-level tokenizer mock
        def tokenize(text: str) -> list[int]:
            return [hash(w) % 1000 for w in text.split()]

        dataset = tokenize_and_pack(
            texts=iter(texts),
            tokenize_fn=tokenize,
            context_length=4,
            sep_token_id=0,
        )

        assert isinstance(dataset, TokenDataset)
        assert len(dataset) > 0
        assert dataset[0].shape == (5,)  # context_length + 1

    def test_empty_iterable_returns_empty_dataset(self):
        """Empty input should return empty dataset."""
        from lmt.data.streaming import tokenize_and_pack

        dataset = tokenize_and_pack(
            texts=iter([]),
            tokenize_fn=lambda t: [1, 2, 3],
            context_length=4,
        )

        assert len(dataset) == 0

    def test_max_documents_limits_input(self):
        """Should stop after max_documents."""
        from lmt.data.streaming import tokenize_and_pack

        texts = [f'doc {i}' for i in range(100)]

        def tokenize(text: str) -> list[int]:
            return [1, 2, 3, 4, 5]  # 5 tokens per doc

        dataset = tokenize_and_pack(
            texts=iter(texts),
            tokenize_fn=tokenize,
            context_length=4,
            max_documents=10,
        )

        # 10 docs * 5 tokens = 50 tokens, 50 // 5 = 10 samples
        assert len(dataset) == 10

    def test_works_with_dataloader(self):
        """TokenDataset from streaming should work with DataLoader."""
        from lmt.data.streaming import tokenize_and_pack

        texts = [f'word{i} word{i + 1} word{i + 2}' for i in range(20)]

        def tokenize(text: str) -> list[int]:
            return [hash(w) % 100 for w in text.split()]

        dataset = tokenize_and_pack(
            texts=iter(texts),
            tokenize_fn=tokenize,
            context_length=2,
        )

        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch.shape == (4, 3)  # batch_size x (context_length + 1)


class TestStreamHFDataset:
    """Test loading from HuggingFace-style datasets."""

    def test_stream_from_dict_rows(self):
        """Should extract text from dict rows (HF dataset format)."""
        from lmt.data.streaming import tokenize_and_pack

        rows = [
            {'text': 'hello world foo bar baz'},
            {'text': 'another document here with words'},
        ]

        def tokenize(text: str) -> list[int]:
            return [hash(w) % 100 for w in text.split()]

        dataset = tokenize_and_pack(
            texts=(row['text'] for row in rows),
            tokenize_fn=tokenize,
            context_length=4,
        )

        assert len(dataset) > 0
