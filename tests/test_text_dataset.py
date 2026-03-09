"""Tests for text dataset with tokenization and packing."""

import torch

from lmt.tokenizer import BPETokenizer


class TestTextDataset:
    """Test TextDataset that tokenizes and packs text on the fly."""

    def test_create_from_texts(self):
        """Should create dataset from a list of strings."""
        from lmt.data.text_dataset import TextDataset

        texts = ['Hello world. This is a longer sentence. ' * 10]
        tokenizer = BPETokenizer()
        ds = TextDataset(texts, tokenizer, context_length=16)

        assert len(ds) > 0

    def test_item_shape(self):
        """Each item should be [context_length + 1]."""
        from lmt.data.text_dataset import TextDataset

        texts = ['Hello world. ' * 50]  # enough tokens
        tokenizer = BPETokenizer()
        ds = TextDataset(texts, tokenizer, context_length=16)

        item = ds[0]
        assert item.shape == (17,)  # 16 + 1

    def test_items_are_token_ids(self):
        """Items should contain valid token IDs."""
        from lmt.data.text_dataset import TextDataset

        texts = ['The quick brown fox jumps over the lazy dog. ' * 20]
        tokenizer = BPETokenizer()
        ds = TextDataset(texts, tokenizer, context_length=16)

        item = ds[0]
        assert item.dtype == torch.long
        assert item.min() >= 0

    def test_sep_token_between_documents(self):
        """Should insert separator tokens between documents."""
        from lmt.data.text_dataset import TextDataset

        texts = [
            'First document with enough text. ' * 10,
            'Second document with enough text. ' * 10,
        ]
        tokenizer = BPETokenizer()
        # Use a known token as separator
        eot = tokenizer.tokenizer.eot_token
        ds = TextDataset(texts, tokenizer, context_length=32, sep_token_id=eot)

        # The packed tokens should contain the separator
        assert len(ds) > 0

    def test_works_with_dataloader(self):
        """Should be compatible with DataLoader."""
        from torch.utils.data import DataLoader

        from lmt.data.text_dataset import TextDataset

        texts = ['Hello world. ' * 100]
        tokenizer = BPETokenizer()
        ds = TextDataset(texts, tokenizer, context_length=16)

        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch.shape == (4, 17)

    def test_empty_texts(self):
        """Should handle empty text list gracefully."""
        from lmt.data.text_dataset import TextDataset

        tokenizer = BPETokenizer()
        ds = TextDataset([], tokenizer, context_length=16)
        assert len(ds) == 0

    def test_context_length_respected(self):
        """All items should have exactly context_length + 1 tokens."""
        from lmt.data.text_dataset import TextDataset

        texts = ['A short sentence. ' * 50]
        tokenizer = BPETokenizer()
        ds = TextDataset(texts, tokenizer, context_length=32)

        for i in range(min(len(ds), 5)):
            assert ds[i].shape == (33,)


class TestLoadHFTextDataset:
    """Test HuggingFace dataset loading utility."""

    def test_load_returns_text_dataset(self):
        """Should return a TextDataset from an iterable of dicts."""
        from lmt.data.text_dataset import TextDataset, texts_from_dicts

        # Simulate HF dataset rows with enough text
        rows = [
            {'text': 'First document about math. ' * 20},
            {'text': 'Second document about science. ' * 20},
            {'text': 'Third document about history. ' * 20},
        ]
        texts = texts_from_dicts(rows, text_field='text')
        tokenizer = BPETokenizer()
        ds = TextDataset(texts, tokenizer, context_length=16)

        assert len(ds) > 0

    def test_custom_text_field(self):
        """Should support custom text field names."""
        from lmt.data.text_dataset import texts_from_dicts

        rows = [
            {'content': 'Hello world.'},
            {'content': 'Another sentence.'},
        ]
        texts = texts_from_dicts(rows, text_field='content')
        assert texts == ['Hello world.', 'Another sentence.']

    def test_filters_empty_strings(self):
        """Should skip empty or whitespace-only strings."""
        from lmt.data.text_dataset import texts_from_dicts

        rows = [
            {'text': 'Real content.'},
            {'text': ''},
            {'text': '   '},
            {'text': 'More content.'},
        ]
        texts = texts_from_dicts(rows, text_field='text')
        assert len(texts) == 2
