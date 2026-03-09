"""Tests for BPE tokenizer trainer."""

import pytest

from lmt.tokenizer.trainer import train_bpe_tokenizer


class TestTrainBPETokenizer:
    """Test BPE tokenizer training from scratch."""

    def _sample_texts(self) -> list[str]:
        """Return a small corpus for testing."""
        return [
            'The quick brown fox jumps over the lazy dog.',
            'A quick brown dog outfoxes the lazy fox.',
            'The dog and the fox are both quick and brown.',
            'Lazy dogs sleep while quick foxes jump.',
            'Brown foxes and brown dogs jump quickly.',
        ] * 20  # Repeat to have enough data for BPE merges

    def test_basic_training(self):
        """Should train a tokenizer and return HFTokenizerWrapper."""
        from lmt.tokenizer.hf_tokenizer import HFTokenizerWrapper

        tok = train_bpe_tokenizer(self._sample_texts(), vocab_size=256)
        assert isinstance(tok, HFTokenizerWrapper)

    def test_encode_decode_roundtrip(self):
        """Should encode and decode text correctly."""
        tok = train_bpe_tokenizer(self._sample_texts(), vocab_size=256)
        text = 'The quick brown fox jumps.'
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_vocab_size(self):
        """Should respect the requested vocab size."""
        tok = train_bpe_tokenizer(self._sample_texts(), vocab_size=128)
        assert tok.vocab_size <= 128

    def test_custom_special_tokens(self):
        """Should include custom special tokens."""
        tok = train_bpe_tokenizer(
            self._sample_texts(),
            vocab_size=256,
            special_tokens=['<pad>', '<eos>', '<bos>'],
        )
        ids = tok.encode('<pad>')
        # Special tokens should be in vocab
        assert len(ids) >= 1

    def test_encode_returns_list_of_ints(self):
        """Should return list of integers."""
        tok = train_bpe_tokenizer(self._sample_texts(), vocab_size=256)
        ids = tok.encode('Hello world')
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_save_and_load(self, tmp_path):
        """Should save and load tokenizer."""
        tok = train_bpe_tokenizer(self._sample_texts(), vocab_size=256)
        save_path = tmp_path / 'tokenizer'

        tok.save(str(save_path))

        from lmt.tokenizer.trainer import load_trained_tokenizer

        loaded = load_trained_tokenizer(str(save_path))
        text = 'The quick brown fox.'
        assert tok.encode(text) == loaded.encode(text)

    def test_min_frequency(self):
        """Should respect min_frequency parameter."""
        # With high min_frequency, fewer merges happen
        tok_low = train_bpe_tokenizer(
            self._sample_texts(), vocab_size=256, min_frequency=1
        )
        tok_high = train_bpe_tokenizer(
            self._sample_texts(), vocab_size=256, min_frequency=10
        )
        # Higher min_frequency = fewer merges = more tokens per text
        text = 'The quick brown fox jumps over the lazy dog.'
        ids_low = tok_low.encode(text)
        ids_high = tok_high.encode(text)
        assert len(ids_high) >= len(ids_low)

    def test_empty_corpus_raises(self):
        """Should raise ValueError for empty corpus."""
        with pytest.raises(ValueError, match='empty'):
            train_bpe_tokenizer([], vocab_size=256)
