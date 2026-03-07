"""Tests for CharTokenizer -- character-level tokenizer."""

import pytest

from lmt.tokenizer.char import CharTokenizer


class TestCharTokenizer:
    """Test the character-level tokenizer."""

    def test_from_text_builds_vocab(self) -> None:
        """from_text should build a sorted vocabulary."""
        tok = CharTokenizer.from_text('hello')
        # 'e', 'h', 'l', 'o' -- 4 unique chars, sorted
        assert tok.vocab_size == 4

    def test_encode_decode_roundtrip(self) -> None:
        """Encoding then decoding should return the original text."""
        text = 'the cat sat on the mat.'
        tok = CharTokenizer.from_text(text)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_returns_list_of_ints(self) -> None:
        """Encode should return a list of integers."""
        tok = CharTokenizer.from_text('abc')
        ids = tok.encode('abc')
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) == 3

    def test_decode_returns_string(self) -> None:
        """Decode should return a string."""
        tok = CharTokenizer.from_text('abc')
        result = tok.decode([0, 1, 2])
        assert isinstance(result, str)

    def test_vocab_is_sorted(self) -> None:
        """Vocabulary should map characters in sorted order."""
        tok = CharTokenizer.from_text('cab')
        # sorted: a=0, b=1, c=2
        assert tok.encode('a') == [0]
        assert tok.encode('b') == [1]
        assert tok.encode('c') == [2]

    def test_empty_string(self) -> None:
        """Encoding empty string should return empty list."""
        tok = CharTokenizer.from_text('abc')
        assert tok.encode('') == []
        assert tok.decode([]) == ''

    def test_unknown_char_raises(self) -> None:
        """Encoding a character not in vocab should raise KeyError."""
        tok = CharTokenizer.from_text('abc')
        with pytest.raises(KeyError):
            tok.encode('xyz')

    def test_from_text_with_whitespace_and_punctuation(self) -> None:
        """Whitespace and punctuation should be part of the vocabulary."""
        text = 'hello, world!'
        tok = CharTokenizer.from_text(text)
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        assert ' ' in tok.str_to_int
        assert ',' in tok.str_to_int

    def test_vocab_size_property(self) -> None:
        """vocab_size should match the number of unique characters."""
        text = 'aabbcc'
        tok = CharTokenizer.from_text(text)
        assert tok.vocab_size == 3  # a, b, c

    def test_explicit_vocab_constructor(self) -> None:
        """Direct construction with a vocab dict should work."""
        vocab = {'a': 0, 'b': 1, 'c': 2}
        tok = CharTokenizer(vocab)
        assert tok.vocab_size == 3
        assert tok.encode('abc') == [0, 1, 2]

    def test_conforms_to_base_tokenizer(self) -> None:
        """CharTokenizer should be an instance of BaseTokenizer."""
        from lmt.tokenizer.base import BaseTokenizer

        tok = CharTokenizer.from_text('test')
        assert isinstance(tok, BaseTokenizer)
