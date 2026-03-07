"""Character-level tokenizer.

Maps each unique character in a text corpus to an integer ID.
Simple, dependency-free, and useful for educational training recipes
where subword tokenization is unnecessary overhead.
"""

from __future__ import annotations

from .base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer that maps individual characters to IDs.

    Each unique character gets a unique integer, assigned in sorted order.
    This is the simplest possible tokenizer -- no merges, no subwords,
    no external dependencies.

    Attributes:
        str_to_int: Mapping from character to token ID.
        int_to_str: Mapping from token ID to character.
        vocab_size: Number of unique characters in the vocabulary.
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        """Initialize with an explicit character-to-ID mapping.

        Args:
            vocab: Dictionary mapping characters to integer IDs.
        """
        self.str_to_int = vocab
        self.int_to_str = {i: ch for ch, i in vocab.items()}
        self.vocab_size = len(vocab)

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Build a CharTokenizer from a text corpus.

        Creates a vocabulary from all unique characters in the text,
        sorted alphabetically so the mapping is deterministic.

        Args:
            text: The text corpus to build vocabulary from.

        Returns:
            A CharTokenizer with vocabulary derived from the text.
        """
        chars = sorted(set(text))
        vocab = {ch: i for i, ch in enumerate(chars)}
        return cls(vocab)

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of character IDs.

        Args:
            text: The string to encode.

        Returns:
            List of integer token IDs.

        Raises:
            KeyError: If text contains a character not in the vocabulary.
        """
        return [self.str_to_int[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of character IDs back into text.

        Args:
            token_ids: List of integer token IDs to decode.

        Returns:
            The decoded string.
        """
        return ''.join(self.int_to_str[i] for i in token_ids)
