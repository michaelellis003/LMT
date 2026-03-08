"""Simple (naive) tokenizer that splits text by whitespace and punctuation."""

import re

from .base import BaseTokenizer


class NaiveTokenizer(BaseTokenizer):
    """A naive tokenizer that splits text by whitespace and punctuation."""

    def __init__(self, vocab: dict, unknown_str: str = '<unk>') -> None:
        """Initialize the tokenizer with a vocabulary."""
        super().__init__()
        self.str_to_int = vocab  # vocabulary mapping from string to token ID
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.unknown_str = unknown_str
        self.unknown_token = self.str_to_int.get(unknown_str, len(vocab))

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        # Preprocess the text by splitting on whitespace and punctuation
        preprocessed = re.split(r'([,.:;?_!()\']|--|\s)', text)

        # Remove empty and whitespace-only strings
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Convert each preprocessed token into its corresponding ID from the
        # vocabulary. If a token is not found in the vocabulary, use the ID
        # for the unknown token.
        ids = [
            self.str_to_int.get(s, self.unknown_token) for s in preprocessed
        ]

        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        text = ' '.join(
            self.int_to_str.get(i, self.unknown_str) for i in token_ids
        )

        # Remove space before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text
