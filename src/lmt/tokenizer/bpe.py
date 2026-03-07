"""Byte Pair Encoding (BPE) tokenizer implementation."""

from __future__ import annotations

from .base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """Byte Pair Encoding (BPE) tokenizer class.

    Using OpenAI's tiktoken library for tokenization.
    Tiktoken is imported lazily so that LMT can be used in
    environments where tiktoken is not installed (e.g. GPU VMs).
    """

    def __init__(
        self, encoding_name: str = 'gpt2', allowed_special: set | None = None
    ):
        """Initializes the BPE tokenizer with a specified encoding."""
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                'tiktoken is required for BPETokenizer. '
                'Install it with: pip install tiktoken'
            ) from exc
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.allowed_special = (
            allowed_special if allowed_special is not None else set()
        )

    def encode(self, text: str):
        """Encodes the input text using BPE."""
        return self.tokenizer.encode(
            text, allowed_special=self.allowed_special
        )

    def decode(self, token_ids: list[int]) -> str:
        """Decodes the encoded text back to original tokens."""
        return self.tokenizer.decode(token_ids)
