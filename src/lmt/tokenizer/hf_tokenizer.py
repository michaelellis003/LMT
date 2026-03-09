"""HuggingFace tokenizer wrapper for LMT.

Wraps a HuggingFace ``AutoTokenizer`` to conform to LMT's
``BaseTokenizer`` interface. This enables using pretrained
tokenizers (e.g. Qwen3, LLaMA) with LMT's training pipelines
and reward functions.

Requires the ``transformers`` package.

Usage::

    from lmt.tokenizer.hf_tokenizer import HFTokenizerWrapper

    tok = HFTokenizerWrapper.from_pretrained('Qwen/Qwen3-0.6B')
    ids = tok.encode('Hello world')
    text = tok.decode(ids)
"""

from __future__ import annotations

from typing import Any

from lmt.tokenizer.base import BaseTokenizer


class HFTokenizerWrapper(BaseTokenizer):
    """Wrapper around a HuggingFace tokenizer.

    Implements LMT's ``BaseTokenizer`` interface so HF tokenizers
    can be used with reward functions, data pipelines, and the
    GRPO trainer.

    Args:
        hf_tokenizer: A HuggingFace tokenizer instance.
    """

    def __init__(self, hf_tokenizer: Any) -> None:
        """Initialize with a HuggingFace tokenizer.

        Args:
            hf_tokenizer: A tokenizer from ``transformers``.
        """
        self._tokenizer = hf_tokenizer

    @classmethod
    def from_pretrained(
        cls, model_name: str, **kwargs: Any
    ) -> HFTokenizerWrapper:
        """Load a pretrained tokenizer from HuggingFace Hub.

        Args:
            model_name: HuggingFace model name or path.
            **kwargs: Extra arguments passed to ``AutoTokenizer``.

        Returns:
            HFTokenizerWrapper instance.
        """
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as e:
            raise ImportError(
                'transformers is required for HF tokenizer support. '
                'Install with: pip install transformers'
            ) from e

        hf_tok = AutoTokenizer.from_pretrained(model_name, **kwargs)
        return cls(hf_tok)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string.

        Returns:
            List of integer token IDs (no special tokens added).
        """
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs.

        Returns:
            Decoded text string.
        """
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int | None:
        """Return end-of-sequence token ID."""
        return self._tokenizer.eos_token_id
