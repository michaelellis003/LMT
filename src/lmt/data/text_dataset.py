"""Text dataset with automatic tokenization and packing.

Bridges raw text strings to packed token sequences for language model
training. Handles tokenization, document packing (with optional
separator tokens), and chunking into fixed-length training samples.

Usage::

    from lmt.data.text_dataset import TextDataset, texts_from_dicts

    # From raw text
    texts = ['First document.', 'Second document.']
    ds = TextDataset(texts, tokenizer, context_length=1024)

    # From HuggingFace dataset rows
    from datasets import load_dataset

    rows = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = texts_from_dicts(rows, text_field='text')
    ds = TextDataset(texts, tokenizer, context_length=1024)

    loader = DataLoader(ds, batch_size=8, shuffle=True)
"""

from __future__ import annotations

from typing import Any

import torch

from lmt.data.token_dataset import TokenDataset, pack_documents
from lmt.tokenizer.base import BaseTokenizer


def texts_from_dicts(
    rows: list[dict[str, Any]] | Any,
    text_field: str = 'text',
) -> list[str]:
    """Extract text strings from a list of dicts.

    Filters out empty or whitespace-only strings. Compatible with
    HuggingFace dataset rows.

    Args:
        rows: Iterable of dicts, each containing a text field.
        text_field: Key to extract text from each dict.

    Returns:
        List of non-empty text strings.
    """
    texts: list[str] = []
    for row in rows:
        text = row[text_field]
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


class TextDataset(TokenDataset):
    """Dataset that tokenizes and packs text into training sequences.

    Wraps ``TokenDataset`` with an automatic tokenization and packing
    step. Takes raw text strings, tokenizes them, packs into a single
    token stream (with optional separator), and yields fixed-length
    chunks for language model training.

    Args:
        texts: List of text strings (documents).
        tokenizer: Tokenizer with an ``encode`` method.
        context_length: Number of input tokens per sample.
        sep_token_id: Token ID to insert between documents.
            If None, documents are concatenated directly.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: BaseTokenizer,
        context_length: int,
        sep_token_id: int | None = None,
    ) -> None:
        """Initialize text dataset.

        Args:
            texts: List of text strings to tokenize and pack.
            tokenizer: Tokenizer with ``encode(str) -> list[int]``.
            context_length: Number of input tokens per sample.
            sep_token_id: Optional separator token between documents.
        """
        # Tokenize all documents
        documents: list[torch.Tensor] = []
        for text in texts:
            ids = tokenizer.encode(text)
            if ids:
                documents.append(torch.tensor(ids, dtype=torch.long))

        # Pack into single token stream
        if documents:
            packed = pack_documents(documents, sep_token_id=sep_token_id)
        else:
            packed = torch.tensor([], dtype=torch.long)

        super().__init__(packed, context_length)
