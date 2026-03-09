"""Streaming data pipeline for pretraining.

Tokenizes and packs text from any iterable (including HuggingFace
streaming datasets) into a ``TokenDataset`` for language model training.
Processes data in a single pass without loading everything into memory.

Usage::

    from lmt.data.streaming import tokenize_and_pack

    # From a HuggingFace streaming dataset:
    ds = load_dataset('wikitext', split='train', streaming=True)
    dataset = tokenize_and_pack(
        texts=(row['text'] for row in ds),
        tokenize_fn=tokenizer.encode,
        context_length=1024,
        sep_token_id=tokenizer.eos_token_id,
        max_documents=10000,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch

from lmt.data.token_dataset import TokenDataset


def tokenize_and_pack(
    texts: Iterator[str],
    tokenize_fn: Callable[[str], list[int]],
    context_length: int,
    sep_token_id: int | None = None,
    max_documents: int | None = None,
) -> TokenDataset:
    """Tokenize and pack text from an iterable into a TokenDataset.

    Streams through the input texts, tokenizing each one and
    concatenating into a flat token buffer. The result is wrapped
    in a ``TokenDataset`` for use with PyTorch ``DataLoader``.

    Args:
        texts: Iterator of text strings to tokenize.
        tokenize_fn: Function mapping text to token ID list.
        context_length: Sequence length for the resulting dataset.
        sep_token_id: Optional separator token between documents.
        max_documents: Maximum number of documents to process.
            ``None`` means process all available.

    Returns:
        A ``TokenDataset`` with packed sequences.
    """
    all_tokens: list[int] = []
    doc_count = 0

    for text in texts:
        if max_documents is not None and doc_count >= max_documents:
            break

        tokens = tokenize_fn(text)
        if not tokens:
            continue

        if all_tokens and sep_token_id is not None:
            all_tokens.append(sep_token_id)
        all_tokens.extend(tokens)
        doc_count += 1

    if not all_tokens:
        return TokenDataset(
            torch.tensor([], dtype=torch.long),
            context_length,
        )

    return TokenDataset(
        torch.tensor(all_tokens, dtype=torch.long),
        context_length,
    )
