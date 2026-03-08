"""Token dataset with sequence packing for pretraining.

Provides a PyTorch Dataset that takes a flat 1D tensor of token IDs
and yields packed, fixed-length sequences for language model training.
Each sample is ``context_length + 1`` tokens: the extra token provides
the final target for next-token prediction.

Document packing concatenates multiple documents (optionally with a
separator token) into a single long token stream, eliminating padding
waste. This is the standard approach used by GPT, LLaMA, and other
modern LLMs.

Usage::

    from lmt.data.token_dataset import TokenDataset, pack_documents

    # Pack documents into a single token stream
    packed = pack_documents(tokenized_docs, sep_token_id=eos_id)

    # Create dataset for DataLoader
    dataset = TokenDataset(packed, context_length=1024)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in loader:
        inputs, targets = batch[:, :-1], batch[:, 1:]
"""

import torch
from torch.utils.data import Dataset


def pack_documents(
    documents: list[torch.Tensor],
    sep_token_id: int | None = None,
) -> torch.Tensor:
    """Pack multiple tokenized documents into a single tensor.

    Concatenates document token tensors, optionally inserting a
    separator token between each pair. This is the standard
    "packing" approach that avoids wasted padding tokens.

    Args:
        documents: List of 1D token ID tensors.
        sep_token_id: Token ID to insert between documents.
            If None, documents are concatenated directly.

    Returns:
        1D tensor of packed token IDs.
    """
    if not documents:
        return torch.tensor([], dtype=torch.long)

    parts: list[torch.Tensor] = []
    for i, doc in enumerate(documents):
        if i > 0 and sep_token_id is not None:
            parts.append(torch.tensor([sep_token_id], dtype=doc.dtype))
        parts.append(doc)

    return torch.cat(parts)


class TokenDataset(Dataset):
    """Dataset of fixed-length token sequences for LM pretraining.

    Takes a flat 1D tensor of token IDs and yields non-overlapping
    chunks of ``context_length + 1`` tokens. The extra token is
    needed for the target shift in next-token prediction:
    ``inputs = chunk[:-1]``, ``targets = chunk[1:]``.

    Any leftover tokens that don't fill a complete chunk are discarded.

    Args:
        tokens: 1D tensor of token IDs (the full corpus).
        context_length: Number of input tokens per sample.
    """

    def __init__(self, tokens: torch.Tensor, context_length: int) -> None:
        """Initialize token dataset.

        Args:
            tokens: 1D tensor of token IDs.
            context_length: Number of input tokens per sample.
                Each returned item has ``context_length + 1`` tokens
                (the extra token is the final prediction target).
        """
        self.tokens = tokens
        self.context_length = context_length
        self._chunk_size = context_length + 1
        self._n_samples = len(tokens) // self._chunk_size

    def __len__(self) -> int:
        """Return number of complete sequences."""
        return self._n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a token sequence by index.

        Args:
            idx: Sample index (supports negative indexing).

        Returns:
            Token IDs ``[context_length + 1]``.

        Raises:
            IndexError: If index is out of range.
        """
        if idx < 0:
            idx += self._n_samples
        if idx < 0 or idx >= self._n_samples:
            raise IndexError(
                f'Index {idx} out of range for dataset '
                f'with {self._n_samples} samples'
            )

        start = idx * self._chunk_size
        return self.tokens[start : start + self._chunk_size]
