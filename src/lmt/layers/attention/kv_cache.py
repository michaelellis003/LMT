"""KV Cache for efficient autoregressive generation.

During autoregressive text generation, each new token needs to attend
to all previous tokens. Without a cache, this means recomputing all
K and V projections at every step -- O(n^2) total work for n tokens.

The KV cache stores previously computed K and V tensors, so each
generation step only computes K/V for the new token and concatenates
with the cache. This reduces generation to O(n) total work.

Cache size per layer = ``2 * num_kv_heads * head_dim * seq_len * dtype_bytes``.
For a 7B model with 32 layers, GQA (8 KV heads), head_dim=128, FP16:
  = 32 * 2 * 8 * 128 * seq_len * 2 bytes
  = ~128 KB per token in the sequence.
At 4096 tokens, that's ~512 MB just for the KV cache.
"""

import torch
from torch import Tensor


class KVCache:
    """Stores past key and value tensors for autoregressive generation.

    The cache accumulates K/V tensors along the sequence dimension.
    An optional ``max_seq_len`` enables a sliding window that drops
    the oldest entries when the cache is full.

    Args:
        max_seq_len: Maximum cached sequence length. ``None`` for
            unlimited. When set, oldest entries are evicted first.

    Example::

        cache = KVCache(max_seq_len=2048)

        # Prefill: process entire prompt
        k, v = compute_kv(prompt)  # [b, heads, prompt_len, d]
        k, v = cache.update(k, v)  # stores in cache

        # Generate: one token at a time
        for _ in range(max_new_tokens):
            k_new, v_new = compute_kv(last_token)  # [b, heads, 1, d]
            k, v = cache.update(k_new, v_new)  # returns full K/V
            # k, v now have shape [b, heads, total_len, d]
    """

    def __init__(self, max_seq_len: int | None = None) -> None:
        """Initialize an empty KV cache.

        Args:
            max_seq_len: Maximum cached sequence length.
        """
        self.max_seq_len = max_seq_len
        self._k: Tensor | None = None
        self._v: Tensor | None = None

    @property
    def seq_len(self) -> int:
        """Current number of cached positions."""
        if self._k is None:
            return 0
        return self._k.shape[2]

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append new K/V and return the full cached tensors.

        Args:
            k: New keys ``[batch, heads, new_seq, head_dim]``.
            v: New values ``[batch, heads, new_seq, head_dim]``.

        Returns:
            Tuple of (full_keys, full_values) with all cached
            positions concatenated along the sequence dimension.
        """
        if self._k is None or self._v is None:
            self._k = k
            self._v = v
        else:
            self._k = torch.cat([self._k, k], dim=2)
            self._v = torch.cat([self._v, v], dim=2)

        # Evict oldest entries if over max length
        if self.max_seq_len is not None and self.seq_len > self.max_seq_len:
            self._k = self._k[:, :, -self.max_seq_len :, :]
            self._v = self._v[:, :, -self.max_seq_len :, :]

        assert self._k is not None and self._v is not None
        return self._k, self._v

    def reset(self) -> None:
        """Clear the cache."""
        self._k = None
        self._v = None
