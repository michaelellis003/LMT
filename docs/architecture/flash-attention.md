# Flash Attention

## The Memory Bottleneck

Standard attention computes $\text{softmax}(QK^T / \sqrt{d}) \cdot V$ by
materializing the full $N \times N$ attention score matrix. For a model with
32 heads and sequence length $N = 4096$, that intermediate matrix is:

$$32 \times 4096^2 \times 4 \text{ bytes} = 2 \text{ GB}$$

This is just for **one layer**. The score matrix exists only to compute
softmax weights -- it's immediately consumed and discarded. Yet it must
be written to GPU global memory (HBM), which is the slowest memory tier.

**Flash Attention** eliminates this bottleneck entirely.

## Key Insight: Tiling + Online Softmax

Flash Attention (Dao et al., 2022) never materializes the full $N \times N$
matrix. Instead, it processes $Q$ and $K/V$ in small **tiles** (blocks),
computing attention incrementally.

The challenge is that softmax is a **global** operation -- you need the
maximum value across all keys to compute it numerically stably. Flash
Attention solves this with **online softmax** (Milakov & Gimelshein, 2018),
which maintains a running maximum and sum, rescaling accumulated outputs
when a new tile produces larger scores.

## The Algorithm

```
For each Q block (rows i):
    Initialize: O_i = 0, max_i = -inf, sum_i = 0

    For each K,V block (columns j):
        1. Compute local scores:  S_ij = Q_i @ K_j^T / sqrt(d)
        2. Apply causal mask if needed
        3. Find local max:        m_ij = max(S_ij)
        4. Update global max:     m_new = max(max_i, m_ij)
        5. Rescale old output:    O_i *= exp(max_i - m_new) * sum_i
        6. Add new contribution:  O_i += exp(m_ij - m_new) * exp(S_ij - m_ij) @ V_j
        7. Update sum:            sum_i = exp(max_i - m_new)*sum_i + exp(m_ij - m_new)*sum(exp(S_ij - m_ij))
        8. max_i = m_new

    Normalize: O_i /= sum_i
```

At any point, only a `block_size x block_size` tile exists in memory --
never the full $N \times N$ matrix.

## Why Online Softmax Works

Standard softmax:

$$\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j$$

The problem: you need $m$ (the global max) before computing any
exponentials. Online softmax solves this by observing that if you've
computed a partial result with max $m_1$ and then see a new block with
max $m_2$, you can **rescale** the old result:

$$e^{x - m_1} = e^{x - m_{\text{new}}} \cdot e^{m_{\text{new}} - m_1}$$

This rescaling is exact -- no approximation. The final result is
mathematically identical to standard attention.

## Memory Complexity

| Method | HBM Memory | HBM Reads/Writes |
|--------|-----------|------------------|
| Standard Attention | $O(N^2)$ | $O(N^2 d + N^2)$ |
| Flash Attention | $O(N)$ | $O(N^2 d^2 / M)$ |

Where $M$ is the SRAM (on-chip memory) size. The key: Flash Attention
trades compute for memory access, which is the actual bottleneck on
modern GPUs (compute-bound vs memory-bound).

## LMT Implementation

Our implementation is a **pure Python/PyTorch educational version** that
demonstrates algorithmic correctness. The actual performance benefit of
Flash Attention comes from the fused CUDA kernel keeping tiles in fast
SRAM -- our version still uses HBM for intermediate results.

### Standalone Function

```python
from lmt.layers.attention.flash_attention import flash_attention

# Q, K, V: [batch, heads, seq_len, head_dim]
output = flash_attention(q, k, v, block_size=64, causal=True)
```

### Drop-in Module

```python
from lmt.layers.attention import FlashAttention
from lmt.models.config import ModelConfig

config = ModelConfig(
    vocab_size=32000,
    embed_dim=512,
    num_heads=8,
    num_layers=6,
    context_length=1024,
)

# Drop-in replacement for MultiHeadAttention
attn = FlashAttention(config, block_size=64)

x = torch.randn(2, 128, 512)
out = attn(x)  # [2, 128, 512]
```

### Using PyTorch's Built-in SDPA

For production use, PyTorch's `scaled_dot_product_attention` automatically
selects the best attention backend (Flash Attention v2, memory-efficient
attention, or math fallback):

```python
import torch.nn.functional as F

# PyTorch handles kernel selection automatically
out = F.scaled_dot_product_attention(
    query, key, value,
    is_causal=True,  # applies causal mask efficiently
)
```

This is the recommended approach for actual training -- our educational
implementation is for **understanding the algorithm**, not performance.

## Verification

Our tiled implementation produces results identical to standard attention
(within floating-point tolerance):

```python
# From our test suite
standard_out = softmax(Q @ K^T / sqrt(d)) @ V
flash_out = flash_attention(Q, K, V, block_size=4)
assert torch.allclose(standard_out, flash_out, atol=1e-5)
```

This holds for any block size, including sizes larger than the sequence
length or sizes that don't evenly divide it.

## References

- Dao, T., Fu, D.Y., Ermon, S., Rudra, A., & Re, C. (2022). FlashAttention:
  Fast and Memory-Efficient Exact Attention with IO-Awareness.
  *Advances in Neural Information Processing Systems (NeurIPS)*.
  [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism
  and Work Partitioning. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

- Milakov, M. & Gimelshein, N. (2018). Online normalizer calculation for
  softmax. [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)

## API Reference

::: lmt.layers.attention.flash_attention.flash_attention

::: lmt.layers.attention.flash_attention.FlashAttention
