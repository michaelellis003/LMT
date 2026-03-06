# Positional Encoding Comparison

An interactive side-by-side comparison of the three positional encoding
strategies implemented (or referenced) in LMT: **RoPE**, **ALiBi**, and
**Sinusoidal**.

## Why Position Matters

Transformers are permutation-invariant by default -- they don't know token
order. Positional encodings inject sequence structure. But *how* they do it
varies dramatically, with real consequences for context length, efficiency,
and attention patterns.

## Interactive Comparison

<div class="pos-encoding-compare" data-seq-len="32" data-d-model="64" data-num-heads="8"></div>

### What to explore

- **Sequence length slider**: Watch how each method scales. RoPE and sinusoidal
  develop richer interference patterns; ALiBi stays rigidly linear.
- **Num heads (ALiBi)**: More heads = more diverse slopes. Head 0 always has the
  steepest penalty; the last head has the gentlest. This diversity is intentional --
  some heads attend locally, others globally.
- **RoPE base**: Higher base = slower decay = longer effective context. This is
  the lever used by YaRN, CodeLlama, and other context extension methods.

### Heatmap guide

| Heatmap | What it shows |
|---------|---------------|
| **RoPE Similarity** | Average cosine similarity between rotated embeddings. Bright diagonal = nearby positions are similar. |
| **ALiBi Bias** | Linear penalty added to attention scores. Pure diagonal pattern -- no learned structure, just geometry. |
| **Sinusoidal Similarity** | Dot product of classic sinusoidal encodings. Notice the periodic banding -- this is absolute position, not relative. |

### Decay comparison chart

The line chart shows how each method's "attention bonus" decays with distance:

- **RoPE** (purple): Oscillating decay from multi-frequency interference.
  Nearby positions get high similarity, but there are subtle periodic peaks.
- **ALiBi** (pink): Perfectly linear decay. Simple, predictable, excellent
  for length extrapolation.
- **Sinusoidal** (green): Periodic structure -- positions at certain distances
  look similar again. This is why absolute sinusoidal encoding struggles
  with length generalization.

### Key insight: ALiBi per-head diversity

The small heatmaps show each attention head's bias pattern. Head 0 has the
steepest slope (strongest locality bias), while the last head has the
gentlest (most global attention). The model learns to use different heads
for different ranges of context.

**Slope formula**: `m_h = 2^(-8/n * h)` where `n` = num_heads, `h` = head index

### Connection to LMT code

- [`RoPE`](../api/layers.md): `src/lmt/layers/positional/rope.py`
- [`ALiBi`](../api/layers.md): `src/lmt/layers/positional/alibi.py`
- Both are registered in the positional encoding registry and can be
  swapped via configuration.

## References & Further Reading

- Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) (2021) -- RoPE
- Press et al., [*Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*](https://arxiv.org/abs/2108.12409) (2022) -- ALiBi
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017) -- sinusoidal positional encoding
- Brenndoerfer, [*Position Encoding Comparison*](https://mbrenndoerfer.com/writing/position-encoding-comparison-transformers) -- comprehensive guide with static visualizations and extrapolation analysis
- [*Positional Embeddings in Transformer Models*](https://iclr-blogposts.github.io/2025/blog/positional-embedding/) (ICLR 2025 Blog) -- survey of positional encoding evolution

Our visualization is an original interactive Canvas implementation. While the
concept of comparing positional encodings is well-established (see resources
above), our specific combination of live heatmaps, decay curves, and per-head
ALiBi slope visualization with real-time parameter adjustment is our own design.
