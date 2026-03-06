# Attention Pattern Explorer

Every attention mechanism defines a **pattern** — which tokens can attend to
which other tokens. This interactive visualization lets you explore the
patterns used by different architectures in LMT.

<div class="attn-viz" data-seq-len="12"></div>

## How to Read This

The heatmap shows an **attention weight matrix** for a single head:

- **Rows** = query positions (the token asking "what should I pay attention to?")
- **Columns** = key positions (the tokens being attended to)
- **Color intensity** = attention weight (darker = stronger attention)
- **Hover** over any cell to see the exact weight

When "Apply softmax" is checked, each row sums to 1.0 — just like in a real
transformer. Turn it off to see the raw binary mask.

## Pattern Guide

### Causal (GPT, LLaMA)

The most common pattern for language modeling. Token $i$ can only see tokens
$j \leq i$. This is what makes generation autoregressive — each token is
predicted using only the past.

In code, this is a **upper-triangular mask filled with $-\infty$** applied
before softmax:

```python
# From lmt/layers/attention/grouped_query_attention.py
mask = torch.full((seq_len, seq_len), float('-inf')).triu(diagonal=1)
attn_scores = attn_scores + mask  # masked positions → 0 after softmax
```

Notice how the softmax redistributes weight: early tokens spread attention
evenly over few positions, while later tokens spread it thinly over many.

### Sliding Window (Mistral, Mixtral)

Each token attends to only the most recent $w$ tokens. This bounds memory
usage to $O(w)$ per layer regardless of sequence length — essential for
long-context models.

**Try adjusting the window size** to see how it affects the pattern. A window
of 1 means each token only sees itself. As the window grows toward the
sequence length, it approaches causal attention.

In LMT, this is implemented in
[`SlidingWindowAttention`](../api/layers.md).

### Global + Local (Longformer, BigBird)

A hybrid: most tokens use a local sliding window, but a few designated
**global tokens** (typically the first few positions) attend to and are
attended by all other tokens. This gives the model both local detail and
global context.

**Experiment**: set global tokens to 1 and watch how just the first row
and column light up. These act as "information highways" across the sequence.

### Strided (Sparse Transformer)

Attends to every $s$-th previous position. When combined with a local
window in alternating layers, this achieves $O(n\sqrt{n})$ complexity.

**Try stride=2**: notice the checkerboard-like pattern. Each token samples
a sparse set of past positions, capturing long-range dependencies without
the full quadratic cost.

### Full (BERT-style)

Every token sees every other token — no masking at all. Used in
bidirectional encoder models like BERT where the full context is
available. Not used for text generation (no autoregressive property).

## Connecting to LMT Code

Each pattern corresponds to a component in the library:

| Pattern | LMT Class | Registry Key |
|---------|-----------|-------------|
| Causal | `GroupedQueryAttention` | `'gqa'` |
| Sliding Window | `SlidingWindowAttention` | `'sliding_window'` |
| Full | `MultiHeadAttention` | `'mha'` |

You can mix patterns using `ConfigurableBlock`:

```python
from lmt.layers.blocks import BlockConfig, ConfigurableBlock
from lmt.models.config import ModelConfig

config = ModelConfig(vocab_size=32000, embed_dim=256,
                     num_heads=4, num_kv_heads=2,
                     num_layers=8, context_length=2048)

# A block with sliding window attention
block_cfg = BlockConfig(attention='sliding_window', ffn='swiglu')
block = ConfigurableBlock(config, block_cfg)
```

## What's Next?

- [Attention Mechanisms](../architecture/attention.md) — deep dive into
  MHA, GQA, MLA with math
- [Transformer Family Tree](../architecture/family-tree.md) — how these
  patterns evolved over time
