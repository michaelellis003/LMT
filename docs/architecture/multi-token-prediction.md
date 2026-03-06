# Multi-Token Prediction

## Overview

Standard language models predict one token at a time: given context $x_1, \ldots, x_t$,
predict $x_{t+1}$. **Multi-Token Prediction** (MTP) extends this by simultaneously
predicting the next $N$ tokens $x_{t+1}, x_{t+2}, \ldots, x_{t+N}$ using $N$ separate
prediction heads that share the same backbone.

This was proposed by Gloeckle et al. (2024) at Meta and has two key benefits:

1. **Richer training signal**: Each position contributes $N$ gradient signals instead of 1,
   improving sample efficiency (especially for code and structured data)
2. **Speculative decoding**: The extra heads can propose candidate tokens that are verified
   in parallel, enabling faster inference

## Architecture

```
Input tokens ──→ [Backbone Model] ──→ Hidden States (h)
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
              [Shared LM Head]    [Future Layer 1]      [Future Layer 2]
                    │              + Shared LM Head      + Shared LM Head
                    ▼                     ▼                     ▼
              logits (t+1)         logits (t+2)          logits (t+3)
```

### Key Components

**Backbone**: Any `BaseModel` (GPT, LLaMA, Mixtral, etc.). MTP is a wrapper, not a new model.

**Future Transformer Layers**: Each additional head $k > 0$ has a lightweight transformer layer
that processes:

- The backbone's hidden states $h$
- The embedding of the target token at position $t+k-1$ (teacher-forced during training)

These are concatenated, projected to $d_\text{model}$, then processed through self-attention + FFN.

**Shared Unembedding**: All heads share the same output projection matrix (weight-tied with the
backbone's LM head). This keeps parameter overhead small — only the future layers add parameters.

## Training

The total loss is a weighted sum across heads:

$$\mathcal{L} = \sum_{k=0}^{N-1} w_k \cdot \text{CE}(\hat{y}^{(k)}, y_{t+k+1})$$

By default, all heads are weighted equally ($w_k = 1/N$). The paper found that $N=4$ works
well across most settings.

### Teacher Forcing

During training, each future head receives the **ground-truth** embedding of the previous
position's target token (not the model's prediction). This is standard teacher forcing — it
stabilizes training and avoids error accumulation.

### Why It Helps

The intuition is that predicting multiple steps ahead forces the backbone to learn
**more abstract representations**. If you need to predict not just the next word but
also the word after that, you need to understand the broader context and structure,
not just local patterns.

The paper found this particularly effective for **code generation**, where multi-step
prediction captures syntactic structure (matching brackets, function signatures, etc.)
that single-token prediction misses.

## Inference

At inference time, you have two options:

1. **Standard**: Use only head 0 (next-token prediction). The extra heads are free
   to ignore — they were just a training signal.

2. **Speculative Decoding**: Use heads 1..N-1 to propose candidates, then verify
   them in a single forward pass. This can give ~2-3x speedup for autoregressive
   generation.

## LMT Usage

```python
from lmt.models.llama import LLaMA
from lmt.models.config import ModelConfig
from lmt.models.multi_token_prediction import MultiTokenPredictionHead

# Create any BaseModel
config = ModelConfig(vocab_size=32000, embed_dim=512, num_heads=8,
                     num_kv_heads=4, num_layers=6, context_length=1024,
                     dropout=0.1)
model = LLaMA(config)

# Wrap with MTP (4 prediction heads)
mtp = MultiTokenPredictionHead(model, num_heads=4)

# Training
logits_list = mtp(input_ids, targets=target_ids)  # list of 4 logit tensors
loss = mtp.compute_loss(logits_list, target_ids)
loss.backward()

# Inference (just use the base model directly)
with torch.no_grad():
    logits = model(input_ids)  # standard next-token prediction
```

### Custom Head Weights

You can weight heads differently — for example, giving more weight to
near-future predictions:

```python
loss = mtp.compute_loss(logits_list, targets,
                        head_weights=[0.5, 0.25, 0.15, 0.1])
```

## Design Philosophy

MTP fits LMT's composable architecture perfectly — it's a **training wrapper**, not
a model variant. Any `BaseModel` can be wrapped with MTP without changing its architecture.
This means you can experiment with MTP + LLaMA, MTP + Mixtral, or MTP + any future model.

The `forward_hidden()` method on `BaseModel` enables this by exposing the hidden states
before the LM head — a general hook that can also be used for other head-based extensions
(e.g., reward models, classification heads, retrieval-augmented generation).

## Complexity

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Backbone | Standard | Unchanged — MTP doesn't modify it |
| Per future layer | ~$12 d^2$ | Linear proj + self-attn + FFN |
| LM head | Shared | Weight-tied, no extra params |
| **Total overhead** | ~$(N-1) \cdot 12 d^2$ | Small relative to backbone |

For a 512-dim model with $N=4$: ~9.4M extra parameters for the 3 future layers,
compared to ~50M+ for the backbone.

## References

- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
  (Gloeckle et al., 2024)
- [Speculative Decoding with Big Little Decoder](https://arxiv.org/abs/2302.07863)
  (Kim et al., 2023) — related work on multi-draft speculation
