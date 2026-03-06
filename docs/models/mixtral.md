# Mixtral

Mistral AI's Mixtral (Jiang et al., 2024) extends LLaMA with sparse
Mixture of Experts and sliding window attention.

## What Changed From LLaMA

| Aspect | LLaMA | Mixtral |
|--------|-------|---------|
| FFN | Dense SwiGLU | **MoE** (8 SwiGLU experts, top-2) |
| Attention | Full GQA | **Sliding window** GQA |
| Parameters | All active | **Sparse** (2/8 experts per token) |
| Extra loss | None | **Aux load balancing loss** |

## Architecture

```
Tokens -> Token Embedding
  -> N x MixtralBlock:
       RMSNorm -> MixtralAttention(sliding window GQA + RoPE) -> residual
       RMSNorm -> MoE(8 experts, top-2, SwiGLU) -> residual
  -> RMSNorm -> Linear head -> logits
```

## Usage

```python
from lmt.models.mixtral import Mixtral
from lmt.models.config import ModelConfig

config = ModelConfig(
    vocab_size=32000,
    embed_dim=512,
    num_heads=8,
    num_kv_heads=4,
    num_layers=8,
    context_length=2048,
    window_size=256,
    dropout=0.0,
)

model = Mixtral(
    config,
    num_experts=8,
    top_k=2,
)
```

## Training with Auxiliary Loss

The MoE router produces a load balancing loss that should be added
to the main language modeling loss:

```python
logits = model(input_ids)
lm_loss = cross_entropy(logits, targets)
aux_loss = model.aux_loss  # aggregated from all MoE layers
total_loss = lm_loss + 0.01 * aux_loss
total_loss.backward()
```

## Sparse Activation

Mixtral-8x7B has 47B total parameters but only ~13B are active per
token (the router selects 2 of 8 experts). This means:

- **8x the capacity** of a dense model of the same compute budget
- Each expert can **specialize** in different types of tokens
- The load balancing loss ensures all experts are utilized

## API Reference

::: lmt.models.mixtral.Mixtral
    options:
      show_source: true
      members:
        - __init__
        - forward
