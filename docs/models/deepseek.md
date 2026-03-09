# DeepSeek-V2

DeepSeek-AI's DeepSeek-V2 (2024) combines Multi-Head Latent Attention
with fine-grained Mixture of Experts for an efficient, high-capacity
language model.

## What Changed From Mixtral

| Aspect | Mixtral | DeepSeek-V2 |
|--------|---------|-------------|
| Attention | GQA + RoPE | **MLA** (compressed KV + decoupled RoPE) |
| FFN | MoE (8 experts, top-2) | **DeepSeekMoE** (many experts + shared experts) |
| KV Cache | Full K,V per head | **Compressed latent** (~93% reduction) |
| Early Layers | All MoE | **Dense SwiGLU** for first N layers |

## Architecture

```
Tokens -> Token Embedding
  -> N x DeepSeekBlock:
       RMSNorm -> MLA(compressed KV + decoupled RoPE) -> residual
       RMSNorm -> MoE/SwiGLU(routed + shared experts) -> residual
  -> RMSNorm -> Linear head -> logits
```

## Multi-Head Latent Attention (MLA)

The key innovation is KV compression with decoupled positional encoding:

\[
c_t^{KV} = x_t W^{DKV} \quad \text{(compress to latent)}
\]

\[
k_C = c^{KV} W^{UK}, \quad v = c^{KV} W^{UV} \quad \text{(decompress)}
\]

\[
k_R = \text{RoPE}(x_t W^{KR}) \quad \text{(separate position path)}
\]

At inference, only the small latent \(c_t^{KV}\) and \(k_R\) are cached
instead of full K and V tensors.

## Usage

```python
from lmt.models.deepseek import DeepSeekV2
from lmt.models.config import ModelConfig

config = ModelConfig(
    vocab_size=32000,
    embed_dim=512,
    num_heads=8,
    num_kv_heads=8,
    num_layers=8,
    context_length=2048,
    dropout=0.0,
)

model = DeepSeekV2(
    config,
    kv_compress_dim=128,
    q_compress_dim=128,
    rope_dim=32,
    num_experts=8,
    top_k=2,
    num_shared_experts=1,
    num_dense_layers=2,
)
```

## Training with Auxiliary Loss

Same pattern as Mixtral -- the MoE router produces a load balancing
loss that should be added to the main loss:

```python
logits = model(input_ids)
lm_loss = cross_entropy(logits, targets)
aux_loss = model.aux_loss  # aggregated from all MoE layers
total_loss = lm_loss + 0.01 * aux_loss
total_loss.backward()
```

## Dense Early Layers

The `num_dense_layers` parameter controls how many initial layers use
dense SwiGLU FFN instead of MoE. This follows the paper's design where
early layers benefit from dense computation while later layers gain
capacity from sparse experts.

## API Reference

See the full API docs at [Models API](../api/models.md#deepseek-v2).

## References

- DeepSeek-AI, [*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434) (2024) -- DeepSeek-V2 architecture
- Jiang et al., [*Mixtral of Experts*](https://arxiv.org/abs/2401.04088) (2024) -- MoE foundation
