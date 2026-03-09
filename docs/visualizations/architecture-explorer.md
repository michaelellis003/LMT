# Architecture Explorer

Click on any model to see its transformer block composition.
Each model in LMT is built from composable blocks using the
`ConfigurableBlock` system -- the same blocks can be mixed and matched
to create custom architectures.

<div class="arch-explorer"></div>

## The ConfigurableBlock System

LMT uses a **registry-based composition** pattern. Instead of hardcoding
each architecture, a `BlockConfig` specifies the components:

```python
from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.models.base import BaseModel

# Create a custom hybrid: GQA + SwiGLU + LayerNorm + learned pos
block = BlockConfig(
    attention='gqa',
    ffn='swiglu',
    norm='layernorm',
)
model = BaseModel(config, block_config=block, learned_pos_embed=True)
```

This is exactly how the [Architecture Ablation](../experiments/architecture-ablation.md)
experiment works -- isolating each feature by changing one component at a time.

## Available Components

### Attention Mechanisms

| Key       | Description                              |
|-----------|------------------------------------------|
| `mha`     | Multi-Head Attention (standard)          |
| `gqa`     | Grouped Query Attention (shared KV heads)|
| `mla`     | Multi-head Latent Attention (DeepSeek)   |
| `flash`   | Flash Attention (memory-efficient)       |
| `sliding` | Sliding Window Attention (Gemma)         |
| `gdn`     | Gated Delta Net (linear attention)       |
| `ssd`     | State Space Duality (Mamba-2)            |

### Feed-Forward Networks

| Key       | Description                              |
|-----------|------------------------------------------|
| `default` | Standard FFN with GELU activation        |
| `swiglu`  | SwiGLU gated FFN (LLaMA, Qwen3)         |
| `geglu`   | GeGLU gated FFN (Gemma)                  |
| `relu2`   | ReLU-squared GLU                         |
| `moe`     | Mixture-of-Experts (Mixtral)             |

### Normalization

| Key         | Description                            |
|-------------|----------------------------------------|
| `layernorm` | Standard Layer Normalization           |
| `rmsnorm`   | Root Mean Square Normalization (faster) |

### Positional Encoding

| Method   | Description                              |
|----------|------------------------------------------|
| Learned  | Absolute position embeddings (GPT)       |
| RoPE     | Rotary Position Embeddings (LLaMA+)     |
