# GPT

The original decoder-only transformer architecture from OpenAI.

## Architecture

GPT uses the classic transformer decoder stack:

- **Learned positional embeddings** (absolute position)
- **Post-norm** LayerNorm
- **Multi-head attention** with causal masking
- **Standard FFN** with GELU activation

```python
from lmt.models.gpt import GPT
from lmt.models.config import ModelConfig, ModelConfigPresets

# Use a preset
config = ModelConfigPresets.small_gpt()
model = GPT(config)

# Or configure manually
config = ModelConfig(
    vocab_size=50257,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    context_length=1024,
    dropout=0.1,
)
model = GPT(config)
```

## API Reference

::: lmt.models.gpt.GPT
    options:
      show_source: true
      members:
        - __init__
        - forward
