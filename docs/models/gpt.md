# GPT

The original decoder-only transformer architecture from OpenAI.

## Architecture

GPT uses the classic transformer decoder stack:

- **Learned positional embeddings** (absolute position)
- **Pre-norm** LayerNorm
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

## References

- Radford et al., [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018) -- GPT-1
- Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019) -- GPT-2
