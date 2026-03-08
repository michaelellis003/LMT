# Kimi K2

Moonshot AI's Kimi K2 (2025) uses the DeepSeek-V2/V3 architecture
family: Multi-Head Latent Attention (MLA) combined with Mixture of
Experts (MoE).

## Architecture

Kimi K2 is architecturally a DeepSeek-V2 model. The main innovations
are on the **training side** (MuonClip optimizer, multi-stage
training), not the architecture.

| Component | Implementation |
|-----------|---------------|
| Attention | **MLA** (compressed KV cache) |
| FFN | **MoE** with shared experts |
| Normalization | RMSNorm |
| Early layers | Dense SwiGLU (not MoE) |

See [DeepSeek-V2](deepseek.md) for detailed explanation of MLA
and the MoE architecture.

## Usage

```python
from lmt import Kimi, ModelConfig

config = ModelConfig(
    vocab_size=32000,
    embed_dim=512,
    num_heads=8,
    num_layers=8,
    context_length=2048,
    dropout=0.0,
)

model = Kimi(
    config,
    kv_compress_dim=32,    # KV latent dimension for MLA
    q_compress_dim=48,     # Q latent dimension for MLA
    rope_dim=16,           # decoupled RoPE dimension
    num_experts=4,         # MoE experts per layer
    top_k=2,               # active experts per token
    num_shared_experts=1,  # always-active shared experts
    num_dense_layers=1,    # dense layers before MoE
)
```

## API Reference

::: lmt.models.kimi.Kimi
    options:
      show_source: true
      members:
        - __init__
        - forward

## References

- Kimi Team, Moonshot AI, [*Kimi K2: Open Agentic Intelligence*](https://arxiv.org/abs/2507.20534) (2025)
