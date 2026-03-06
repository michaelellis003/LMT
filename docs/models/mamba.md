# Mamba

Mamba (Gu & Dao, 2023) is a **Selective State Space Model** -- a fundamentally
different approach to sequence modeling that replaces attention with a linear
recurrence. It processes sequences in O(n) time with a constant-size hidden
state, eliminating the need for a KV cache entirely.

## What Changed From Transformers

| Aspect | Transformer | Mamba |
|--------|-------------|-------|
| Sequence Mixing | Attention (O(n^2)) | **SSM** (O(n)) |
| Memory | KV cache (grows with sequence) | **Fixed-size state** |
| Position | RoPE / learned / ALiBi | **Implicit** (via recurrence) |
| FFN | Separate FFN block | **Integrated** (gating in same block) |
| Parallelism | Trivially parallel | **Parallel scan** (more complex) |

## Architecture

```
Tokens -> Token Embedding
  -> N x MambaBlock:
       RMSNorm -> in_proj -> split(x, z)
       x: Conv1d -> SiLU -> SelectiveSSM
       Gate: y * SiLU(z)
       out_proj -> + residual
  -> RMSNorm -> Linear head -> logits
```

## The Selective SSM

The core mechanism is a linear recurrence with **input-dependent** parameters:

\[
h_t = \bar{A} \cdot h_{t-1} + \bar{B} \cdot x_t
\]
\[
y_t = C \cdot h_t + D \cdot x_t
\]

where the discretization converts continuous parameters to discrete:

\[
\bar{A} = \exp(\Delta \cdot A), \quad \bar{B} = \Delta \cdot B
\]

### What Makes It "Selective"

In classical SSMs (like S4), the parameters A, B, C are fixed after training.
In Mamba, **B, C, and Delta are all projected from the input x**, making them
content-dependent. This means the model learns *what to remember* based on
what it's currently seeing -- analogous to how attention learns *what to
attend to*.

### Numerical Stability

- **A in log-space**: The state matrix A must be negative real for stable
  dynamics (decaying memory). It's stored as `A_log` and computed as
  `A = -exp(A_log)`, guaranteeing negativity.
- **Delta via softplus**: The time step Delta must be positive. Softplus
  (`log(1 + exp(x))`) ensures this smoothly.

## Usage

```python
from lmt.models.config import ModelConfig
from lmt.models.mamba import Mamba

config = ModelConfig(
    vocab_size=50257,
    embed_dim=768,
    num_layers=24,
    num_heads=1,       # unused by Mamba
    context_length=2048,
    dropout=0.0,
)

model = Mamba(
    config,
    d_state=16,    # SSM state dimension
    expand=2,      # inner dim = 2 * embed_dim
    d_conv=4,      # causal conv1d kernel size
)
```

## How It Compares

| Property | Mamba | GPT | LLaMA |
|----------|-------|-----|-------|
| Time complexity | O(n) | O(n^2) | O(n^2) |
| Inference memory | O(1) state | O(n) KV cache | O(n) KV cache |
| Position encoding | Implicit | Learned | RoPE |
| Training parallelism | Parallel scan | Attention matrix | Attention matrix |
| Theoretical expressivity | Selective dynamics | Full attention | Full attention |

## API Reference

::: lmt.layers.ssm.SelectiveSSM
::: lmt.layers.ssm.MambaBlock
::: lmt.models.mamba.Mamba

## References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) -- Gu & Dao (2023)
- [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) -- S4 (Gu et al., 2021)
- [mamba.py](https://github.com/alxndrTL/mamba.py) -- Pure PyTorch reference implementation
