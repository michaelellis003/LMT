# The Transformer Family Tree

How did we get from "Attention Is All You Need" to modern LLMs like LLaMA and
DeepSeek? This page traces the evolution of transformer architectures,
highlighting the key innovations at each step.

## Evolution Overview

```mermaid
graph TD
    A["Transformer<br/>(Vaswani et al., 2017)"] --> B["GPT<br/>(Radford et al., 2018)"]
    A --> C["BERT<br/>(Devlin et al., 2018)"]
    B --> D["GPT-2/3<br/>(Radford/Brown, 2019-20)"]
    D --> E["LLaMA<br/>(Touvron et al., 2023)"]
    E --> F["Mixtral<br/>(Jiang et al., 2024)"]
    E --> G["LLaMA 2/3<br/>(Touvron et al., 2023-24)"]
    D --> H["DeepSeek-V2/V3<br/>(DeepSeek, 2024)"]

    style A fill:#6a1b9a,color:#fff
    style B fill:#4527a0,color:#fff
    style D fill:#4527a0,color:#fff
    style E fill:#1565c0,color:#fff
    style F fill:#00838f,color:#fff
    style G fill:#1565c0,color:#fff
    style H fill:#2e7d32,color:#fff
    style C fill:#757575,color:#fff
```

!!! note "Decoder-only focus"
    LMT focuses on **decoder-only** (autoregressive) transformers used for
    language modeling. BERT and encoder-decoder models are shown for context
    but are not implemented in the library.

## Key Innovations Timeline

```mermaid
timeline
    title Transformer Architecture Innovations
    2017 : Attention Is All You Need
         : Multi-Head Attention
         : Positional Encoding (sinusoidal)
    2018 : GPT — Decoder-only pretraining
         : Learned positional embeddings
    2019 : GPT-2 — Scaling up
         : Pre-norm (LayerNorm before sublayer)
    2020 : GPT-3 — Few-shot learning
         : Multi-Query Attention (MQA)
    2022 : RoPE — Rotary Position Encoding
         : Flash Attention
         : Grouped Query Attention (GQA)
    2023 : LLaMA — RMSNorm + SwiGLU + RoPE + GQA
         : Mixtral — Mixture of Experts (MoE)
    2024 : DeepSeek-V2 — Multi-Head Latent Attention
         : Decoupled RoPE for MLA
```

## What Changed and Why

### 1. Attention: MHA → GQA → MLA

The biggest bottleneck in transformer inference is the **KV cache** — storing
key and value tensors for all previous tokens. Each innovation reduces this
cost:

| Mechanism | KV Heads | Cache per Token | Quality |
|-----------|----------|-----------------|---------|
| **MHA** | `num_heads` | `2 * num_heads * head_dim` | Baseline |
| **MQA** | 1 | `2 * head_dim` | Slight degradation |
| **GQA** | `num_kv_heads` | `2 * num_kv_heads * head_dim` | Near-MHA |
| **MLA** | Compressed | `compress_dim + rope_dim` | Near-MHA |

```mermaid
graph LR
    subgraph "MHA (2017)"
        Q1["Q₁"] --> K1["K₁"]
        Q2["Q₂"] --> K2["K₂"]
        Q3["Q₃"] --> K3["K₃"]
        Q4["Q₄"] --> K4["K₄"]
    end

    subgraph "GQA (2022)"
        Q5["Q₁"] --> K5["K₁₋₂"]
        Q6["Q₂"] --> K5
        Q7["Q₃"] --> K6["K₃₋₄"]
        Q8["Q₄"] --> K6
    end

    subgraph "MLA (2024)"
        Q9["Q₁"] --> C["c_KV<br/>(compressed)"]
        Q10["Q₂"] --> C
        Q11["Q₃"] --> C
        Q12["Q₄"] --> C
    end
```

??? info "How MLA compression works"
    MLA compresses the full KV representation into a small latent vector:

    1. **Compress**: `c_KV = x @ W_DKV` (project to low-rank space)
    2. **Cache**: Store only `c_KV` (much smaller than full K, V)
    3. **Decompress**: `K = c_KV @ W_UK`, `V = c_KV @ W_UV`

    The key insight: K and V information is highly redundant across heads,
    so a shared low-rank factorization captures most of the signal.

### 2. Normalization: Post-norm → Pre-norm, LayerNorm → RMSNorm

```mermaid
graph TD
    subgraph "Post-norm (Original Transformer)"
        I1["Input"] --> A1["Attention"]
        A1 --> R1["+ Residual"]
        R1 --> N1["LayerNorm"]
        N1 --> O1["Output"]
        I1 --> R1
    end

    subgraph "Pre-norm (LLaMA)"
        I2["Input"] --> N2["RMSNorm"]
        N2 --> A2["Attention"]
        A2 --> R2["+ Residual"]
        R2 --> O2["Output"]
        I2 --> R2
    end
```

**Why pre-norm?** In post-norm, gradients must flow through the normalization
layer on the residual path. Pre-norm leaves the residual stream "clean" —
gradients flow directly through addition, making deep networks easier to train.

**Why RMSNorm over LayerNorm?** RMSNorm drops mean-centering (only normalizes
by root-mean-square). This is ~15% faster with no quality loss — the learned
scale parameter can compensate for any mean shift.

### 3. Activation: GELU → SwiGLU

Standard FFN: `FFN(x) = W₂ · GELU(W₁ · x)`

SwiGLU FFN: `SwiGLU(x) = W₂ · (SiLU(W_g · x) ⊙ W₁ · x)`

The **gating mechanism** `SiLU(W_g · x)` learns *which features to pass
through* rather than applying a fixed nonlinearity. This is more expressive —
the gate can selectively amplify or suppress different feature dimensions.

!!! tip "Parameter count trick"
    SwiGLU uses 3 weight matrices instead of 2, but compensates by using
    `hidden_dim = ⅔ × 4d` instead of `4d`, keeping total parameters similar.

### 4. Position Encoding: Learned → RoPE

```mermaid
graph LR
    subgraph "Learned (GPT)"
        P1["position → embedding table → add to input"]
    end

    subgraph "RoPE (LLaMA)"
        P2["position → rotation matrix → multiply Q,K inside attention"]
    end
```

**Key advantage of RoPE**: The dot product `q · k` depends only on the
*relative* position `m - n`, not absolute positions. This means the model
generalizes better to sequences longer than those seen during training.

RoPE applies rotations in 2D subspaces of each attention head:

$$
\text{RoPE}(x, m) = \begin{pmatrix} x_1 \cos m\theta - x_2 \sin m\theta \\ x_2 \cos m\theta + x_1 \sin m\theta \end{pmatrix}
$$

where $m$ is the position and $\theta_i = \text{base}^{-2i/d}$ controls the
frequency for each dimension pair.

## Models in LMT

Here's how each model in the library maps to these innovations:

| Model | Norm | Attention | FFN | Position | Init |
|-------|------|-----------|-----|----------|------|
| [GPT](../models/gpt.md) | Post-norm LayerNorm | MHA | GELU FFN | Learned | Standard |
| [LLaMA](../models/llama.md) | Pre-norm RMSNorm | GQA | SwiGLU | RoPE | Scaled residual |
| [Mixtral](../models/mixtral.md) | Pre-norm RMSNorm | GQA + SWA | MoE (SwiGLU) | RoPE | Scaled residual |

All models can be built using the [`ConfigurableBlock`](../api/layers.md)
system, which lets you mix and match these components via string keys.

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The original transformer
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU
