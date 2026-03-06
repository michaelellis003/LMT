# Architecture Overview

LMT implements a progression of transformer architectures, from the
original attention mechanism to modern designs like LLaMA and Mixtral.

## The Transformer Building Blocks

Every transformer model in LMT is built from four types of components:

```
Input Tokens
    |
    v
[Token Embedding]
    |
    v
+----------------------------+
| Transformer Block (x N)   |
|                            |
|  [Normalization] -------+  |
|       |                 |  |
|  [Attention]            |  |
|       |                 |  |
|       + <--- residual --+  |
|       |                    |
|  [Normalization] -------+  |
|       |                 |  |
|  [Feed-Forward]         |  |
|       |                 |  |
|       + <--- residual --+  |
+----------------------------+
    |
    v
[Final Norm]
    |
    v
[Output Head] -> Logits
```

### 1. Normalization

Stabilizes training by normalizing activations. LMT implements:

- **RMSNorm**: Simpler and faster than LayerNorm -- uses root mean
  square instead of mean + variance. Used in LLaMA, Mixtral.

### 2. Attention (Sequence Mixing)

The core mechanism that lets tokens attend to each other:

- **Multi-Head Attention**: The original from "Attention Is All You Need"
- **Grouped Query Attention**: Fewer KV heads for efficiency (LLaMA 2+)
- **Sliding Window Attention**: Local attention for long sequences (Mixtral)
- **Multi-Head Latent Attention**: Compressed KV cache (DeepSeek-V2)

### 3. Feed-Forward (Channel Mixing)

Processes each token independently through a nonlinear transform:

- **SwiGLU**: Gated FFN with Swish activation (LLaMA, Mixtral)
- **Mixture of Experts**: Sparse routing to specialized FFN experts (Mixtral)

### 4. Positional Encoding

Injects position information so the model knows token order:

- **RoPE**: Rotary embeddings that encode relative positions through
  rotation in the complex plane. No learnable parameters.

## Pre-Norm vs Post-Norm

LMT's modern architectures (LLaMA, Mixtral) use **pre-normalization**:

```
x = x + Attention(Norm(x))   # norm before attention
x = x + FFN(Norm(x))         # norm before FFN
```

This is more stable during training than the original post-norm design
(`x = Norm(x + Attention(x))`), especially for deep models.

## Design Philosophy

All sequence mixers share a common interface:

```python
def forward(self, x: Tensor) -> Tensor:
    """
    Args:
        x: [batch, seq_len, d_model]
    Returns:
        [batch, seq_len, d_model]
    """
```

This makes layers interchangeable -- you can swap GQA for sliding
window attention without changing anything else.

Auxiliary outputs (like MoE load balancing loss) are stored as module
attributes rather than returned, keeping the interface clean:

```python
moe = MoEFeedForward(d_model=256, num_experts=8, top_k=2)
out = moe(x)             # clean return: just the tensor
loss = moe.aux_loss      # auxiliary output on attribute
```
