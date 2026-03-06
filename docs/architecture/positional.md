# Positional Encoding

Transformers process all tokens in parallel -- they need positional
encoding to know token order. LMT implements the modern approach.

## Rotary Position Embedding (RoPE)

From Su et al. (2021). Used in LLaMA, Mistral, Mixtral, GPT-NeoX.

**Key idea**: Encode position by rotating Q and K vectors in 2D
subspaces. The dot product between rotated vectors depends only on
their relative position, not absolute:

\[
\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)
\]

### How It Works

Split each head's \(d\)-dimensional vector into \(d/2\) pairs.
Each pair is rotated by an angle proportional to position:

\[
\theta_i = 10000^{-2i/d}
\]

At position \(m\), pair \(i\) is rotated by \(m \cdot \theta_i\):

\[
\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} =
\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
\]

```python
from lmt.layers.positional import RoPE

rope = RoPE(
    d_model=64,           # head_dim, not full embed_dim
    context_length=2048,
)

# Applied inside attention, after Q/K projection:
q_rot, k_rot = rope.apply_rotary_emb(q, k)
```

### Why Not Learned Positional Embeddings?

| Property | Learned | RoPE |
|----------|---------|------|
| Parameters | \(L \times d\) | 0 |
| Relative position | No | Yes |
| Extrapolation | Poor | Better |
| Efficiency | Lookup | Elementwise multiply |

RoPE has no learnable parameters -- the cos/sin values are precomputed
as buffers. It naturally encodes relative position through the rotation
property, and it generalizes better to sequence lengths not seen during
training.

### Implementation Detail

LMT precomputes cos/sin caches at init time for the full context length.
During forward, it slices to the actual sequence length:

```python
cos = self.cos_cache[:seq_len]  # [seq_len, d_model]
sin = self.sin_cache[:seq_len]
```

This avoids recomputing the same values every forward pass.
