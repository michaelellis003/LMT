# Positional Encoding

Transformers process all tokens in parallel -- they need positional
encoding to know token order. LMT implements two modern approaches:
**RoPE** (rotation-based) and **ALiBi** (bias-based).

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
    max_seq_len=2048,
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

## ALiBi (Attention with Linear Biases)

From Press et al. (2022). Used in BLOOM, MPT.

**Key idea**: Don't add any positional information to token embeddings.
Instead, add a **static linear bias** to attention scores that penalizes
distant query-key pairs:

\[
\text{Attention}(Q, K, V) = \text{softmax}\!\left(
    \frac{Q K^\top}{\sqrt{d_k}} + B
\right) V
\]

where $B_{h,i,j} = -m_h \cdot |i - j|$.

### Head-Specific Slopes

Each attention head gets a different slope from a geometric sequence:

\[
m_h = 2^{-\frac{8}{n} \cdot h}, \quad h = 1, 2, \ldots, n
\]

For 8 heads, the slopes are $\frac{1}{2}, \frac{1}{4}, \frac{1}{8},
\ldots, \frac{1}{256}$. This means:

- **Head 1** (slope = 0.5): strong recency bias, focuses on nearby tokens
- **Head 8** (slope ≈ 0.004): mild bias, spreads attention broadly

Different heads naturally specialize in different context ranges.

```python
from lmt.layers.positional import ALiBi

alibi = ALiBi(num_heads=8, max_seq_len=2048)

# Inside attention, after computing QK^T:
attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
attn_scores = attn_scores + alibi.get_bias(seq_len)
attn_weights = torch.softmax(attn_scores, dim=-1)
```

### Why ALiBi Extrapolates

Because the bias is a simple linear function of distance (not a learned
embedding indexed by position), it naturally extends to longer sequences.
A model trained on 1024 tokens can infer on 2048+ tokens — the bias at
position 1500 is just $-m_h \cdot 1500$, which is well-defined even if
the model never saw position 1500 during training.

## RoPE vs ALiBi

| Property | RoPE | ALiBi |
|----------|------|-------|
| How it works | Rotates Q, K vectors | Biases attention scores |
| Parameters | 0 | 0 |
| Where applied | Inside attention (Q, K only) | After QK^T dot product |
| Position info | Direction-aware (rotation) | Distance only (scalar) |
| Extrapolation | Moderate | Strong |
| Used in | LLaMA, Mistral, GPT-NeoX | BLOOM, MPT |

Both encode **relative** position — the attention between tokens depends
on their distance, not their absolute positions. RoPE is more expressive
(rotation captures directional relationships), while ALiBi is simpler
and extrapolates better to unseen sequence lengths.

## References

- Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) (2021)
- Press et al., [*Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*](https://arxiv.org/abs/2108.12409) (2022)
