# Attention Mechanisms

LMT implements four attention variants, each building on the last.

## Multi-Head Attention (MHA)

The original attention mechanism from Vaswani et al. (2017).

**Key idea**: Project input into Q, K, V across multiple heads, compute
scaled dot-product attention independently per head, then concatenate.

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

```python
from lmt.layers.attention import MultiHeadAttention
from lmt.models.config import ModelConfig

config = ModelConfig(embed_dim=256, num_heads=8, ...)
attn = MultiHeadAttention(config)
```

**KV cache size**: \(2 \times n_{\text{heads}} \times d_{\text{head}}\) per token.

---

## Grouped Query Attention (GQA)

From Ainslie et al. (2023). Used in LLaMA 2, LLaMA 3, Mistral.

**Key idea**: Share K and V projections across groups of query heads.
If `num_kv_heads < num_heads`, multiple Q heads attend to the same K,V.

\[
\text{num\_groups} = \frac{n_{\text{heads}}}{n_{\text{kv\_heads}}}
\]

Special cases:

- `num_kv_heads == num_heads`: standard MHA
- `num_kv_heads == 1`: Multi-Query Attention (MQA)

```python
from lmt.layers.attention import GroupedQueryAttention

config = ModelConfig(embed_dim=256, num_heads=8, num_kv_heads=4, ...)
gqa = GroupedQueryAttention(config)  # 2x KV cache reduction
```

**Why it works**: KV representations are more redundant across heads
than Q representations. Sharing them barely hurts quality while halving
(or more) the KV cache.

---

## Sliding Window Attention

From Beltagy et al. (2020), used in Mixtral.

**Key idea**: Each token only attends to the previous \(w\) tokens
instead of the full sequence. Reduces attention from \(O(n^2)\) to \(O(nw)\).

```python
from lmt.layers.attention import SlidingWindowAttention

config = ModelConfig(embed_dim=256, num_heads=8, window_size=256, ...)
swa = SlidingWindowAttention(config)
```

**Why it works**: In practice, most attention weight is concentrated on
nearby tokens. With multiple layers, information can still propagate
across the full sequence -- layer \(l\) can see \(l \times w\) tokens back.

---

## Multi-Head Latent Attention (MLA)

From DeepSeek-AI (2024). The most aggressive KV cache compression.

**Key idea**: Compress KV into a shared low-rank latent, and use
**decoupled RoPE** to separate content from positional information.

### Content Path (compressed)

\[
c_t^{KV} = x_t W^{DKV}, \quad k_C = c^{KV} W^{UK}, \quad v = c^{KV} W^{UV}
\]

\[
c_t^Q = x_t W^{DQ}, \quad q_C = c^Q W^{UQ}
\]

### Positional Path (decoupled RoPE)

\[
q_R = \text{RoPE}(c^Q \cdot W^{QR}), \quad k_R = \text{RoPE}(x_t \cdot W^{KR})
\]

Note the **asymmetry**: \(q_R\) derives from the query latent \(c^Q\),
but \(k_R\) derives from the raw input \(x_t\). This is critical for
the weight absorption trick to work.

### Final Attention

\[
Q = [q_C; q_R], \quad K = [k_C; k_R], \quad \text{Attn}(Q, K, V)
\]

At inference, only \(c_t^{KV}\) and \(k_R\) are cached -- up to 93%
memory reduction vs standard MHA.

```python
from lmt.layers.attention import MultiHeadLatentAttention

config = ModelConfig(embed_dim=256, num_heads=8, ...)
mla = MultiHeadLatentAttention(
    config,
    kv_compress_dim=64,   # much smaller than 256
    q_compress_dim=64,
    rope_dim=16,          # small positional vectors
)
```

**Why decoupled RoPE?** Standard RoPE applied to compressed vectors
breaks the low-rank structure -- the rotation matrix gets "stuck in
the middle" and prevents weight absorption. By separating content and
position into independent paths, MLA gets both compression efficiency
and positional encoding.
