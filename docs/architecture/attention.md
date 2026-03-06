# Attention Mechanisms

LMT implements seven attention variants, each building on the last.

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

## Flash Attention

From Dao et al. (2022). Used in virtually all modern LLMs.

**Key idea**: Standard attention materializes the full $N \times N$ score
matrix, which is memory-bound on GPUs. Flash Attention processes Q and K/V
in **tiles**, using **online softmax** to avoid ever storing the full matrix.

The result is mathematically identical to standard attention, but uses
$O(N)$ HBM memory instead of $O(N^2)$.

```python
from lmt.layers.attention import FlashAttention

config = ModelConfig(embed_dim=256, num_heads=8, ...)
flash = FlashAttention(config, block_size=64)
```

Our implementation is a **pure Python educational version** demonstrating the
tiling algorithm. For production, use PyTorch's `scaled_dot_product_attention`.

See [Flash Attention](flash-attention.md) for the full algorithm walkthrough.

---

## Gated Delta Networks (Linear Attention)

From Yang et al. (2024, ICLR 2025). Used in Qwen3-Next and Kimi Linear.

**Key idea**: Replace softmax attention with a **recurrent state matrix**
\(S \in \mathbb{R}^{d_h \times d_h}\) updated via the gated delta rule.
Causality is inherent in the recurrence -- no mask needed.

### Standard linear attention (naive)

\[
S_t = S_{t-1} + v_t k_t^\top
\]

This accumulates forever, filling the memory with noise.

### Delta rule (targeted correction)

\[
S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^\top
\]

This is one step of gradient descent on \(\|Sk - v\|^2\). It writes
"what should be there minus what's already there" -- a surgical update.

### Gated DeltaNet (with decay)

\[
S_t = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1} k_t) k_t^\top
\]

\[
o_t = S_t q_t
\]

Where \(\alpha_t \in (0, 1]\) is a decay gate (like SSM forgetting) and
\(\beta_t \in (0, 1)\) controls update strength.

```python
from lmt.layers.attention import GatedDeltaNet

config = ModelConfig(embed_dim=256, num_heads=8, ...)
gdn = GatedDeltaNet(config)
```

**Complexity**: \(O(n \cdot d_h^2)\) per layer -- linear in sequence length,
quadratic in head dimension. Wins over \(O(n^2 \cdot d_h)\) softmax attention
when sequences are long relative to head dimension.

**Connection to SSMs**: The decay gate \(\alpha\) IS the SSM decay factor.
Mamba-2 proved that SSMs and linear attention are mathematically dual --
GatedDeltaNet is a concrete instance of this duality.

**Hybrid architectures**: In practice, 3 GatedDeltaNet layers alternate
with 1 full attention layer (3:1 ratio), because the fixed-size state
compresses context lossily.

---

## Structured State Space Duality (SSD)

From Dao & Gu (2024). The theoretical foundation for Mamba-2.

**Key idea**: SSMs with scalar decay and linear attention are
mathematically equivalent -- two different algorithms for the
same matrix-vector product.

### The Duality

Given a selective SSM with scalar \(a_t\):

\[
h_t = a_t h_{t-1} + k_t v_t^\top, \quad o_t = h_t q_t
\]

the output can equivalently be computed as:

\[
Y = (L \odot Q K^\top) V
\]

where \(L\) is a lower-triangular decay mask:

\[
L_{ij} = \prod_{s=j+1}^{i} a_s
\]

When all \(a_t = 1\), \(L\) becomes the standard causal mask and the
model is exactly **causal linear attention**.

```python
from lmt.layers.attention import SSDAttention

config = ModelConfig(embed_dim=256, num_heads=8, ...)
ssd = SSDAttention(config)

# Both produce identical outputs:
y_recurrent = ssd.forward_recurrent(x)
y_quadratic = ssd.forward_quadratic(x)
```

**Our implementation** exposes both forms and verifies they match,
serving as a concrete replication of the paper's core claim.

**Relation to GatedDeltaNet**: SSD is a strict simplification --
no delta rule correction. GatedDeltaNet adds the associative memory
update \(\beta(v - Sk)k^\top\) on top of SSD's decay mechanism.

---

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017) -- multi-head attention
- Ainslie et al., [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245) (2023) -- grouped query attention
- Beltagy et al., [*Longformer: The Long-Document Transformer*](https://arxiv.org/abs/2004.05150) (2020) -- sliding window attention
- DeepSeek-AI, [*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434) (2024) -- multi-head latent attention
- Dao et al., [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135) (2022) -- flash attention
- Yang et al., [*Gated Delta Networks: Improving Mamba2 with Delta Rule*](https://arxiv.org/abs/2412.06464) (2024) -- gated delta networks
- Dao & Gu, [*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*](https://arxiv.org/abs/2405.21060) (2024) -- structured state space duality
