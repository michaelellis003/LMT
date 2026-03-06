# RoPE Explorer

An interactive visualization of **Rotary Position Embedding (RoPE)**, the
positional encoding used in LLaMA, Mixtral, and our MLA implementation.

## How RoPE Works

RoPE encodes position by **rotating pairs of dimensions** in the embedding
space. Each dimension pair `(x1, x2)` rotates at a different frequency --
low-indexed pairs rotate slowly (capturing long-range structure), while
high-indexed pairs rotate quickly (capturing local patterns).

The magic: when you compute `Q @ K^T`, the rotation cancels out to depend
only on the **relative distance** between tokens, not their absolute positions.

## Interactive Visualization

<div class="rope-viz" data-d-model="64" data-base="10000" data-max-pos="128"></div>

### What to explore

- **Position slider**: Watch how the vector traces a circle as position increases.
  Low dimension pairs barely move; high pairs spin rapidly.
- **Dim pair slider**: Compare dimension pair 0 (slowest rotation) with pair 31
  (fastest). This multi-scale structure is what gives RoPE its power.
- **Base slider**: Higher base = slower rotation overall = longer effective
  context window. This is the principle behind YaRN and other context extension
  methods.
- **Animate button**: Watch the rotation in real-time. Notice how each dimension
  pair creates a unique "fingerprint" of position.

### Panel guide

| Panel | Shows |
|-------|-------|
| **Rotation Circle** | A single dimension pair `(x1, x2)` rotating with position. The dashed line shows the current angle `m * theta_i`. |
| **Frequency Spectrum** | All dimension pairs' frequencies. Green = slow (low-freq), red = fast (high-freq). Arcs above bars show cumulative rotation. |
| **Dot Product Decay** | Average cosine similarity vs. relative distance. Peak at 0 = nearby tokens are most similar. This is the relative position property in action. |

### Connection to LMT code

This visualization mirrors our
[`RoPE`](../api/layers.md) implementation:

```python
# From src/lmt/layers/positional/rope.py
freqs = 1.0 / (base ** (torch.arange(0, half_d).float() / half_d))
angles = torch.outer(positions, freqs)  # [seq_len, half_d]

# Each position m rotates dim pair i by angle m * theta_i
x1 * cos - x2 * sin, x2 * cos + x1 * sin
```

### Design insights

**Why geometric frequency spacing?** The base-exponent formula creates
frequencies that span several orders of magnitude. This means:

- Low-frequency pairs change slowly -- they distinguish positions far apart
- High-frequency pairs change rapidly -- they distinguish nearby positions
- Together, they form a **multi-resolution** position encoding

**Why does the dot product decay?** When you average `cos(delta * theta_i)`
across many frequencies, constructive interference only happens at `delta = 0`.
At larger distances, the different frequencies destructively interfere,
creating natural position-dependent attention decay.

**Why a base of 10000?** The original RoPE paper found this gives a good
balance: the slowest-rotating pair completes one revolution in ~10000 positions,
while the fastest pair rotates nearly every position. Larger bases extend
this range (used for long-context models).

## References & Further Reading

- Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) (2021) -- the original RoPE paper
- EleutherAI, [*Rotary Embeddings: A Relative Revolution*](https://blog.eleuther.ai/rotary-embeddings/) -- excellent intuitive explanation
- Brenndoerfer, [*Rotary Position Embedding (RoPE)*](https://mbrenndoerfer.com/writing/rotary-position-embedding-rope-transformers) -- static visualizations of rotation and frequency patterns

Our visualization is an original interactive implementation built specifically
for LMT's educational docs. The three-panel layout (rotation circle, frequency
spectrum, dot-product decay) with live sliders and animation is our own design,
though the underlying concepts are well-covered in the resources above.
