# Architecture Ablation Study

The [architecture comparison](architecture-comparison.md) showed that LLaMA
outperforms GPT by ~11% BPB on WikiText-2. But LLaMA differs from GPT in
**four ways**: RoPE, SwiGLU, GQA, and RMSNorm. Which features drive the
improvement?

## Experiment Design

Starting from the GPT baseline (MHA + GELU + LayerNorm + learned pos),
we add **one LLaMA feature at a time** using the `ConfigurableBlock` system:

| Variant        | Attention | FFN     | Norm      | Position |
|---------------|-----------|---------|-----------|----------|
| GPT baseline  | MHA       | GELU    | LayerNorm | Learned  |
| GPT + RoPE    | MHA       | GELU    | LayerNorm | **RoPE** |
| GPT + SwiGLU  | MHA       | **SwiGLU** | LayerNorm | Learned |
| GPT + GQA     | **GQA**   | GELU    | LayerNorm | Learned  |
| GPT + RMSNorm | MHA       | GELU    | **RMSNorm** | Learned |
| LLaMA (all 4) | **GQA**   | **SwiGLU** | **RMSNorm** | **RoPE** |

All variants use the same model dimensions (64 embed, 4 layers, 4 heads)
and training configuration (500 steps, lr=1e-3, batch=32) on WikiText-2
with character-level tokenization.

!!! tip "ConfigurableBlock"
    The `BlockConfig` dataclass makes this experiment trivial to implement.
    Each variant is just a different combination of string keys:

    ```python
    block_config = BlockConfig(
        attention='gqa',   # or 'mha'
        ffn='swiglu',      # or 'default'
        norm='rmsnorm',    # or 'layernorm'
        rope=rope_module,  # or None
    )
    model = BaseModel(config, block_config=block_config)
    ```

## Hypothesis

SwiGLU and RoPE contribute the most to the BPB improvement, while the
norm type (RMSNorm vs LayerNorm) and attention type (GQA vs MHA) have
smaller effects at toy scale.

## Results

**The hypothesis was wrong.** No single feature drives the improvement.
The effect is almost entirely due to **synergy** between the features.

| Variant        | Best BPB | vs GPT   |
|---------------|----------|----------|
| GPT baseline  | 3.602    | --       |
| GPT + RoPE    | 3.587    | -0.4%   |
| GPT + SwiGLU  | 3.617    | +0.4%   |
| GPT + GQA     | 3.583    | -0.5%   |
| GPT + RMSNorm | 3.607    | +0.1%   |
| **LLaMA (all 4)** | **3.226** | **-10.4%** |

<div class="exp-chart" data-experiment="arch-ablation"></div>

### Individual Feature Contributions

| Feature  | BPB Change | % Change |
|----------|-----------|----------|
| RoPE     | -0.015    | -0.4%   |
| GQA      | -0.019    | -0.5%   |
| RMSNorm  | +0.005    | +0.1%   |
| SwiGLU   | +0.015    | +0.4%   |

**Sum of individual contributions: 0.014 BPB** (essentially zero)

**Full LLaMA improvement: 0.376 BPB** (10.4%)

**Interaction (synergy): +0.362 BPB** -- the features are **26x more
effective together** than the sum of their individual effects.

## Analysis

!!! warning "Surprising Result"
    This is a textbook case of **emergent behavior through feature
    interaction**. Each component contributes almost nothing alone,
    but together they produce a 10% improvement.

### Why does this happen?

The features are **complementary** -- they address different bottlenecks
that only matter when the other bottlenecks are also resolved:

1. **RoPE** provides better position information, but this only helps
   when the FFN can *use* that positional signal effectively (SwiGLU's
   gating helps here)

2. **SwiGLU** enables feature selection through gating, but this only
   helps when the attention layer provides high-quality features to
   select from (GQA's shared KV heads create more diverse features)

3. **GQA** forces the model to learn more generalizable key-value
   representations, but this only pays off when the rest of the
   architecture can exploit the generalization

4. **RMSNorm** is computationally lighter, freeing up "optimization
   budget" for the other components -- but alone, it's functionally
   equivalent to LayerNorm

Think of it like a relay team: replacing one runner doesn't change much,
but replacing all four creates a team that works together.

### Falsification Check

Could this be a training artifact? Several sanity checks:

- The GPT baseline BPB (3.60) matches our full-data result (3.56),
  confirming the reduced dataset is representative
- All variants train stably (no divergence or NaN)
- The LLaMA full result (3.23) matches the architecture comparison
  result for LLaMA (3.17), accounting for the smaller dataset
- Each variant uses the same seed (42) and identical training config

### Implications

1. **Benchmarking individual features is misleading** at toy scale.
   A paper claiming "SwiGLU provides X% improvement" may not replicate
   if tested in isolation rather than as part of a full architecture.

2. **The LLaMA recipe works as a package**, not as a collection of
   independent improvements. This suggests these features were
   co-designed or at least co-tuned.

3. **At larger scale**, individual features likely contribute more
   because the model has enough capacity to exploit each optimization
   independently. This is a toy-scale phenomenon.

## Pairwise Ablation: Finding the "Magic Pair"

The single-feature ablation showed massive synergy. But *where* is the
synergy concentrated? We tested all 6 pairwise combinations to find out.

| Pair | BPB | vs GPT | vs LLaMA |
|------|-----|--------|----------|
| **RoPE + GQA** | **3.140** | **-12.8%** | **-2.7%** |
| GQA + RMSNorm | 3.585 | -0.5% | +11.1% |
| RoPE + RMSNorm | 3.594 | -0.2% | +11.4% |
| SwiGLU + GQA | 3.596 | -0.2% | +11.5% |
| RoPE + SwiGLU | 3.613 | +0.3% | +12.0% |
| SwiGLU + RMSNorm | 3.619 | +0.5% | +12.2% |

<div class="exp-chart" data-experiment="pairwise-ablation"></div>

!!! success "RoPE + GQA is the magic pair"
    RoPE + GQA alone achieves **-12.8% vs GPT** -- actually *better*
    than full LLaMA (-10.4%)! It accounts for **123%** of the full
    LLaMA improvement. All other pairs contribute essentially nothing.

### Why RoPE + GQA?

The synergy between RoPE and GQA makes intuitive sense:

- **GQA** shares key-value heads across query groups. This forces the
  model to learn more generalizable KV representations.
- **RoPE** provides relative position information directly in the
  attention computation. With GQA's shared KV heads, each shared head
  must serve multiple query groups -- RoPE's position signal helps
  disambiguate which tokens each query group should attend to.
- Together, they create a powerful combination: GQA provides efficient,
  shared representations while RoPE provides the position-awareness
  needed to make those shared representations useful.

### Why RoPE + GQA > full LLaMA?

The fact that RoPE + GQA (3.14) outperforms full LLaMA (3.23) suggests
that at this tiny scale, **SwiGLU and RMSNorm actually hurt slightly**
when combined with RoPE + GQA. This could be because:

- SwiGLU has more parameters in the gating mechanism, consuming
  parameter budget that could be used elsewhere at this small scale
- RMSNorm's simplification provides no benefit when LayerNorm's
  mean-centering is useful for such small models
- The interaction effects between 4 features are complex -- 2-way
  synergy doesn't guarantee 4-way synergy

## Running the Experiments

```bash
# Single-feature ablation
uv run python experiments/arch_ablation.py

# Pairwise ablation
.venv/bin/python experiments/pairwise_ablation.py
```

Results are saved to `experiments/results/`.

## Source Code

- [`experiments/arch_ablation.py`](https://github.com/michaelellis003/LMT/blob/main/experiments/arch_ablation.py)
  -- Single-feature ablation
- [`experiments/pairwise_ablation.py`](https://github.com/michaelellis003/LMT/blob/main/experiments/pairwise_ablation.py)
  -- Pairwise ablation
- [`tests/test_arch_ablation.py`](https://github.com/michaelellis003/LMT/blob/main/tests/test_arch_ablation.py)
  -- Tests for single-feature variants
- [`tests/test_pairwise_ablation.py`](https://github.com/michaelellis003/LMT/blob/main/tests/test_pairwise_ablation.py)
  -- Tests for pairwise variants
