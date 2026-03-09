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

## Running the Experiment

```bash
uv run python experiments/arch_ablation.py
```

Results are saved to `experiments/results/arch_ablation/registry.jsonl`.

## Source Code

- [`experiments/arch_ablation.py`](https://github.com/michaelellis003/LMT/blob/main/experiments/arch_ablation.py)
  -- Full ablation experiment
- [`tests/test_arch_ablation.py`](https://github.com/michaelellis003/LMT/blob/main/tests/test_arch_ablation.py)
  -- Tests for all 6 ablation variants
