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

**Rationale**:

- **SwiGLU** introduces a gating mechanism that enables feature selection,
  fundamentally changing the FFN's computational capacity
- **RoPE** provides relative position information that scales better than
  learned absolute embeddings
- **RMSNorm** is computationally simpler but functionally similar to
  LayerNorm at this scale
- **GQA** is primarily an efficiency feature (fewer KV heads) that
  shouldn't affect quality with only 4 heads

## Feature Contribution Analysis

When the full ablation completes, the analysis compares:

1. **Individual contributions**: How much does each feature improve BPB
   over the GPT baseline?
2. **Sum of individual contributions** vs **full LLaMA delta**: If the sum
   is less than the full delta, the features have **synergy** (they work
   better together). If the sum exceeds the delta, there is **overlap**
   (features partially substitute for each other).

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
