"""Architecture ablation study: isolating RoPE, SwiGLU, GQA, RMSNorm.

Starting from the GPT baseline (MHA + GELU + LayerNorm + learned pos),
we add one LLaMA feature at a time to measure each contribution to
the 11% BPB improvement we observed on WikiText-2.

Variants tested:
  1. GPT baseline       (MHA + GELU + LayerNorm + learned pos)
  2. GPT + RoPE         (MHA + GELU + LayerNorm + RoPE)
  3. GPT + SwiGLU       (MHA + SwiGLU + LayerNorm + learned pos)
  4. GPT + GQA          (GQA + GELU + LayerNorm + learned pos)
  5. GPT + RMSNorm      (MHA + GELU + RMSNorm + learned pos)
  6. LLaMA (all four)   (GQA + SwiGLU + RMSNorm + RoPE)

Hypothesis: SwiGLU and RoPE contribute the most, while the norm and
attention type have smaller effects at toy scale.

Usage::

    uv run python experiments/arch_ablation.py

Results are saved to experiments/results/arch_ablation/
"""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.data.hf_datasets import download_wikitext2
from lmt.data.text_dataset import TextDataset
from lmt.eval.bpb import compute_bpb
from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.layers.positional import RoPE
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig
from lmt.research.experiment import (
    ExperimentResult,
    ExperimentRunConfig,
    MetricTracker,
)
from lmt.research.registry import ExperimentRegistry
from lmt.tokenizer.char import CharTokenizer
from lmt.training.profiler import profile_model
from lmt.training.reproducibility import set_seed
from lmt.training.train_config import TrainConfig
from lmt.training.train_loop import train_loop

# --- Configuration ---

BASE_CONFIG = ModelConfig(
    embed_dim=64,
    num_heads=4,
    num_kv_heads=2,
    num_layers=4,
    ffn_hidden_dim=128,
    vocab_size=256,  # Overridden by tokenizer
    context_length=128,
    tie_weights=True,
    qk_norm=True,
    dropout=0.0,
)

TRAIN_CONFIG = TrainConfig(
    lr=1e-3,
    total_steps=500,
    warmup_steps=50,
    weight_decay=0.1,
    max_grad_norm=1.0,
)

BATCH_SIZE = 32
EVAL_INTERVAL = 100
SEED = 42
OUTPUT_DIR = Path('experiments/results/arch_ablation')


@dataclass
class AblationVariant:
    """Defines one ablation variant."""

    name: str
    attention: str
    ffn: str
    norm: str
    use_rope: bool
    learned_pos: bool


# The six variants: GPT baseline → add one feature → full LLaMA
VARIANTS = [
    AblationVariant(
        name='gpt_baseline',
        attention='mha',
        ffn='default',
        norm='layernorm',
        use_rope=False,
        learned_pos=True,
    ),
    AblationVariant(
        name='gpt+rope',
        attention='mha',
        ffn='default',
        norm='layernorm',
        use_rope=True,
        learned_pos=False,
    ),
    AblationVariant(
        name='gpt+swiglu',
        attention='mha',
        ffn='swiglu',
        norm='layernorm',
        use_rope=False,
        learned_pos=True,
    ),
    AblationVariant(
        name='gpt+gqa',
        attention='gqa',
        ffn='default',
        norm='layernorm',
        use_rope=False,
        learned_pos=True,
    ),
    AblationVariant(
        name='gpt+rmsnorm',
        attention='mha',
        ffn='default',
        norm='rmsnorm',
        use_rope=False,
        learned_pos=True,
    ),
    AblationVariant(
        name='llama_full',
        attention='gqa',
        ffn='swiglu',
        norm='rmsnorm',
        use_rope=True,
        learned_pos=False,
    ),
]


def build_model(variant: AblationVariant, config: ModelConfig) -> BaseModel:
    """Build a model for the given ablation variant."""
    rope = None
    if variant.use_rope:
        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(d_model=head_dim, max_seq_len=config.context_length)

    block_config = BlockConfig(
        attention=variant.attention,
        ffn=variant.ffn,
        norm=variant.norm,
        rope=rope,
    )

    return BaseModel(
        config,
        block_config=block_config,
        learned_pos_embed=variant.learned_pos,
    )


def loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Next-token prediction loss."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    return f.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def run_single(
    variant: AblationVariant,
    config: ModelConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> dict:
    """Train one variant and return results."""
    set_seed(SEED)

    model = build_model(variant, config)
    profile = profile_model(model)

    print(f'\n  {variant.name.upper()}')
    print(
        f'    attn={variant.attention}, ffn={variant.ffn}, '
        f'norm={variant.norm}, '
        f'pos={"rope" if variant.use_rope else "learned"}'
    )
    print(f'    Params: {profile.human_readable_params()}')

    eval_results: list[tuple[int, float]] = []

    def eval_fn(m: nn.Module, step: int) -> dict[str, float]:
        m.eval()
        bpb = compute_bpb(m, val_data, bytes_per_token=1.0)
        eval_results.append((step, bpb))
        print(f'    Step {step:4d} | val_bpb={bpb:.4f}')
        return {'val_bpb': bpb}

    start = time.time()
    state = train_loop(
        model=model,
        train_data=train_data,
        loss_fn=loss_fn,
        config=TRAIN_CONFIG,
        batch_size=BATCH_SIZE,
        eval_fn=eval_fn,
        eval_interval=EVAL_INTERVAL,
    )
    elapsed = time.time() - start

    final_bpb = eval_results[-1][1] if eval_results else None
    best_bpb = min(v for _, v in eval_results) if eval_results else None

    print(f'    Final loss: {state.final_loss:.4f}')
    print(f'    Best BPB:   {best_bpb:.4f}' if best_bpb else '')
    print(f'    Time:       {elapsed:.1f}s')

    return {
        'name': variant.name,
        'params': profile.total_params,
        'final_loss': state.final_loss,
        'final_bpb': final_bpb,
        'best_bpb': best_bpb,
        'elapsed_s': elapsed,
        'losses': state.losses,
        'eval_results': eval_results,
    }


def main() -> None:
    """Run architecture ablation study."""
    print('=' * 60)
    print('Architecture Ablation Study')
    print('=' * 60)
    print(
        f'Config: embed_dim={BASE_CONFIG.embed_dim}, '
        f'layers={BASE_CONFIG.num_layers}, '
        f'steps={TRAIN_CONFIG.total_steps}'
    )
    print(f'Variants: {len(VARIANTS)}')

    # Build tokenizer and load data
    print('\n  Loading WikiText-2...')
    train_texts = download_wikitext2(split='train')
    val_texts = download_wikitext2(split='validation')
    all_text = '\n'.join(train_texts + val_texts)
    tokenizer = CharTokenizer.from_text(all_text)
    print(f'  Vocab size: {tokenizer.vocab_size}')

    config = ModelConfig(
        embed_dim=BASE_CONFIG.embed_dim,
        num_heads=BASE_CONFIG.num_heads,
        num_kv_heads=BASE_CONFIG.num_kv_heads,
        num_layers=BASE_CONFIG.num_layers,
        ffn_hidden_dim=BASE_CONFIG.ffn_hidden_dim,
        vocab_size=tokenizer.vocab_size,
        context_length=BASE_CONFIG.context_length,
        tie_weights=BASE_CONFIG.tie_weights,
        qk_norm=BASE_CONFIG.qk_norm,
        dropout=BASE_CONFIG.dropout,
    )

    train_dataset = TextDataset(
        train_texts, tokenizer, context_length=config.context_length
    )
    val_dataset = TextDataset(
        val_texts, tokenizer, context_length=config.context_length
    )
    train_data = torch.stack(
        [train_dataset[i] for i in range(len(train_dataset))]
    )
    val_limit = min(500, len(val_dataset))
    val_data = torch.stack([val_dataset[i] for i in range(val_limit)])
    print(f'  Train: {len(train_dataset)} sequences')
    print(f'  Val:   {val_limit} sequences')

    # Run all variants
    results = []
    registry = ExperimentRegistry(OUTPUT_DIR / 'registry.jsonl')

    for variant in VARIANTS:
        result = run_single(variant, config, train_data, val_data)
        results.append(result)

        tracker = MetricTracker()
        for step, bpb in result['eval_results']:
            tracker.log('val_bpb', bpb, step=step)

        exp_config = ExperimentRunConfig(
            name=f'ablation_{variant.name}',
            lr=TRAIN_CONFIG.lr,
            train_steps=TRAIN_CONFIG.total_steps,
            batch_size=BATCH_SIZE,
        )
        exp_result = ExperimentResult(
            config=exp_config,
            metrics=tracker,
            final_loss=result['final_loss'],
        )
        registry.add(exp_result)

    # Summary table
    print('\n' + '=' * 70)
    print('ABLATION RESULTS (WikiText-2, character-level)')
    print('=' * 70)
    header = (
        f'{"Variant":<18} {"Params":>8} {"BPB":>8} {"vs GPT":>8} {"Time":>6}'
    )
    print(header)
    print('-' * 70)

    gpt_bpb = next(
        r['best_bpb'] for r in results if r['name'] == 'gpt_baseline'
    )

    for r in results:
        bpb = r['best_bpb']
        if bpb and gpt_bpb:
            delta = bpb - gpt_bpb
            delta_pct = delta / gpt_bpb * 100
            delta_str = f'{delta_pct:+.1f}%'
        else:
            delta_str = '-'
        bpb_str = f'{bpb:.4f}' if bpb else '-'
        print(
            f'{r["name"]:<18} {r["params"]:>8,} '
            f'{bpb_str:>8} {delta_str:>8} '
            f'{r["elapsed_s"]:>5.1f}s'
        )

    # Feature contribution analysis
    print('\n' + '-' * 70)
    print('FEATURE CONTRIBUTIONS (BPB reduction from GPT baseline)')
    print('-' * 70)

    if gpt_bpb:
        contributions = []
        for r in results:
            if r['name'] != 'gpt_baseline' and r['name'] != 'llama_full':
                bpb = r['best_bpb']
                if bpb:
                    reduction = gpt_bpb - bpb
                    feature = r['name'].replace('gpt+', '')
                    contributions.append((feature, reduction))

        contributions.sort(key=lambda x: x[1], reverse=True)
        for feature, reduction in contributions:
            pct = reduction / gpt_bpb * 100
            bar = '#' * int(pct * 2)
            print(f'  {feature:<12} {reduction:+.4f} BPB ({pct:+.1f}%) {bar}')

        llama_bpb = next(
            r['best_bpb'] for r in results if r['name'] == 'llama_full'
        )
        if llama_bpb:
            total = gpt_bpb - llama_bpb
            sum_individual = sum(c[1] for c in contributions)
            print(f'\n  Sum of individual: {sum_individual:.4f}')
            print(f'  Full LLaMA delta:  {total:.4f}')
            if sum_individual > 0:
                interaction = total - sum_individual
                print(
                    f'  Interaction:       {interaction:+.4f} '
                    f'({"synergy" if interaction > 0 else "overlap"})'
                )

    print(f'\nResults saved to: {OUTPUT_DIR / "registry.jsonl"}')
    print(registry.summary(metric='val_bpb'))


if __name__ == '__main__':
    main()
