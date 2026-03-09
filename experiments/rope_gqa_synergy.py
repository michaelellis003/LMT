"""RoPE + GQA synergy analysis experiment.

Investigates why RoPE and GQA have synergistic effects on language
modeling at small scale. Tests the interaction between position
encoding type and attention head structure.

Hypotheses tested:
  H1 (Complementary bottlenecks): GQA's shared K/V + RoPE's
      position info are jointly necessary — neither alone helps.
  H2 (Position encoding specificity): The synergy is RoPE-specific,
      not just "any position encoding + GQA."
  H3 (GQA ratio matters): The optimal KV sharing ratio depends on
      whether RoPE is present.

Experiments:
  1. Position × Attention matrix (3×3 = 9 variants)
     Position: {learned, rope, none}
     Attention: {mha, gqa, mqa}

  2. GQA ratio sweep with/without RoPE
     KV heads: {1, 2, 4} × {rope, learned}

Usage::

    python experiments/rope_gqa_synergy.py

Results saved to experiments/results/rope_gqa_synergy.json
"""

import json
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
from lmt.tokenizer.char import CharTokenizer
from lmt.training.profiler import profile_model
from lmt.training.reproducibility import set_seed
from lmt.training.train_config import TrainConfig
from lmt.training.train_loop import train_loop

# --- Configuration ---

SEED = 42
BATCH_SIZE = 32
TRAIN_STEPS = 500
MAX_TRAIN_SEQS = 2000
MAX_VAL_SEQS = 200
MAX_ITEMS = 5000
RESULTS_DIR = Path('experiments/results')


@dataclass
class Variant:
    """One experimental variant."""

    name: str
    attention: str  # mha, gqa, mqa
    use_rope: bool
    learned_pos: bool
    num_kv_heads: int | None  # None = num_heads (MHA)


# Experiment 1: Position × Attention matrix
# Position: learned, rope, none
# Attention: mha (kv=4), gqa (kv=2), mqa (kv=1)
MATRIX_VARIANTS = [
    # Learned position + {MHA, GQA, MQA}
    Variant('learned+mha', 'mha', False, True, None),
    Variant('learned+gqa', 'gqa', False, True, 2),
    Variant('learned+mqa', 'gqa', False, True, 1),
    # RoPE + {MHA, GQA, MQA}
    Variant('rope+mha', 'mha', True, False, None),
    Variant('rope+gqa', 'gqa', True, False, 2),
    Variant('rope+mqa', 'gqa', True, False, 1),
    # No position encoding + {MHA, GQA, MQA}
    Variant('none+mha', 'mha', False, False, None),
    Variant('none+gqa', 'gqa', False, False, 2),
    Variant('none+mqa', 'gqa', False, False, 1),
]

# Experiment 2: GQA ratio sweep (kv_heads=1,2,4)
# with RoPE and with learned pos
RATIO_VARIANTS = [
    # RoPE + different KV ratios
    Variant('rope+kv1', 'gqa', True, False, 1),
    Variant('rope+kv2', 'gqa', True, False, 2),
    Variant('rope+kv4', 'mha', True, False, None),
    # Learned + different KV ratios
    Variant('learned+kv1', 'gqa', False, True, 1),
    Variant('learned+kv2', 'gqa', False, True, 2),
    Variant('learned+kv4', 'mha', False, True, None),
]


def build_model(
    variant: Variant,
    config: ModelConfig,
) -> BaseModel:
    """Build a model for the given variant."""
    rope = None
    if variant.use_rope:
        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(
            d_model=head_dim,
            max_seq_len=config.context_length,
        )

    block_config = BlockConfig(
        attention=variant.attention,
        ffn='default',
        norm='layernorm',
        rope=rope,
    )

    return BaseModel(
        config,
        block_config=block_config,
        learned_pos_embed=variant.learned_pos,
    )


def loss_fn(
    model: nn.Module,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Next-token prediction loss."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    return f.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def run_variant(
    variant: Variant,
    base_config: ModelConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    train_config: TrainConfig,
) -> dict:
    """Train one variant and return results."""
    set_seed(SEED)

    # Override num_kv_heads per variant
    config = ModelConfig(
        embed_dim=base_config.embed_dim,
        num_heads=base_config.num_heads,
        num_kv_heads=variant.num_kv_heads,
        num_layers=base_config.num_layers,
        ffn_hidden_dim=base_config.ffn_hidden_dim,
        vocab_size=base_config.vocab_size,
        context_length=base_config.context_length,
        tie_weights=base_config.tie_weights,
        qk_norm=base_config.qk_norm,
        dropout=base_config.dropout,
    )

    model = build_model(variant, config)
    profile = profile_model(model)

    eval_bpbs: list[tuple[int, float]] = []

    def make_eval_fn(
        bpb_list: list[tuple[int, float]],
    ):  # noqa: ANN202
        """Create eval callback bound to bpb_list."""

        def eval_fn(m: nn.Module, step: int) -> dict[str, float]:
            m.eval()
            bpb = compute_bpb(m, val_data, bytes_per_token=1.0)
            bpb_list.append((step, bpb))
            return {'val_bpb': bpb}

        return eval_fn

    start = time.time()
    train_loop(
        model=model,
        train_data=train_data,
        loss_fn=loss_fn,
        config=train_config,
        batch_size=BATCH_SIZE,
        eval_fn=make_eval_fn(eval_bpbs),
        eval_interval=100,
    )
    elapsed = time.time() - start

    best_bpb = min(b for _, b in eval_bpbs) if eval_bpbs else float('inf')

    return {
        'name': variant.name,
        'attention': variant.attention,
        'use_rope': variant.use_rope,
        'learned_pos': variant.learned_pos,
        'num_kv_heads': variant.num_kv_heads or config.num_heads,
        'params': profile.total_params,
        'bpb': best_bpb,
        'train_time': elapsed,
    }


def main() -> None:
    """Run RoPE + GQA synergy analysis."""
    print('=' * 60)
    print('RoPE + GQA Synergy Analysis')
    print('=' * 60)

    # Load data
    print('\nLoading WikiText-2...')
    train_texts = download_wikitext2(split='train', max_items=MAX_ITEMS)
    val_texts = download_wikitext2(
        split='validation', max_items=MAX_ITEMS // 5
    )
    all_text = '\n'.join(train_texts + val_texts)
    tokenizer = CharTokenizer.from_text(all_text)
    print(f'Vocab size: {tokenizer.vocab_size}')

    base_config = ModelConfig(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=4,
        ffn_hidden_dim=128,
        vocab_size=tokenizer.vocab_size,
        context_length=128,
        tie_weights=True,
        qk_norm=True,
        dropout=0.0,
    )

    train_dataset = TextDataset(
        train_texts,
        tokenizer,
        context_length=base_config.context_length,
    )
    val_dataset = TextDataset(
        val_texts,
        tokenizer,
        context_length=base_config.context_length,
    )
    train_data = torch.stack(
        [
            train_dataset[i]
            for i in range(min(MAX_TRAIN_SEQS, len(train_dataset)))
        ]
    )
    val_data = torch.stack(
        [val_dataset[i] for i in range(min(MAX_VAL_SEQS, len(val_dataset)))]
    )
    print(f'Train: {len(train_data)}, Val: {len(val_data)}')

    train_config = TrainConfig(
        lr=1e-3,
        total_steps=TRAIN_STEPS,
        warmup_steps=50,
        weight_decay=0.1,
        max_grad_norm=1.0,
    )

    # --- Experiment 1: Position × Attention Matrix ---
    print('\n' + '=' * 60)
    print('Experiment 1: Position × Attention Matrix')
    print('=' * 60)

    matrix_results = []
    for v in MATRIX_VARIANTS:
        pos = (
            'rope' if v.use_rope else ('learned' if v.learned_pos else 'none')
        )
        print(f'\n  {v.name} (pos={pos}, attn={v.attention})')
        result = run_variant(
            v, base_config, train_data, val_data, train_config
        )
        matrix_results.append(result)
        print(
            f'    BPB={result["bpb"]:.3f} '
            f'params={result["params"]:,} '
            f'[{result["train_time"]:.0f}s]'
        )

    # --- Experiment 2: GQA Ratio Sweep ---
    print('\n' + '=' * 60)
    print('Experiment 2: GQA Ratio Sweep')
    print('=' * 60)

    ratio_results = []
    for v in RATIO_VARIANTS:
        pos = 'rope' if v.use_rope else 'learned'
        kv = v.num_kv_heads or base_config.num_heads
        print(f'\n  {v.name} (pos={pos}, kv_heads={kv})')
        result = run_variant(
            v, base_config, train_data, val_data, train_config
        )
        ratio_results.append(result)
        print(
            f'    BPB={result["bpb"]:.3f} '
            f'params={result["params"]:,} '
            f'[{result["train_time"]:.0f}s]'
        )

    # --- Print Results ---
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)

    # Matrix results as table
    print('\nExperiment 1: Position × Attention')
    print(f'{"Variant":<20} {"BPB":>8} {"Params":>10}')
    print('-' * 40)
    baseline_bpb = None
    for r in matrix_results:
        if r['name'] == 'learned+mha':
            baseline_bpb = r['bpb']
        delta = ''
        if baseline_bpb and r['bpb'] != baseline_bpb:
            pct = (r['bpb'] - baseline_bpb) / baseline_bpb
            delta = f' ({pct:+.1%})'
        print(f'{r["name"]:<20} {r["bpb"]:>8.3f} {r["params"]:>10,}{delta}')

    # Interaction effects
    print('\nInteraction Analysis:')
    bpb_map = {r['name']: r['bpb'] for r in matrix_results}
    if 'learned+mha' in bpb_map and 'rope+gqa' in bpb_map:
        base = bpb_map.get('learned+mha', 0)
        rope_only = bpb_map.get('rope+mha', 0)
        gqa_only = bpb_map.get('learned+gqa', 0)
        both = bpb_map.get('rope+gqa', 0)

        rope_effect = base - rope_only
        gqa_effect = base - gqa_only
        combined = base - both
        additive = rope_effect + gqa_effect
        synergy = combined - additive

        print(f'  RoPE alone:    {rope_effect:+.3f} BPB')
        print(f'  GQA alone:     {gqa_effect:+.3f} BPB')
        print(f'  Additive pred: {additive:+.3f} BPB')
        print(f'  RoPE+GQA:      {combined:+.3f} BPB')
        print(f'  Synergy:       {synergy:+.3f} BPB')
        if additive != 0:
            ratio = combined / additive
            print(f'  Synergy ratio: {ratio:.1f}x')

    # Ratio sweep results
    print('\nExperiment 2: GQA Ratio Sweep')
    print(f'{"Variant":<20} {"KV heads":>10} {"BPB":>8}')
    print('-' * 40)
    for r in ratio_results:
        print(f'{r["name"]:<20} {r["num_kv_heads"]:>10} {r["bpb"]:>8.3f}')

    # Save all results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        'seed': SEED,
        'train_steps': TRAIN_STEPS,
        'model_dim': base_config.embed_dim,
        'num_heads': base_config.num_heads,
        'num_layers': base_config.num_layers,
        'matrix_results': matrix_results,
        'ratio_results': ratio_results,
    }
    output_file = RESULTS_DIR / 'rope_gqa_synergy.json'
    output_file.write_text(json.dumps(output, indent=2))
    print(f'\nResults saved to {output_file}')


if __name__ == '__main__':
    main()
