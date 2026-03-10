"""Multi-seed validation of RoPE + GQA synergy.

Validates the 13.6x super-additive synergy found in the single-seed
factorial experiment. Tests only the 4 critical variants across 3 seeds
to check whether the effect is robust or a fluke.

Variants: learned+mha (baseline), rope+mha, learned+gqa, rope+gqa
Seeds: 42, 123, 456

Usage::

    python experiments/rope_gqa_multiseed.py
"""

import json
import time
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

SEEDS = [42, 123, 456]
BATCH_SIZE = 32
TRAIN_STEPS = 500
MAX_TRAIN_SEQS = 2000
MAX_VAL_SEQS = 200
MAX_ITEMS = 5000
RESULTS_DIR = Path('experiments/results')


def build_variant(
    name: str,
    config: ModelConfig,
    use_rope: bool,
    use_gqa: bool,
) -> BaseModel:
    """Build model for one of the 4 critical variants."""
    rope = None
    if use_rope:
        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(
            d_model=head_dim,
            max_seq_len=config.context_length,
        )

    block_config = BlockConfig(
        attention='gqa' if use_gqa else 'mha',
        ffn='default',
        norm='layernorm',
        rope=rope,
    )

    return BaseModel(
        config,
        block_config=block_config,
        learned_pos_embed=not use_rope,
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


VARIANTS = [
    ('learned+mha', False, False),
    ('rope+mha', True, False),
    ('learned+gqa', False, True),
    ('rope+gqa', True, True),
]


def main() -> None:
    """Run multi-seed validation."""
    print('=' * 60)
    print('RoPE + GQA Multi-Seed Validation')
    print(f'Seeds: {SEEDS}')
    print(f'Variants: {[v[0] for v in VARIANTS]}')
    print('=' * 60)

    # Load data once
    print('\nLoading WikiText-2...')
    train_texts = download_wikitext2(split='train', max_items=MAX_ITEMS)
    val_texts = download_wikitext2(
        split='validation', max_items=MAX_ITEMS // 5
    )
    all_text = '\n'.join(train_texts + val_texts)
    tokenizer = CharTokenizer.from_text(all_text)

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

    train_config = TrainConfig(
        lr=1e-3,
        total_steps=TRAIN_STEPS,
        warmup_steps=50,
        weight_decay=0.1,
        max_grad_norm=1.0,
    )

    # Run all combinations
    all_results: dict[str, list[float]] = {v[0]: [] for v in VARIANTS}

    for seed in SEEDS:
        print(f'\n--- Seed {seed} ---')
        for name, use_rope, use_gqa in VARIANTS:
            set_seed(seed)

            config = ModelConfig(
                embed_dim=base_config.embed_dim,
                num_heads=base_config.num_heads,
                num_kv_heads=(base_config.num_kv_heads if use_gqa else None),
                num_layers=base_config.num_layers,
                ffn_hidden_dim=base_config.ffn_hidden_dim,
                vocab_size=base_config.vocab_size,
                context_length=base_config.context_length,
                tie_weights=base_config.tie_weights,
                qk_norm=base_config.qk_norm,
                dropout=base_config.dropout,
            )

            model = build_variant(name, config, use_rope, use_gqa)
            profile = profile_model(model)

            eval_bpbs: list[tuple[int, float]] = []

            def make_eval(
                bpb_list: list[tuple[int, float]],
            ):  # noqa: ANN202
                """Create eval callback."""

                def fn(m: nn.Module, step: int) -> dict[str, float]:
                    m.eval()
                    bpb = compute_bpb(m, val_data, bytes_per_token=1.0)
                    bpb_list.append((step, bpb))
                    return {'val_bpb': bpb}

                return fn

            start = time.time()
            train_loop(
                model=model,
                train_data=train_data,
                loss_fn=loss_fn,
                config=train_config,
                batch_size=BATCH_SIZE,
                eval_fn=make_eval(eval_bpbs),
                eval_interval=100,
            )
            elapsed = time.time() - start

            best_bpb = min(b for _, b in eval_bpbs)
            all_results[name].append(best_bpb)
            print(
                f'  {name:<15} BPB={best_bpb:.3f} '
                f'params={profile.total_params:,} '
                f'[{elapsed:.0f}s]'
            )

    # Summary
    print('\n' + '=' * 60)
    print('MULTI-SEED RESULTS')
    print('=' * 60)
    print(
        f'\n{"Variant":<15} '
        + ' '.join(f'{"s" + str(s):>8}' for s in SEEDS)
        + f' {"Mean":>8} {"Std":>8}'
    )
    print('-' * 60)

    means = {}
    for name, bpbs in all_results.items():
        mean = sum(bpbs) / len(bpbs)
        std = (sum((b - mean) ** 2 for b in bpbs) / len(bpbs)) ** 0.5
        means[name] = mean
        vals = ' '.join(f'{b:>8.3f}' for b in bpbs)
        print(f'{name:<15} {vals} {mean:>8.3f} {std:>8.4f}')

    # Interaction analysis
    if all(k in means for k in all_results):
        base = means['learned+mha']
        rope_eff = base - means['rope+mha']
        gqa_eff = base - means['learned+gqa']
        combined = base - means['rope+gqa']
        additive = rope_eff + gqa_eff
        synergy = combined - additive

        print('\nInteraction (on means):')
        print(f'  RoPE effect:   {rope_eff:+.3f}')
        print(f'  GQA effect:    {gqa_eff:+.3f}')
        print(f'  Additive pred: {additive:+.3f}')
        print(f'  RoPE+GQA:      {combined:+.3f}')
        print(f'  Synergy:       {synergy:+.3f}')
        if additive != 0:
            print(f'  Ratio:         {combined / additive:.1f}x')

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / 'rope_gqa_multiseed.json'
    output_file.write_text(
        json.dumps(
            {
                'seeds': SEEDS,
                'results': all_results,
                'means': means,
            },
            indent=2,
        )
    )
    print(f'\nResults saved to {output_file}')


if __name__ == '__main__':
    main()
