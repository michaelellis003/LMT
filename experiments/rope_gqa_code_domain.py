"""RoPE + GQA synergy on Python code domain.

Tests whether the super-additive synergy between RoPE and GQA
(validated on 3 natural language datasets) also holds on Python
source code — a structurally different domain with:
  - Rigid syntax (indentation, brackets, colons)
  - Long-range dependencies (function definitions, returns)
  - Repetitive patterns (import blocks, docstrings)

Uses CodeSearchNet Python (real GitHub functions) as data source.
Same 4 critical variants, single seed.

Usage::

    python experiments/rope_gqa_code_domain.py
"""

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as f

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

SEED = 42
BATCH_SIZE = 32
TRAIN_STEPS = 500
MAX_TRAIN_SEQS = 2000
MAX_VAL_SEQS = 200
RESULTS_DIR = Path('experiments/results')

VARIANTS = [
    ('learned+mha', False, False),
    ('rope+mha', True, False),
    ('learned+gqa', False, True),
    ('rope+gqa', True, True),
]


def download_codesearchnet_python(
    max_items: int = 5000,
    val_fraction: float = 0.2,
) -> tuple[list[str], list[str]]:
    """Download Python functions from CodeSearchNet.

    Args:
        max_items: Total items to download.
        val_fraction: Fraction to use for validation.

    Returns:
        (train_texts, val_texts) tuple.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            'datasets package required: uv pip install datasets'
        ) from e

    ds = load_dataset(
        'Nan-Do/code-search-net-python',
        split='train',
        streaming=True,
    )

    texts: list[str] = []
    for row in ds:
        code = row.get('code', '')
        if not code.strip():
            continue
        # Filter very short or very long functions
        if len(code) < 50 or len(code) > 5000:
            continue
        texts.append(code)
        if len(texts) >= max_items:
            break

    # Split into train/val
    n_val = max(1, int(len(texts) * val_fraction))
    val_texts = texts[:n_val]
    train_texts = texts[n_val:]

    return train_texts, val_texts


def build_variant(
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


def main() -> None:
    """Run code-domain RoPE+GQA validation."""
    print('=' * 60)
    print('RoPE + GQA Code Domain Validation')
    print(f'Seed: {SEED}')
    print('=' * 60)

    # Load Python code
    print('\nDownloading CodeSearchNet Python...')
    train_texts, val_texts = download_codesearchnet_python(
        max_items=5000,
    )
    print(f'Loaded {len(train_texts)} train, {len(val_texts)} val')

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
    print(f'Train seqs: {len(train_data)}, Val seqs: {len(val_data)}')

    train_config = TrainConfig(
        lr=1e-3,
        total_steps=TRAIN_STEPS,
        warmup_steps=50,
        weight_decay=0.1,
        max_grad_norm=1.0,
    )

    # Run all 4 variants
    results: dict[str, Any] = {}
    for name, use_rope, use_gqa in VARIANTS:
        set_seed(SEED)

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

        model = build_variant(config, use_rope, use_gqa)
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
        results[name] = best_bpb
        print(
            f'  {name:<15} BPB={best_bpb:.3f} '
            f'params={profile.total_params:,} '
            f'[{elapsed:.0f}s]'
        )

    # Interaction analysis
    base = results['learned+mha']
    rope_eff = base - results['rope+mha']
    gqa_eff = base - results['learned+gqa']
    combined = base - results['rope+gqa']
    additive = rope_eff + gqa_eff
    synergy = combined - additive

    print('\n' + '=' * 60)
    print('CODE DOMAIN RESULTS')
    print('=' * 60)
    print(f'\n{"Variant":<15} {"BPB":>8} {"vs baseline":>12}')
    print('-' * 40)
    for name, bpb in results.items():
        delta = (bpb - base) / base
        print(f'{name:<15} {bpb:>8.3f} {delta:>+11.1%}')

    print('\nInteraction analysis:')
    print(f'  RoPE effect:   {rope_eff:+.3f}')
    print(f'  GQA effect:    {gqa_eff:+.3f}')
    print(f'  Additive pred: {additive:+.3f}')
    print(f'  RoPE+GQA:      {combined:+.3f}')
    print(f'  Synergy:       {synergy:+.3f}')
    if additive != 0:
        print(f'  Ratio:         {combined / additive:.1f}x')

    # Compare with natural language results
    print('\nComparison with natural language:')
    print(f'  {"Domain":<15} {"Synergy":>10} {"Combined":>10}')
    print(f'  {"Code (Python)":<15} {synergy:>+10.3f} {combined:>+10.3f}')
    print(f'  {"WikiText-2":<15} {"+0.428":>10} {"+0.462":>10}')
    print(f'  {"BabyLM":<15} {"+0.582":>10} {"+0.646":>10}')
    print(f'  {"Shakespeare":<15} {"+0.414":>10} {"+0.398":>10}')

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        'dataset': 'codesearchnet_python',
        'seed': SEED,
        'results': results,
        'rope_effect': rope_eff,
        'gqa_effect': gqa_eff,
        'combined': combined,
        'additive_pred': additive,
        'synergy': synergy,
    }
    if additive != 0:
        output['ratio'] = combined / additive

    output_file = RESULTS_DIR / 'rope_gqa_code_domain.json'
    output_file.write_text(json.dumps(output, indent=2))
    print(f'\nResults saved to {output_file}')


if __name__ == '__main__':
    main()
