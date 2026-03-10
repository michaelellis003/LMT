"""Cross-dataset validation of RoPE + GQA synergy.

Tests whether the 10.9x super-additive synergy between RoPE and GQA
(validated across 3 seeds on WikiText-2) generalizes to different
text domains:

  1. WikiText-2 (Wikipedia articles) -- already validated
  2. BabyLM (child-directed speech, subtitles, children's books)
  3. TinyShakespeare (classic literature, Early Modern English)

Same 4 critical variants, single seed (42) per dataset.
If the synergy holds across these very different distributions,
it's likely an architectural property, not a dataset artifact.

Usage::

    python experiments/rope_gqa_cross_dataset.py
"""

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.data.hf_datasets import download_babylm, download_wikitext2
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


def download_tiny_shakespeare(
    split: str = 'train',
    max_items: int | None = None,
) -> list[str]:
    """Download TinyShakespeare from HuggingFace.

    Args:
        split: 'train', 'test', or 'validation'.
        max_items: Maximum lines to return.

    Returns:
        List of non-empty text strings.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            'datasets package required: uv pip install datasets'
        ) from e

    ds = load_dataset('Trelis/tiny-shakespeare', split=split)

    texts: list[str] = []
    for row in ds:
        text = row.get('Text', row.get('text', ''))
        if not text.strip():
            continue
        texts.append(text)
        if max_items is not None and len(texts) >= max_items:
            break

    return texts


def load_dataset_texts(
    name: str,
    max_train: int = 5000,
    max_val: int = 1000,
) -> tuple[list[str], list[str]]:
    """Load train/val text for a named dataset.

    Args:
        name: One of 'wikitext2', 'babylm', 'shakespeare'.
        max_train: Max training text items.
        max_val: Max validation text items.

    Returns:
        (train_texts, val_texts) tuple.
    """
    if name == 'wikitext2':
        train = download_wikitext2(split='train', max_items=max_train)
        val = download_wikitext2(split='validation', max_items=max_val)
    elif name == 'babylm':
        train = download_babylm(
            track='strict-small',
            split='train',
            max_items=max_train,
        )
        val = download_babylm(
            track='strict-small',
            split='validation',
            max_items=max_val,
        )
    elif name == 'shakespeare':
        # TinyShakespeare has train/test only (no validation split)
        train = download_tiny_shakespeare(split='train', max_items=max_train)
        val = download_tiny_shakespeare(split='test', max_items=max_val)
    else:
        raise ValueError(f'Unknown dataset: {name}')

    return train, val


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


def run_dataset(
    dataset_name: str,
    train_texts: list[str],
    val_texts: list[str],
) -> dict[str, Any]:
    """Run all 4 variants on one dataset.

    Returns:
        Dict with variant BPBs and interaction analysis.
    """
    all_text = '\n'.join(train_texts + val_texts)
    tokenizer = CharTokenizer.from_text(all_text)
    print(f'  Vocab size: {tokenizer.vocab_size}')

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
    print(f'  Train seqs: {len(train_data)}, Val seqs: {len(val_data)}')

    train_config = TrainConfig(
        lr=1e-3,
        total_steps=TRAIN_STEPS,
        warmup_steps=50,
        weight_decay=0.1,
        max_grad_norm=1.0,
    )

    results: dict[str, float] = {}
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
            f'    {name:<15} BPB={best_bpb:.3f} '
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
    ratio = combined / additive if additive != 0 else float('inf')

    return {
        'dataset': dataset_name,
        'results': results,
        'rope_effect': rope_eff,
        'gqa_effect': gqa_eff,
        'combined': combined,
        'additive_pred': additive,
        'synergy': synergy,
        'ratio': ratio,
    }


DATASETS = ['wikitext2', 'babylm', 'shakespeare']


def main() -> None:
    """Run cross-dataset validation."""
    print('=' * 60)
    print('RoPE + GQA Cross-Dataset Validation')
    print(f'Datasets: {DATASETS}')
    print(f'Seed: {SEED}')
    print('=' * 60)

    all_dataset_results: list[dict[str, Any]] = []

    for ds_name in DATASETS:
        print(f'\n{"=" * 60}')
        print(f'Dataset: {ds_name}')
        print('=' * 60)

        train_texts, val_texts = load_dataset_texts(ds_name)
        print(f'  Loaded {len(train_texts)} train, {len(val_texts)} val texts')

        ds_result = run_dataset(ds_name, train_texts, val_texts)
        all_dataset_results.append(ds_result)

    # Cross-dataset summary
    print('\n' + '=' * 60)
    print('CROSS-DATASET RESULTS')
    print('=' * 60)

    header = f'{"Dataset":<15}'
    for name, _, _ in VARIANTS:
        header += f' {name:>12}'
    header += f' {"Synergy":>10} {"Ratio":>8}'
    print(f'\n{header}')
    print('-' * 80)

    for ds_r in all_dataset_results:
        row = f'{ds_r["dataset"]:<15}'
        for name, _, _ in VARIANTS:
            row += f' {ds_r["results"][name]:>12.3f}'
        row += f' {ds_r["synergy"]:>+10.3f}'
        row += f' {ds_r["ratio"]:>8.1f}x'
        print(row)

    print('\nInteraction breakdown:')
    print(
        f'{"Dataset":<15} {"RoPE":>8} {"GQA":>8} '
        f'{"Additive":>10} {"Combined":>10} {"Synergy":>10}'
    )
    print('-' * 65)
    for ds_r in all_dataset_results:
        print(
            f'{ds_r["dataset"]:<15} '
            f'{ds_r["rope_effect"]:>+8.3f} '
            f'{ds_r["gqa_effect"]:>+8.3f} '
            f'{ds_r["additive_pred"]:>+10.3f} '
            f'{ds_r["combined"]:>+10.3f} '
            f'{ds_r["synergy"]:>+10.3f}'
        )

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / 'rope_gqa_cross_dataset.json'
    output_file.write_text(json.dumps(all_dataset_results, indent=2))
    print(f'\nResults saved to {output_file}')


if __name__ == '__main__':
    main()
