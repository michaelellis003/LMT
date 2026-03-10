"""Position encoding × attention structure factorial ablation.

Tests whether the RoPE+GQA synergy is RoPE-specific or if ANY
position encoding synergizes with GQA. This is the critical
experiment to distinguish between:

  H1: RoPE is special (only RoPE + GQA synergizes)
  H2: Any PE + GQA synergizes (GQA needs position info, any source)
  H3: Only rotation-based PEs synergize (RoPE/ALiBi but not learned)

Design: 4 PE types × 2 attention types = 8 variants
  PE: {none, learned, ALiBi, RoPE}
  Attention: {MHA, GQA}

Single seed per variant on WikiText-2. If results are clear, can
add multi-seed follow-up.

NOTE: Previous experiments had a confound — MHA ignored RoPE in
ConfigurableBlock, so 'rope+mha' was actually 'no_pe+mha'. This
experiment fixes that by passing rope/alibi directly to MHA.

Usage::

    python experiments/rope_gqa_pe_ablation.py
"""

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.data.hf_datasets import download_wikitext2
from lmt.data.text_dataset import TextDataset
from lmt.eval.bpb import compute_bpb
from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.layers.positional import ALiBi, RoPE
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

# 4 PE types × 2 attention types = 8 variants
# (name, pe_type, use_gqa)
# pe_type: 'none', 'learned', 'alibi', 'rope'
VARIANTS = [
    ('none+mha', 'none', False),
    ('none+gqa', 'none', True),
    ('learned+mha', 'learned', False),
    ('learned+gqa', 'learned', True),
    ('alibi+mha', 'alibi', False),
    ('alibi+gqa', 'alibi', True),
    ('rope+mha', 'rope', False),
    ('rope+gqa', 'rope', True),
]


def build_variant(
    config: ModelConfig,
    pe_type: str,
    use_gqa: bool,
) -> BaseModel:
    """Build model for one variant.

    Args:
        config: Model configuration.
        pe_type: Position encoding type ('none', 'learned', 'alibi',
            'rope').
        use_gqa: Whether to use GQA (True) or MHA (False).

    Returns:
        Initialized model.
    """
    rope = None
    alibi = None
    learned_pos = False

    if pe_type == 'rope':
        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(
            d_model=head_dim,
            max_seq_len=config.context_length,
        )
    elif pe_type == 'alibi':
        alibi = ALiBi(
            num_heads=config.num_heads,
            max_seq_len=config.context_length,
        )
    elif pe_type == 'learned':
        learned_pos = True
    # pe_type == 'none': no position encoding at all

    block_config = BlockConfig(
        attention='gqa' if use_gqa else 'mha',
        ffn='default',
        norm='layernorm',
        rope=rope,
        alibi=alibi,
    )

    return BaseModel(
        config,
        block_config=block_config,
        learned_pos_embed=learned_pos,
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
    """Run position encoding ablation."""
    print('=' * 60)
    print('Position Encoding × Attention Structure Ablation')
    print(f'Variants: {len(VARIANTS)} (4 PE × 2 attn)')
    print(f'Seed: {SEED}, Steps: {TRAIN_STEPS}')
    print('=' * 60)

    # Load data
    train_texts = download_wikitext2(split='train', max_items=5000)
    val_texts = download_wikitext2(split='validation', max_items=1000)
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

    results: dict[str, dict[str, Any]] = {}

    for name, pe_type, use_gqa in VARIANTS:
        set_seed(SEED)

        # Build config — only set num_kv_heads for GQA
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

        model = build_variant(config, pe_type, use_gqa)
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
        results[name] = {
            'bpb': best_bpb,
            'params': profile.total_params,
            'time': elapsed,
            'pe_type': pe_type,
            'attention': 'gqa' if use_gqa else 'mha',
        }
        print(
            f'  {name:<15} BPB={best_bpb:.3f} '
            f'params={profile.total_params:,} [{elapsed:.0f}s]'
        )

    # Analysis
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)

    # Table
    header = f'{"Variant":<15} {"BPB":>8} {"Params":>10}'
    print(f'\n{header}')
    print('-' * 35)
    for name, info in results.items():
        print(f'{name:<15} {info["bpb"]:>8.3f} {info["params"]:>10,}')

    # Synergy analysis for each PE type
    print('\n' + '=' * 60)
    print('SYNERGY ANALYSIS (per PE type)')
    print('=' * 60)
    print(
        f'{"PE Type":<10} {"MHA BPB":>10} {"GQA BPB":>10} '
        f'{"Delta":>10} {"% Impr":>8}'
    )
    print('-' * 52)

    pe_types = ['none', 'learned', 'alibi', 'rope']
    for pe in pe_types:
        mha_key = f'{pe}+mha'
        gqa_key = f'{pe}+gqa'
        mha_bpb = results[mha_key]['bpb']
        gqa_bpb = results[gqa_key]['bpb']
        delta = mha_bpb - gqa_bpb
        pct = 100.0 * delta / mha_bpb if mha_bpb > 0 else 0
        print(
            f'{pe:<10} {mha_bpb:>10.3f} {gqa_bpb:>10.3f} '
            f'{delta:>+10.3f} {pct:>+7.1f}%'
        )

    # Cross-PE comparison: what's the effect of each PE on MHA/GQA?
    print('\n' + '=' * 60)
    print('PE EFFECT (vs no-PE baseline)')
    print('=' * 60)
    none_mha = results['none+mha']['bpb']
    none_gqa = results['none+gqa']['bpb']

    print(f'\nMHA baseline (no PE): {none_mha:.3f}')
    print(f'GQA baseline (no PE): {none_gqa:.3f}')

    print(
        f'\n{"PE":<10} {"MHA delta":>10} {"GQA delta":>10} '
        f'{"MHA %":>8} {"GQA %":>8}'
    )
    print('-' * 50)
    for pe in ['learned', 'alibi', 'rope']:
        mha_bpb = results[f'{pe}+mha']['bpb']
        gqa_bpb = results[f'{pe}+gqa']['bpb']
        mha_delta = none_mha - mha_bpb
        gqa_delta = none_gqa - gqa_bpb
        mha_pct = 100.0 * mha_delta / none_mha
        gqa_pct = 100.0 * gqa_delta / none_gqa
        print(
            f'{pe:<10} {mha_delta:>+10.3f} {gqa_delta:>+10.3f} '
            f'{mha_pct:>+7.1f}% {gqa_pct:>+7.1f}%'
        )

    # Interaction: does PE × GQA interact?
    # For each PE type, compute:
    #   PE_effect = none_attn - pe_attn (same attn type)
    #   GQA_effect = mha_pe - gqa_pe (same PE)
    #   Combined = none_mha - pe_gqa
    #   Additive = PE_effect_mha + GQA_effect_none
    #   Synergy = Combined - Additive
    print('\n' + '=' * 60)
    print('INTERACTION ANALYSIS (2×2 factorial per PE)')
    print('=' * 60)

    gqa_effect_none = none_mha - none_gqa  # GQA effect with no PE

    print(
        f'\n{"PE":<10} {"PE_eff":>8} {"GQA_eff":>8} '
        f'{"Additive":>10} {"Combined":>10} '
        f'{"Synergy":>10} {"Ratio":>8}'
    )
    print('-' * 70)
    for pe in ['learned', 'alibi', 'rope']:
        pe_eff_mha = none_mha - results[f'{pe}+mha']['bpb']
        combined = none_mha - results[f'{pe}+gqa']['bpb']
        additive = pe_eff_mha + gqa_effect_none
        synergy = combined - additive
        ratio = combined / additive if additive != 0 else float('inf')
        print(
            f'{pe:<10} {pe_eff_mha:>+8.3f} '
            f'{gqa_effect_none:>+8.3f} '
            f'{additive:>+10.3f} {combined:>+10.3f} '
            f'{synergy:>+10.3f} {ratio:>7.1f}x'
        )

    # Determine which hypothesis is supported
    print('\n' + '=' * 60)
    print('HYPOTHESIS TEST')
    print('=' * 60)

    rope_syn = (
        none_mha
        - results['rope+gqa']['bpb']
        - (none_mha - results['rope+mha']['bpb'])
        - gqa_effect_none
    )
    alibi_syn = (
        none_mha
        - results['alibi+gqa']['bpb']
        - (none_mha - results['alibi+mha']['bpb'])
        - gqa_effect_none
    )
    learned_syn = (
        none_mha
        - results['learned+gqa']['bpb']
        - (none_mha - results['learned+mha']['bpb'])
        - gqa_effect_none
    )

    print(f'RoPE synergy:    {rope_syn:+.3f}')
    print(f'ALiBi synergy:   {alibi_syn:+.3f}')
    print(f'Learned synergy: {learned_syn:+.3f}')
    print()

    threshold = 0.05  # 50 millibits
    rope_sig = abs(rope_syn) > threshold
    alibi_sig = abs(alibi_syn) > threshold
    learned_sig = abs(learned_syn) > threshold

    if rope_sig and not alibi_sig and not learned_sig:
        verdict = 'H1: RoPE is special (only RoPE synergizes)'
    elif rope_sig and alibi_sig and not learned_sig:
        verdict = 'H3: Rotation/bias PEs synergize (RoPE+ALiBi, not learned)'
    elif rope_sig and alibi_sig and learned_sig:
        verdict = 'H2: Any PE synergizes with GQA'
    else:
        verdict = 'Inconclusive (need multi-seed validation)'
    print(f'Verdict: {verdict}')

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        'results': results,
        'gqa_effect_none': gqa_effect_none,
        'synergies': {
            'rope': rope_syn,
            'alibi': alibi_syn,
            'learned': learned_syn,
        },
        'verdict': verdict,
    }
    output_file = RESULTS_DIR / 'rope_gqa_pe_ablation.json'
    output_file.write_text(json.dumps(output, indent=2))
    print(f'\nResults saved to {output_file}')


if __name__ == '__main__':
    main()
