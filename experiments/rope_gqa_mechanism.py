"""Mechanistic analysis of RoPE + GQA synergy.

Tests the "deposit head" hypothesis: RoPE concentrates positional
processing into ~1 head, freeing GQA's shared KV to focus on
content. Runs 3 analyses on all 4 variants after training:

  1. Head ablation: Zero out each head, measure BPB increase.
     If deposit pattern exists, one head should be critical.

  2. Attention entropy: Per-head entropy of attention distributions.
     Diverse entropy = better head specialization.

  3. Position bias: Correlation between attention weight and
     relative position distance. High correlation = position head.

Usage::

    python experiments/rope_gqa_mechanism.py
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
MAX_ITEMS = 5000
RESULTS_DIR = Path('experiments/results')

VARIANTS = [
    ('learned+mha', False, False),
    ('rope+mha', True, False),
    ('learned+gqa', False, True),
    ('rope+gqa', True, True),
]


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


def compute_attention_entropy(
    attn_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute entropy of attention distributions per head.

    Args:
        attn_weights: [batch, num_heads, seq_len, seq_len]

    Returns:
        [num_heads] mean entropy per head.
    """
    # Clamp to avoid log(0)
    attn = attn_weights.clamp(min=1e-10)
    # H = -sum(p * log(p))
    entropy = -(attn * attn.log()).sum(dim=-1)  # [b, heads, seq]
    # Average over batch and sequence positions
    return entropy.mean(dim=(0, 2))  # [heads]


def compute_position_bias(
    attn_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute correlation between attention and relative position.

    Measures how much each head attends based on distance vs content.
    High absolute correlation = position-sensitive head.

    Args:
        attn_weights: [batch, num_heads, seq_len, seq_len]

    Returns:
        [num_heads] correlation per head.
    """
    _, num_heads, seq_len, _ = attn_weights.shape

    # Build distance matrix: dist[i,j] = |i - j|
    pos = torch.arange(seq_len, device=attn_weights.device).float()
    dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
    # Invert: nearby = high, far = low (matching typical attention)
    proximity = 1.0 / (1.0 + dist)  # [seq, seq]

    correlations = []
    for h in range(num_heads):
        # Average attention pattern for this head
        avg_attn = attn_weights[:, h].mean(dim=0)  # [seq, seq]

        # Only look at causal part (lower triangle)
        mask = torch.tril(torch.ones_like(avg_attn)).bool()
        a = avg_attn[mask]
        p = proximity[mask]

        # Pearson correlation
        a_centered = a - a.mean()
        p_centered = p - p.mean()
        numer = (a_centered * p_centered).sum()
        denom = a_centered.pow(2).sum().sqrt() * p_centered.pow(2).sum().sqrt()
        corr = numer / denom.clamp(min=1e-10)
        correlations.append(corr.item())

    return torch.tensor(correlations)


def head_ablation_study(
    model: BaseModel,
    val_data: torch.Tensor,
) -> dict[str, Any]:
    """Zero out each head and measure BPB increase.

    For MHA: ablates each of num_heads heads per layer.
    For GQA: ablates each of num_heads query heads per layer
    (by zeroing the corresponding slice of out_proj).

    Returns:
        Dict with baseline BPB and per-head BPB increases.
    """
    model.eval()
    baseline_bpb = compute_bpb(model, val_data, bytes_per_token=1.0)

    results: dict[str, Any] = {
        'baseline_bpb': baseline_bpb,
        'layers': [],
    }

    for layer_idx, block in enumerate(model.blocks):
        attn = block.attn
        layer_results: list[dict[str, float]] = []

        # Get the output projection weight
        out_weight = attn.out_proj.weight.data
        head_dim = attn.head_dim
        num_heads = attn.num_heads

        for head_idx in range(num_heads):
            # Save original weights
            start = head_idx * head_dim
            end = start + head_dim
            original = out_weight[:, start:end].clone()

            # Zero out this head's contribution
            out_weight[:, start:end] = 0.0

            # Measure BPB
            ablated_bpb = compute_bpb(model, val_data, bytes_per_token=1.0)
            bpb_increase = ablated_bpb - baseline_bpb

            layer_results.append(
                {
                    'head': head_idx,
                    'ablated_bpb': ablated_bpb,
                    'bpb_increase': bpb_increase,
                }
            )

            # Restore
            out_weight[:, start:end] = original

        results['layers'].append(
            {
                'layer': layer_idx,
                'heads': layer_results,
            }
        )

    return results


def collect_attention_patterns(
    model: BaseModel,
    val_data: torch.Tensor,
    n_batches: int = 5,
) -> list[list[torch.Tensor]]:
    """Forward data and collect attention weights per layer.

    Returns:
        List of [layer][tensor of shape [batch, heads, seq, seq]].
    """
    model.eval()
    all_weights: list[list[torch.Tensor]] = [
        [] for _ in range(len(model.blocks))
    ]

    with torch.no_grad():
        for i in range(min(n_batches, len(val_data) // BATCH_SIZE)):
            batch = val_data[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, :-1]
            _ = model(batch)

            for layer_idx, block in enumerate(model.blocks):
                w = block.attn._attn_weights
                if w is not None:
                    all_weights[layer_idx].append(w.cpu())

    # Concatenate batches
    merged = []
    for layer_weights in all_weights:
        if layer_weights:
            merged.append(torch.cat(layer_weights, dim=0))
        else:
            merged.append(None)

    return merged


def main() -> None:
    """Run mechanistic analysis."""
    print('=' * 60)
    print('RoPE + GQA Mechanistic Analysis')
    print('=' * 60)

    # Load data
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

    all_results: dict[str, Any] = {}

    for name, use_rope, use_gqa in VARIANTS:
        print(f'\n{"=" * 60}')
        print(f'Variant: {name}')
        print('=' * 60)

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

        # Train
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
        print(
            f'  Training: BPB={best_bpb:.3f} '
            f'params={profile.total_params:,} [{elapsed:.0f}s]'
        )

        # --- Analysis 1: Head Ablation ---
        print('\n  Head Ablation Study:')
        ablation = head_ablation_study(model, val_data)
        print(f'    Baseline BPB: {ablation["baseline_bpb"]:.3f}')

        max_increase = 0.0
        max_head_info = ''
        for layer_info in ablation['layers']:
            layer_idx = layer_info['layer']
            for head_info in layer_info['heads']:
                inc = head_info['bpb_increase']
                marker = ' ***' if inc > 0.1 else ''
                print(
                    f'    L{layer_idx}H{head_info["head"]}: '
                    f'BPB +{inc:.3f}{marker}'
                )
                if inc > max_increase:
                    max_increase = inc
                    max_head_info = f'L{layer_idx}H{head_info["head"]}'

        deposit_ratio = max_increase / max(
            ablation['baseline_bpb'] * 0.01, 0.001
        )
        print(f'    Most critical: {max_head_info} (+{max_increase:.3f} BPB)')
        print(
            f'    Deposit concentration: {deposit_ratio:.1f}x vs 1% baseline'
        )

        # --- Analysis 2: Attention Entropy ---
        print('\n  Attention Entropy (per head, per layer):')
        attn_patterns = collect_attention_patterns(model, val_data)

        entropy_data: list[list[float]] = []
        for layer_idx, weights in enumerate(attn_patterns):
            if weights is None:
                continue
            entropies = compute_attention_entropy(weights)
            layer_entropies = entropies.tolist()
            entropy_data.append(layer_entropies)
            ent_str = ' '.join(f'{e:.2f}' for e in layer_entropies)
            print(f'    L{layer_idx}: [{ent_str}]')

        # Entropy diversity: std across heads (higher = more specialized)
        all_ent = [e for layer in entropy_data for e in layer]
        ent_std = (
            (
                sum((e - sum(all_ent) / len(all_ent)) ** 2 for e in all_ent)
                / len(all_ent)
            )
            ** 0.5
            if all_ent
            else 0.0
        )
        ent_range = max(all_ent) - min(all_ent) if all_ent else 0.0
        print(f'    Entropy diversity (std): {ent_std:.3f}')
        print(f'    Entropy range: {ent_range:.3f}')

        # --- Analysis 3: Position Bias ---
        print('\n  Position Bias (correlation with proximity):')
        pos_bias_data: list[list[float]] = []
        for layer_idx, weights in enumerate(attn_patterns):
            if weights is None:
                continue
            biases = compute_position_bias(weights)
            layer_biases = biases.tolist()
            pos_bias_data.append(layer_biases)
            bias_str = ' '.join(f'{b:.3f}' for b in layer_biases)
            print(f'    L{layer_idx}: [{bias_str}]')

        # Position specialization: max absolute bias across all heads
        all_bias = [abs(b) for layer in pos_bias_data for b in layer]
        max_pos_bias = max(all_bias) if all_bias else 0.0
        bias_range = max(all_bias) - min(all_bias) if all_bias else 0.0
        print(f'    Max position bias: {max_pos_bias:.3f}')
        print(f'    Bias range: {bias_range:.3f}')

        # Store results
        all_results[name] = {
            'bpb': best_bpb,
            'ablation': ablation,
            'entropy': entropy_data,
            'position_bias': pos_bias_data,
            'entropy_diversity': ent_std,
            'entropy_range': ent_range,
            'max_position_bias': max_pos_bias,
            'position_bias_range': bias_range,
            'deposit_concentration': deposit_ratio,
            'most_critical_head': max_head_info,
            'max_ablation_increase': max_increase,
        }

    # --- Cross-variant comparison ---
    print('\n' + '=' * 60)
    print('CROSS-VARIANT COMPARISON')
    print('=' * 60)

    print(
        f'\n{"Variant":<15} {"BPB":>6} {"Deposit":>10} '
        f'{"Ent Div":>8} {"Pos Bias":>10} {"Critical":>10}'
    )
    print('-' * 65)
    for name, data in all_results.items():
        print(
            f'{name:<15} {data["bpb"]:>6.3f} '
            f'{data["deposit_concentration"]:>10.1f}x '
            f'{data["entropy_diversity"]:>8.3f} '
            f'{data["max_position_bias"]:>10.3f} '
            f'{data["most_critical_head"]:>10}'
        )

    # Test deposit pattern prediction
    print('\nDeposit Pattern Test:')
    print('  Prediction: RoPE+GQA should have the sharpest deposit')
    print('  (one head much more critical than others)')
    deposits = {
        name: data['deposit_concentration']
        for name, data in all_results.items()
    }
    best_deposit = max(deposits, key=deposits.get)
    dep = deposits[best_deposit]
    print(f'  Sharpest deposit: {best_deposit} ({dep:.1f}x)')
    if best_deposit == 'rope+gqa':
        print('  ✅ Prediction CONFIRMED')
    else:
        print('  ❌ Prediction FAILED (expected rope+gqa)')

    print('\nEntropy Diversity Test:')
    print('  Prediction: RoPE+GQA should have widest entropy spread')
    ent_divs = {
        name: data['entropy_diversity'] for name, data in all_results.items()
    }
    best_ent = max(ent_divs, key=ent_divs.get)
    print(f'  Most diverse: {best_ent} ({ent_divs[best_ent]:.3f})')
    if best_ent == 'rope+gqa':
        print('  ✅ Prediction CONFIRMED')
    else:
        print('  ❌ Prediction FAILED (expected rope+gqa)')

    print('\nPosition Specialization Test:')
    print('  Prediction: RoPE variants should have higher position bias')
    for name, data in all_results.items():
        print(f'  {name:<15} max_pos_bias={data["max_position_bias"]:.3f}')

    # Save
    # Convert non-serializable items for JSON
    serializable = {}
    for name, data in all_results.items():
        s = dict(data)
        # ablation already serializable
        serializable[name] = s

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / 'rope_gqa_mechanism.json'
    output_file.write_text(json.dumps(serializable, indent=2))
    print(f'\nResults saved to {output_file}')


if __name__ == '__main__':
    main()
