"""Pairwise architecture ablation: which feature pairs have strongest synergy?

Follow-up to the single-feature ablation which found 26x synergy between
LLaMA's four innovations. This experiment tests all 6 pairwise combinations
to identify where the synergy is concentrated.

Variants tested (on top of GPT baseline):
  1. GPT baseline          (reference)
  2. RoPE + SwiGLU
  3. RoPE + GQA
  4. RoPE + RMSNorm
  5. SwiGLU + GQA
  6. SwiGLU + RMSNorm
  7. GQA + RMSNorm
  8. LLaMA (all four)      (reference)

Usage::

    .venv/bin/python experiments/pairwise_ablation.py

Results are printed as a summary table at the end.
"""

import time
from dataclasses import dataclass

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
EVAL_INTERVAL = 100
TRAIN_STEPS = 500
MAX_TRAIN_SEQS = 2000
MAX_VAL_SEQS = 200
MAX_ITEMS = 5000  # Limit HF download for memory safety


@dataclass
class PairVariant:
    """Defines one pairwise ablation variant."""

    name: str
    attention: str
    ffn: str
    norm: str
    use_rope: bool
    learned_pos: bool


# All 8 variants: baseline + 6 pairs + full LLaMA
VARIANTS = [
    PairVariant(
        name='gpt_baseline',
        attention='mha',
        ffn='default',
        norm='layernorm',
        use_rope=False,
        learned_pos=True,
    ),
    PairVariant(
        name='rope+swiglu',
        attention='mha',
        ffn='swiglu',
        norm='layernorm',
        use_rope=True,
        learned_pos=False,
    ),
    PairVariant(
        name='rope+gqa',
        attention='gqa',
        ffn='default',
        norm='layernorm',
        use_rope=True,
        learned_pos=False,
    ),
    PairVariant(
        name='rope+rmsnorm',
        attention='mha',
        ffn='default',
        norm='rmsnorm',
        use_rope=True,
        learned_pos=False,
    ),
    PairVariant(
        name='swiglu+gqa',
        attention='gqa',
        ffn='swiglu',
        norm='layernorm',
        use_rope=False,
        learned_pos=True,
    ),
    PairVariant(
        name='swiglu+rmsnorm',
        attention='mha',
        ffn='swiglu',
        norm='rmsnorm',
        use_rope=False,
        learned_pos=True,
    ),
    PairVariant(
        name='gqa+rmsnorm',
        attention='gqa',
        ffn='default',
        norm='rmsnorm',
        use_rope=False,
        learned_pos=True,
    ),
    PairVariant(
        name='llama_full',
        attention='gqa',
        ffn='swiglu',
        norm='rmsnorm',
        use_rope=True,
        learned_pos=False,
    ),
]


def build_model(variant: PairVariant, config: ModelConfig) -> BaseModel:
    """Build a model for the given pairwise variant."""
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


def main() -> None:
    """Run pairwise architecture ablation study."""
    print('=' * 60)
    print('Pairwise Architecture Ablation Study')
    print('=' * 60)
    print(f'Variants: {len(VARIANTS)}')

    # Load data
    print('\n  Loading WikiText-2...')
    train_texts = download_wikitext2(split='train', max_items=MAX_ITEMS)
    val_texts = download_wikitext2(
        split='validation', max_items=MAX_ITEMS // 5
    )
    all_text = '\n'.join(train_texts + val_texts)
    tokenizer = CharTokenizer.from_text(all_text)
    print(f'  Vocab size: {tokenizer.vocab_size}')

    config = ModelConfig(
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
        train_texts, tokenizer, context_length=config.context_length
    )
    val_dataset = TextDataset(
        val_texts, tokenizer, context_length=config.context_length
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
    print(f'  Train: {len(train_data)}, Val: {len(val_data)}')

    train_config = TrainConfig(
        lr=1e-3,
        total_steps=TRAIN_STEPS,
        warmup_steps=50,
        weight_decay=0.1,
        max_grad_norm=1.0,
    )

    # Run all variants
    results = []
    for variant in VARIANTS:
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

        eval_bpbs: list[tuple[int, float]] = []

        def make_eval_fn(
            bpb_list: list[tuple[int, float]],
        ):  # noqa: ANN202
            """Create eval callback bound to bpb_list."""

            def eval_fn(m: nn.Module, step: int) -> dict[str, float]:
                m.eval()
                bpb = compute_bpb(m, val_data, bytes_per_token=1.0)
                bpb_list.append((step, bpb))
                print(f'    Step {step:4d} | bpb={bpb:.4f}')
                return {'val_bpb': bpb}

            return eval_fn

        start = time.time()
        state = train_loop(
            model=model,
            train_data=train_data,
            loss_fn=loss_fn,
            config=train_config,
            batch_size=BATCH_SIZE,
            eval_fn=make_eval_fn(eval_bpbs),
            eval_interval=EVAL_INTERVAL,
        )
        elapsed = time.time() - start

        best_bpb = min(b for _, b in eval_bpbs) if eval_bpbs else None
        print(f'    Best BPB: {best_bpb:.4f}, Time: {elapsed:.1f}s')

        results.append(
            {
                'name': variant.name,
                'best_bpb': best_bpb,
                'final_loss': state.final_loss,
                'elapsed': elapsed,
            }
        )

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print('\n' + '=' * 60)
    print('PAIRWISE ABLATION RESULTS')
    print('=' * 60)

    gpt_bpb = results[0]['best_bpb']
    llama_bpb = results[-1]['best_bpb']

    header = f'{"Variant":<18} {"BPB":>8} {"vs GPT":>8} {"vs LLaMA":>10}'
    print(header)
    print('-' * 50)

    for r in results:
        bpb = r['best_bpb']
        vs_gpt = (bpb - gpt_bpb) / gpt_bpb * 100 if bpb and gpt_bpb else 0
        vs_llama = (
            (bpb - llama_bpb) / llama_bpb * 100 if bpb and llama_bpb else 0
        )
        print(
            f'{r["name"]:<18} {bpb:>8.4f} {vs_gpt:>+7.1f}% {vs_llama:>+9.1f}%'
        )

    # Rank pairs by improvement
    print('\n' + '-' * 50)
    print('PAIR RANKING (best to worst)')
    print('-' * 50)

    pairs = [r for r in results[1:-1]]  # Exclude baseline and full
    pairs.sort(key=lambda r: r['best_bpb'] or float('inf'))

    for i, r in enumerate(pairs):
        bpb = r['best_bpb']
        if bpb and gpt_bpb:
            improvement = gpt_bpb - bpb
            pct = improvement / gpt_bpb * 100
            bar = '#' * max(1, int(pct * 5))
            print(f'  {i + 1}. {r["name"]:<18} {pct:>+5.1f}% {bar}')

    # Interaction analysis
    if gpt_bpb and llama_bpb:
        best_pair = pairs[0]
        best_pair_improvement = gpt_bpb - best_pair['best_bpb']
        full_improvement = gpt_bpb - llama_bpb
        ratio = best_pair_improvement / full_improvement * 100

        print(
            f'\n  Best pair accounts for {ratio:.1f}% of '
            f'full LLaMA improvement'
        )
        print(
            f'  Best pair: {best_pair["name"]} '
            f'({best_pair_improvement:.4f} BPB)'
        )
        print(f'  Full LLaMA: {full_improvement:.4f} BPB')


if __name__ == '__main__':
    main()
