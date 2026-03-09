"""3-seed noise floor check: does the GPT vs LLaMA gap survive replication?

First real use of the Bayesian comparison framework. Tests whether the
~11% BPB gap (3.56 vs 3.17) from our single-seed architecture comparison
is detectable across 3 seeds.

This is NOT a rigorous comparison — 3 seeds is below the minimum for
formal inference. It's a calibration experiment: if 3 seeds can't even
agree on which architecture is better when the gap is 11%, our noise
floor is too high for any of our ablation results to be meaningful.

Usage::

    .venv/bin/python experiments/noise_floor_check.py

Expected: ~15-30 min on CPU (3 seeds × 2 architectures × ~3 min each).
"""

import dataclasses

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.data.hf_datasets import download_wikitext2
from lmt.data.text_dataset import TextDataset
from lmt.eval.bpb import compute_bpb
from lmt.models.config import ModelConfig
from lmt.models.factory import create_model
from lmt.research.multi_seed import (
    MultiSeedConfig,
    compare_variants,
    run_multi_seed,
)
from lmt.research.stats import multi_seed_summary
from lmt.tokenizer.char import CharTokenizer
from lmt.training.reproducibility import set_seed
from lmt.training.train_config import TrainConfig
from lmt.training.train_loop import train_loop

# --- Configuration ---

MAX_TRAIN_SEQS = 2000
MAX_VAL_SEQS = 200
MAX_ITEMS = 5000  # Limit HF download for memory safety
BATCH_SIZE = 32
EVAL_INTERVAL = 100
TRAIN_STEPS = 500
ROPE_BPB = 0.05  # ROPE: differences < 0.05 BPB are "equivalent"

SEED_CONFIG = MultiSeedConfig(n_seeds=3, seeds=[42, 137, 256])


def setup_data() -> tuple[ModelConfig, torch.Tensor, torch.Tensor]:
    """Load WikiText-2 data with memory-safe limits."""
    print('Loading WikiText-2...')
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
    return config, train_data, val_data


def make_train_fn(
    arch: str,
    config: ModelConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> callable:
    """Create a train function for run_multi_seed."""
    train_config = TrainConfig(
        lr=1e-3,
        total_steps=TRAIN_STEPS,
        warmup_steps=50,
        weight_decay=0.1,
        max_grad_norm=1.0,
    )

    def loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        return f.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

    def train_fn(seed: int) -> float:
        set_seed(seed)
        model = create_model(arch, dataclasses.replace(config))

        best_bpb = float('inf')

        def eval_fn(m: nn.Module, step: int) -> dict[str, float]:
            nonlocal best_bpb
            m.eval()
            bpb = compute_bpb(m, val_data, bytes_per_token=1.0)
            best_bpb = min(best_bpb, bpb)
            return {'val_bpb': bpb}

        train_loop(
            model=model,
            train_data=train_data,
            loss_fn=loss_fn,
            config=train_config,
            batch_size=BATCH_SIZE,
            eval_fn=eval_fn,
            eval_interval=EVAL_INTERVAL,
        )

        del model
        return best_bpb

    return train_fn


def main() -> None:
    """Run 3-seed noise floor check."""
    print('=' * 60)
    print('Noise Floor Check: GPT vs LLaMA (3 seeds)')
    print('=' * 60)
    print(f'Seeds: {SEED_CONFIG.seeds}')
    print(f'ROPE: ±{ROPE_BPB} BPB')
    print()

    config, train_data, val_data = setup_data()

    # Run GPT
    print('\n--- GPT ---')
    gpt_fn = make_train_fn('gpt', config, train_data, val_data)
    gpt_result = run_multi_seed('gpt', gpt_fn, SEED_CONFIG)

    # Run LLaMA
    print('\n--- LLaMA ---')
    llama_fn = make_train_fn('llama', config, train_data, val_data)
    llama_result = run_multi_seed('llama', llama_fn, SEED_CONFIG)

    # Bayesian comparison
    print('\n' + '=' * 60)
    print('BAYESIAN COMPARISON')
    print('=' * 60)

    comparison = compare_variants(llama_result, gpt_result, rope=ROPE_BPB)

    gpt_summary = multi_seed_summary(gpt_result.to_samples())
    llama_summary = multi_seed_summary(llama_result.to_samples())

    print(
        f'\nGPT:   {gpt_summary["mean"]:.4f} ± {gpt_summary["std"]:.4f} '
        f'(n={gpt_summary["n_seeds"]})'
    )
    print(
        f'LLaMA: {llama_summary["mean"]:.4f} ± {llama_summary["std"]:.4f} '
        f'(n={llama_summary["n_seeds"]})'
    )

    print(f'\nP(LLaMA better):     {comparison.prob_a_better:.3f}')
    print(f'P(equivalent):       {comparison.prob_rope:.3f}')
    print(f'P(GPT better):       {comparison.prob_b_better:.3f}')
    print(f"Cohen's d:           {comparison.cohens_d:.2f}")
    print(f'Mean diff:           {comparison.mean_difference:.4f}')
    print(
        f'95% CI:              [{comparison.ci_lower:.4f}, '
        f'{comparison.ci_upper:.4f}]'
    )
    print(f'\nInterpretation: {comparison.interpretation}')

    # Per-seed breakdown
    print('\n--- Per-seed BPB ---')
    print(f'{"Seed":>6}  {"GPT":>8}  {"LLaMA":>8}  {"Diff":>8}')
    for g, ll in zip(
        gpt_result.seed_results,
        llama_result.seed_results,
        strict=True,
    ):
        diff = ll.best_bpb - g.best_bpb
        print(
            f'{g.seed:>6}  {g.best_bpb:>8.4f}  {ll.best_bpb:>8.4f}'
            f'  {diff:>+8.4f}'
        )


if __name__ == '__main__':
    main()
