"""Architecture comparison on WikiText-2 (real language data).

Controlled comparison of GPT, LLaMA, Qwen3, Gemma, and Mixtral on
WikiText-2 using character-level tokenization. Unlike the random data
experiment, real text has learnable structure, so architecture differences
may emerge even at toy scale.

Hypothesis: On real data, modern architectures (LLaMA, Qwen3, Gemma)
with RoPE and SwiGLU should have a small but measurable advantage over
vanilla GPT, because these features help even at small scale when there
are real patterns to learn.

Usage::

    uv run python experiments/arch_comparison_wikitext.py

Results are saved to experiments/results/arch_comparison_wikitext/
and a summary table is printed at the end.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.data.hf_datasets import download_wikitext2
from lmt.data.text_dataset import TextDataset
from lmt.eval.bpb import compute_bpb
from lmt.models.config import ModelConfig
from lmt.models.factory import create_model
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

MODEL_CONFIG = ModelConfig(
    embed_dim=64,
    num_heads=4,
    num_kv_heads=2,
    num_layers=4,
    ffn_hidden_dim=128,
    vocab_size=256,  # Will be overridden by tokenizer vocab size
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

ARCHITECTURES = ['gpt', 'llama', 'qwen3', 'gemma', 'mixtral']
BATCH_SIZE = 32
EVAL_INTERVAL = 100
SEED = 42
OUTPUT_DIR = Path('experiments/results/arch_comparison_wikitext')


def load_data(
    tokenizer: CharTokenizer, context_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and tokenize WikiText-2 data."""
    print('  Loading WikiText-2...')
    train_texts = download_wikitext2(split='train')
    val_texts = download_wikitext2(split='validation')
    print(f'  Train: {len(train_texts)} paragraphs')
    print(f'  Val:   {len(val_texts)} paragraphs')

    train_dataset = TextDataset(
        train_texts, tokenizer, context_length=context_length
    )
    val_dataset = TextDataset(
        val_texts, tokenizer, context_length=context_length
    )
    print(f'  Train sequences: {len(train_dataset)}')
    print(f'  Val sequences:   {len(val_dataset)}')

    train_data = torch.stack(
        [train_dataset[i] for i in range(len(train_dataset))]
    )
    val_limit = min(500, len(val_dataset))
    val_data = torch.stack([val_dataset[i] for i in range(val_limit)])
    return train_data, val_data


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
    arch: str,
    config: ModelConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> dict:
    """Train one architecture and return results."""
    set_seed(SEED)

    model = create_model(arch, config)
    profile = profile_model(model)

    print(f'\n  {arch.upper()}')
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
        'arch': arch,
        'params': profile.total_params,
        'final_loss': state.final_loss,
        'final_bpb': final_bpb,
        'best_bpb': best_bpb,
        'elapsed_s': elapsed,
        'losses': state.losses,
        'eval_results': eval_results,
    }


def main() -> None:
    """Run architecture comparison on WikiText-2."""
    print('=' * 60)
    print('Architecture Comparison on WikiText-2')
    print('=' * 60)
    print(
        f'Config: embed_dim={MODEL_CONFIG.embed_dim}, '
        f'layers={MODEL_CONFIG.num_layers}, '
        f'steps={TRAIN_CONFIG.total_steps}'
    )
    print(f'Architectures: {", ".join(ARCHITECTURES)}')

    # Build character tokenizer from WikiText-2
    print('\n  Building character tokenizer...')
    all_texts = download_wikitext2(split='train')
    val_texts = download_wikitext2(split='validation')
    all_text = '\n'.join(all_texts + val_texts)  # Full corpus for vocab
    tokenizer = CharTokenizer.from_text(all_text)
    print(f'  Vocab size: {tokenizer.vocab_size}')

    # Override vocab size in config
    config = ModelConfig(
        embed_dim=MODEL_CONFIG.embed_dim,
        num_heads=MODEL_CONFIG.num_heads,
        num_kv_heads=MODEL_CONFIG.num_kv_heads,
        num_layers=MODEL_CONFIG.num_layers,
        ffn_hidden_dim=MODEL_CONFIG.ffn_hidden_dim,
        vocab_size=tokenizer.vocab_size,
        context_length=MODEL_CONFIG.context_length,
        tie_weights=MODEL_CONFIG.tie_weights,
        qk_norm=MODEL_CONFIG.qk_norm,
        dropout=MODEL_CONFIG.dropout,
    )

    # Load data
    train_data, val_data = load_data(tokenizer, config.context_length)

    # Run experiments
    results = []
    registry = ExperimentRegistry(OUTPUT_DIR / 'registry.jsonl')

    for arch in ARCHITECTURES:
        result = run_single(arch, config, train_data, val_data)
        results.append(result)

        tracker = MetricTracker()
        for step, bpb in result['eval_results']:
            tracker.log('val_bpb', bpb, step=step)
        for i, loss_val in enumerate(result['losses']):
            tracker.log('train_loss', loss_val, step=i)

        exp_config = ExperimentRunConfig(
            name=f'wikitext_{arch}',
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

    # Print summary table
    print('\n' + '=' * 60)
    print('RESULTS SUMMARY (WikiText-2, character-level)')
    print('=' * 60)
    header = (
        f'{"Arch":<10} {"Params":>8} {"Loss":>8} '
        f'{"BPB":>8} {"Best BPB":>10} {"Time":>6}'
    )
    print(header)
    print('-' * 60)

    results.sort(key=lambda r: r['best_bpb'] or float('inf'))

    for r in results:
        bpb_str = f'{r["final_bpb"]:.4f}' if r['final_bpb'] else '-'
        best_str = f'{r["best_bpb"]:.4f}' if r['best_bpb'] else '-'
        print(
            f'{r["arch"]:<10} {r["params"]:>8,} '
            f'{r["final_loss"]:>8.4f} {bpb_str:>8} '
            f'{best_str:>10} {r["elapsed_s"]:>5.1f}s'
        )

    # Analysis
    bpbs = [r['best_bpb'] for r in results if r['best_bpb']]
    if bpbs:
        spread = max(bpbs) - min(bpbs)
        mean_bpb = sum(bpbs) / len(bpbs)
        print(
            f'\nBPB range: {min(bpbs):.4f} - {max(bpbs):.4f} '
            f'(spread: {spread:.4f}, {spread / mean_bpb * 100:.1f}%)'
        )

        if spread / mean_bpb < 0.05:
            print('=> Architecture choice has <5% effect at this scale')
        else:
            print('=> Architecture choice matters even at toy scale!')
            print(f'   Winner: {results[0]["arch"]}')

    print(f'\nResults saved to: {OUTPUT_DIR / "registry.jsonl"}')
    print(registry.summary(metric='val_bpb'))


if __name__ == '__main__':
    main()
