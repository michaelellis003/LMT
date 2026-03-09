"""WikiText-2 pretraining recipe.

Trains a small Qwen3 model on WikiText-2 using character-level
tokenization. Demonstrates the full LMT training pipeline:

- Data: WikiText-2 from HuggingFace Hub
- Tokenizer: Character-level (simple, no dependencies)
- Model: Qwen3 with ~90K parameters
- Training: AdamW with cosine warmup, weight decay param groups
- Evaluation: val_bpb at regular intervals
- Early stopping: stops if val_bpb plateaus

Usage::

    uv run python recipes/wikitext_pretrain.py

Runs in ~60 seconds on CPU. Expect val_bpb to decrease from ~5.5
to ~3.5 as the model learns character-level patterns.

Note: Character-level BPB on WikiText-2 is inherently higher than
subword BPB because each character is 1 byte (bytes_per_token=1.0),
while subword tokens cover multiple bytes.
"""

import math

import torch
from torch.nn import functional as f

from lmt.data.text_dataset import TextDataset, texts_from_dicts
from lmt.eval.bpb import compute_bpb
from lmt.models.config import ModelConfig
from lmt.models.qwen3.qwen3 import Qwen3
from lmt.tokenizer.char import CharTokenizer
from lmt.training.early_stopping import EarlyStopping
from lmt.training.lr_schedule import get_cosine_warmup_scheduler
from lmt.training.param_groups import get_param_groups
from lmt.training.reproducibility import set_seed


def load_wikitext2() -> tuple[list[str], list[str]]:
    """Download WikiText-2 and return train/val text lists."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError('Install datasets: uv pip install datasets') from e

    train_ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    val_ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    return (
        texts_from_dicts(train_ds, text_field='text'),
        texts_from_dicts(val_ds, text_field='text'),
    )


def main() -> None:
    """Run WikiText-2 pretraining."""
    set_seed(42)
    print('WikiText-2 Pretraining Recipe')
    print('=' * 50)

    # 1. Load data
    print('\n1. Loading WikiText-2...')
    train_texts, val_texts = load_wikitext2()
    print(f'   Train: {len(train_texts)} documents')
    print(f'   Val:   {len(val_texts)} documents')

    # 2. Build tokenizer from training text
    print('\n2. Building character tokenizer...')
    all_text = '\n'.join(train_texts + val_texts)
    tokenizer = CharTokenizer.from_text(all_text)
    print(f'   Vocab size: {tokenizer.vocab_size}')

    # 3. Tokenize and pack
    context_length = 128
    print(f'\n3. Tokenizing (context_length={context_length})...')
    train_dataset = TextDataset(
        train_texts, tokenizer, context_length=context_length
    )
    val_dataset = TextDataset(
        val_texts, tokenizer, context_length=context_length
    )
    print(f'   Train sequences: {len(train_dataset)}')
    print(f'   Val sequences:   {len(val_dataset)}')

    # Convert to tensors
    train_data = torch.stack(
        [train_dataset[i] for i in range(len(train_dataset))]
    )
    val_data = torch.stack(
        [val_dataset[i] for i in range(min(200, len(val_dataset)))]
    )

    # 4. Create model
    print('\n4. Creating model...')
    config = ModelConfig(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=4,
        num_layers=4,
        ffn_hidden_dim=128,
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        tie_weights=True,
        dropout=0.0,
    )
    model = Qwen3(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'   Parameters: {num_params:,}')

    # 5. Set up training
    lr = 1e-3
    total_steps = 500
    warmup_steps = 50
    batch_size = 32
    eval_interval = 50

    print('\n5. Training config:')
    print(f'   lr={lr}, steps={total_steps}, warmup={warmup_steps}')
    print(f'   batch_size={batch_size}, eval_interval={eval_interval}')

    param_groups = get_param_groups(model, weight_decay=0.1, lr=lr)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    scheduler = get_cosine_warmup_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    early_stop = EarlyStopping(patience=5, min_delta=0.01)

    # 6. Train
    print('\n6. Training...')
    model.train()
    num_samples = train_data.shape[0]
    best_bpb = float('inf')

    for step in range(total_steps):
        indices = torch.randint(0, num_samples, (batch_size,))
        batch = train_data[indices]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = f.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % eval_interval == 0:
            model.eval()
            val_bpb = compute_bpb(model, val_data, bytes_per_token=1.0)
            model.train()

            marker = ''
            if val_bpb < best_bpb:
                best_bpb = val_bpb
                marker = ' *'
            current_lr = scheduler.get_last_lr()[0]
            print(
                f'   Step {step + 1:4d} | '
                f'loss={loss.item():.4f} | '
                f'val_bpb={val_bpb:.4f} | '
                f'lr={current_lr:.2e}{marker}'
            )

            if early_stop.step(val_bpb):
                print(f'   Early stopping at step {step + 1}')
                break

    # 7. Final evaluation
    print('\n7. Results:')
    model.eval()
    final_bpb = compute_bpb(model, val_data, bytes_per_token=1.0)
    print(f'   Final val_bpb: {final_bpb:.4f}')
    print(f'   Best val_bpb:  {best_bpb:.4f}')

    # For reference: random baseline
    random_bpb = math.log2(tokenizer.vocab_size)
    print(f'   Random baseline: {random_bpb:.4f}')
    print(
        f'   Improvement over random: '
        f'{(1 - final_bpb / random_bpb) * 100:.1f}%'
    )


if __name__ == '__main__':
    main()
