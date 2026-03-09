r"""Muon vs AdamW ablation on a tiny language model.

Compares the Muon optimizer against AdamW on the same model architecture
and random data. This tests whether Muon's orthogonal updates provide
any advantage at small scale (~50K params).

Hypothesis: At this tiny scale, both optimizers should achieve similar
final loss, but Muon may converge faster due to better-conditioned updates.

Usage::

    uv run python experiments/muon_vs_adamw.py
"""

import torch
from torch.nn import functional as f

from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig
from lmt.training.muon import Muon
from lmt.training.reproducibility import set_seed


def create_model(seed: int) -> BaseModel:
    """Create a tiny model with a fixed seed."""
    set_seed(seed)
    config = ModelConfig(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        ffn_hidden_dim=128,
        vocab_size=256,
        context_length=64,
        tie_weights=True,
        qk_norm=True,
        dropout=0.0,
    )
    return BaseModel(config)


def train_with_optimizer(
    optimizer_name: str,
    seed: int,
    train_steps: int = 100,
    batch_size: int = 8,
) -> list[float]:
    """Train a model with the given optimizer and return loss curve."""
    model = create_model(seed)
    set_seed(seed)

    # Create random training data
    data = torch.randint(0, 256, (200, 65))  # 200 sequences of length 65

    if optimizer_name == 'muon':
        # Muon for hidden weights, AdamW for embeddings
        hidden_params = [
            p
            for n, p in model.named_parameters()
            if p.ndim >= 2 and 'tok_embed' not in n and 'out_head' not in n
        ]
        other_params = [
            p
            for n, p in model.named_parameters()
            if p.ndim < 2 or 'tok_embed' in n or 'out_head' in n
        ]
        muon_opt = Muon(hidden_params, lr=0.02)
        adam_opt = torch.optim.AdamW(other_params, lr=3e-4)
        optimizers = [muon_opt, adam_opt]
    else:
        optimizers = [torch.optim.AdamW(model.parameters(), lr=3e-4)]

    model.train()
    losses = []

    for _step in range(train_steps):
        # Sample batch
        indices = torch.randint(0, len(data), (batch_size,))
        batch = data[indices]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = f.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()

        losses.append(loss.item())

    return losses


def main() -> None:
    """Run the ablation."""
    print('=== Muon vs AdamW Ablation ===')
    print()

    n_seeds = 3
    train_steps = 200

    adamw_results = []
    muon_results = []

    for seed in range(n_seeds):
        print(f'Seed {seed}...')
        adamw_losses = train_with_optimizer(
            'adamw', seed=seed, train_steps=train_steps
        )
        muon_losses = train_with_optimizer(
            'muon', seed=seed, train_steps=train_steps
        )
        adamw_results.append(adamw_losses)
        muon_results.append(muon_losses)

        print(
            f'  AdamW final loss: {adamw_losses[-1]:.4f}, '
            f'Muon final loss: {muon_losses[-1]:.4f}'
        )

    # Average final losses
    adamw_final = sum(r[-1] for r in adamw_results) / n_seeds
    muon_final = sum(r[-1] for r in muon_results) / n_seeds

    # Average losses at step 50 (early convergence check)
    adamw_early = sum(r[49] for r in adamw_results) / n_seeds
    muon_early = sum(r[49] for r in muon_results) / n_seeds

    print()
    print(f'Average final loss (step {train_steps}):')
    print(f'  AdamW: {adamw_final:.4f}')
    print(f'  Muon:  {muon_final:.4f}')
    print(f'  Diff:  {abs(adamw_final - muon_final):.4f}')
    print()
    print('Average loss at step 50 (early convergence):')
    print(f'  AdamW: {adamw_early:.4f}')
    print(f'  Muon:  {muon_early:.4f}')
    print()

    if abs(adamw_final - muon_final) < 0.1:
        print('Result: No meaningful difference at this scale.')
        print(
            'Expected — orthogonal updates matter more '
            'with larger weight matrices.'
        )
    elif muon_final < adamw_final:
        print('Result: Muon achieved lower final loss.')
    else:
        print('Result: AdamW achieved lower final loss.')


if __name__ == '__main__':
    main()
