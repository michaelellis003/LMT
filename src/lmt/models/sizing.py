"""Model sizing utilities for parameter count estimation.

Helps design model configurations to hit a target parameter count
without instantiating the model. Useful for planning experiments
and meeting competition constraints (e.g., BabyLM parameter limits).

Usage::

    from lmt.models.sizing import estimate_params, suggest_config

    # Estimate params for a specific config
    params = estimate_params(
        embed_dim=512,
        num_layers=12,
        ffn_hidden_dim=2048,
        vocab_size=32000,
        tie_weights=True,
    )
    print(f'{params:,} parameters')

    # Find a config near a target size
    config = suggest_config(target_params=10_000_000, vocab_size=32000)
    print(config)
    # {'embed_dim': 384, 'num_layers': 8, 'ffn_hidden_dim': 1024}

The estimates assume a standard transformer architecture with:
- Token embeddings (vocab_size x embed_dim)
- Self-attention (4 x embed_dim^2 per layer: Q, K, V, O projections)
- FFN (2 x embed_dim x ffn_hidden_dim per layer: up + down projections)
- Layer norms (2 x embed_dim per layer)
- Optional LM head (vocab_size x embed_dim, shared with embeddings if tied)
"""

from __future__ import annotations


def estimate_params(
    embed_dim: int,
    num_layers: int,
    ffn_hidden_dim: int,
    vocab_size: int,
    tie_weights: bool = True,
) -> int:
    """Estimate total parameter count for a transformer model.

    Uses a simplified formula assuming standard architecture:
    - Self-attention: 4 * embed_dim^2 per layer (Q, K, V, O)
    - FFN: 2 * embed_dim * ffn_hidden_dim per layer (up + down)
    - Layer norms: 2 * embed_dim per layer (attention + FFN norms)
    - Embeddings: vocab_size * embed_dim
    - LM head: vocab_size * embed_dim (unless tied)

    Args:
        embed_dim: Model embedding dimension.
        num_layers: Number of transformer layers.
        ffn_hidden_dim: FFN intermediate dimension.
        vocab_size: Vocabulary size.
        tie_weights: If True, LM head shares embedding weights.

    Returns:
        Estimated parameter count (integer).
    """
    # Embedding layer
    embedding = vocab_size * embed_dim

    # Per-layer parameters
    attention = 4 * embed_dim * embed_dim  # Q, K, V, O
    ffn = 2 * embed_dim * ffn_hidden_dim  # up + down projections
    layer_norms = 2 * embed_dim  # attention norm + FFN norm
    per_layer = attention + ffn + layer_norms

    # Total
    total = embedding + num_layers * per_layer

    # LM head (if not tied with embeddings)
    if not tie_weights:
        total += vocab_size * embed_dim

    # Final layer norm
    total += embed_dim

    return total


def suggest_config(
    target_params: int,
    vocab_size: int,
    tie_weights: bool = True,
) -> dict[str, int]:
    """Suggest model dimensions to hit a target parameter count.

    Searches over common dimension/layer combinations to find one
    close to the target. Uses heuristic: ffn_hidden_dim = 4 * embed_dim,
    num_heads = embed_dim // 64.

    Args:
        target_params: Target total parameter count.
        vocab_size: Vocabulary size.
        tie_weights: Whether embeddings are tied.

    Returns:
        Dict with ``embed_dim``, ``num_layers``, ``ffn_hidden_dim``,
        and ``num_heads`` keys.
    """
    best_config: dict[str, int] | None = None
    best_diff = float('inf')

    # Search over common embed_dim values (multiples of 64)
    for embed_dim in range(64, 4097, 64):
        ffn_hidden_dim = 4 * embed_dim

        # Estimate per-layer cost
        per_layer = (
            4 * embed_dim * embed_dim
            + 2 * embed_dim * ffn_hidden_dim
            + 2 * embed_dim
        )
        if per_layer == 0:
            continue

        # Embedding cost
        embed_cost = vocab_size * embed_dim
        if not tie_weights:
            embed_cost *= 2
        embed_cost += embed_dim  # final layer norm

        # Solve for num_layers
        remaining = target_params - embed_cost
        if remaining <= 0:
            continue
        num_layers = max(1, round(remaining / per_layer))

        # Compute actual params
        actual = estimate_params(
            embed_dim,
            num_layers,
            ffn_hidden_dim,
            vocab_size,
            tie_weights,
        )
        diff = abs(actual - target_params)

        if diff < best_diff:
            best_diff = diff
            num_heads = max(1, embed_dim // 64)
            best_config = {
                'embed_dim': embed_dim,
                'num_layers': num_layers,
                'ffn_hidden_dim': ffn_hidden_dim,
                'num_heads': num_heads,
            }

    if best_config is None:
        # Fallback for very small targets
        embed_dim = 64
        best_config = {
            'embed_dim': embed_dim,
            'num_layers': 1,
            'ffn_hidden_dim': 4 * embed_dim,
            'num_heads': 1,
        }

    return best_config
