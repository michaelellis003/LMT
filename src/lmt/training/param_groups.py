"""Weight decay parameter groups for AdamW optimization.

Separates model parameters into decay and no-decay groups following
standard transformer training practice (GPT-2, LLaMA, nanoGPT):

- **Decay**: 2D+ weight matrices (linear projections, etc.)
- **No decay**: biases, LayerNorm/RMSNorm weights, embeddings

Usage::

    from lmt.training.param_groups import get_param_groups

    groups = get_param_groups(model, weight_decay=0.1, lr=3e-4)
    optimizer = torch.optim.AdamW(groups)
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def get_param_groups(
    model: nn.Module,
    weight_decay: float,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Create AdamW parameter groups with proper weight decay.

    Parameters are split into two groups:

    - **decay**: 2D+ tensors that are not embeddings (weight matrices)
    - **no_decay**: 1D tensors (biases, norm weights) and embeddings

    Args:
        model: The model to create parameter groups for.
        weight_decay: Weight decay for the decay group.
        **kwargs: Additional optimizer kwargs (e.g. ``lr``) applied
            to both groups.

    Returns:
        List of two parameter group dicts suitable for
        ``torch.optim.AdamW``.
    """
    # Collect embedding parameter ids
    embedding_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters():
                embedding_ids.add(id(p))

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for param in model.parameters():
        if not param.requires_grad:
            continue

        # 1D params (biases, norm weights) and embeddings: no decay
        if param.dim() <= 1 or id(param) in embedding_ids:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {'params': decay_params, 'weight_decay': weight_decay, **kwargs},
        {'params': no_decay_params, 'weight_decay': 0.0, **kwargs},
    ]
    return groups
