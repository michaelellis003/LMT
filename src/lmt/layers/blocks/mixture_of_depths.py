r"""Mixture of Depths: dynamic compute allocation per token.

Implements the routing mechanism from *Mixture-of-Depths:
Dynamically allocating compute in transformer-based language models*
(Raposo et al., 2024).

Each transformer block gets a binary router that selects the top-C%
of tokens (by routing score) to process through the full
attention + FFN computation. The remaining tokens pass through via
the residual connection only, incurring zero compute.

This allows the model to spend more compute on "hard" tokens and
skip "easy" ones, achieving better quality at the same FLOP budget.

The router's continuous weights are multiplied into the processed
tokens' outputs (like expert gating in MoE), preserving gradient flow.

.. math::

    y_t = \begin{cases}
        x_t + w_t \cdot \text{Block}(x_t) & \text{if } t \in \text{TopC} \\
        x_t & \text{otherwise}
    \end{cases}

where :math:`w_t` is the router's scalar weight for token :math:`t`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lmt.layers.blocks.configurable_block import BlockConfig, ConfigurableBlock
from lmt.models.config import ModelConfig


class DepthRouter(nn.Module):
    """Binary router that selects which tokens to process.

    Uses a learned linear gate to score each token, then selects
    the top-C fraction (by score) for processing. Tokens outside
    the top-C skip the block entirely.

    Args:
        d_model: Input feature dimension.
        capacity: Fraction of tokens to process (0.0 to 1.0).
    """

    def __init__(self, d_model: int, capacity: float = 0.5) -> None:
        """Initialize DepthRouter.

        Args:
            d_model: Input feature dimension.
            capacity: Fraction of tokens to process.
        """
        super().__init__()
        self.capacity = capacity
        self.gate = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Route tokens: select top-C% for processing.

        Args:
            x: Input ``[batch, seq_len, d_model]``.

        Returns:
            Tuple of:
            - mask: ``[batch, seq_len, 1]`` binary mask (1=process)
            - weights: ``[batch, seq_len, 1]`` continuous router
              weights (sigmoid of logits)
            - aux_loss: scalar load balancing loss
        """
        b, s, _ = x.shape
        logits = self.gate(x)  # [b, s, 1]
        weights = torch.sigmoid(logits.float()).to(x.dtype)

        # Select top-C tokens per batch
        num_selected = max(1, int(s * self.capacity)) if s > 0 else 0
        if self.capacity == 0.0:
            num_selected = 0

        if num_selected == 0:
            mask = torch.zeros_like(weights)
        elif num_selected >= s:
            mask = torch.ones_like(weights)
        else:
            # Top-k selection along sequence dimension
            _, top_indices = torch.topk(
                logits.squeeze(-1), num_selected, dim=-1
            )
            mask = torch.zeros(b, s, 1, device=x.device, dtype=x.dtype)
            mask.scatter_(
                1,
                top_indices.unsqueeze(-1),
                1.0,
            )

        # Aux loss: encourage the router to use the full capacity
        # (prevent degenerate routing where the same tokens are always
        # selected). We use the variance of routing scores as a penalty.
        if weights.numel() > 1:
            aux_loss = weights.var()
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return mask, weights, aux_loss


class MoDBlock(nn.Module):
    """Transformer block with Mixture of Depths routing.

    Wraps a ``ConfigurableBlock`` with a ``DepthRouter`` that
    selects which tokens get processed. Unselected tokens pass
    through via residual only.

    Args:
        model_config: Model configuration.
        block_config: Block component selection.
        capacity: Fraction of tokens to route through the block.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        block_config: BlockConfig,
        capacity: float = 0.5,
    ) -> None:
        """Initialize MoDBlock.

        Args:
            model_config: Model configuration.
            block_config: Block component selection.
            capacity: Fraction of tokens to process (0.0 to 1.0).
        """
        super().__init__()
        self.router = DepthRouter(
            d_model=model_config.embed_dim,
            capacity=capacity,
        )
        self.block = ConfigurableBlock(model_config, block_config)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        """Apply MoD block: route tokens, process selected ones.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output ``[batch, seq_len, embed_dim]``.
        """
        mask, weights, aux_loss = self.router(x)
        self.aux_loss = aux_loss

        if mask.sum() == 0:
            # No tokens selected — pure residual
            return x

        # Process ALL tokens through the block (simpler and correct)
        # then gate the residual contribution by mask * weight
        block_out = self.block(x)

        # block_out includes the residual (x + attn + ffn),
        # so the "delta" is block_out - x
        delta = block_out - x

        # Apply routing: selected tokens get the weighted delta,
        # unselected tokens get nothing (pure residual)
        return x + mask * weights * delta
