r"""Multi-Token Prediction head.

Wraps any :class:`BaseModel` to predict multiple future tokens
simultaneously, as described in *Better & Faster Large Language Models
via Multi-token Prediction* (Gloeckle et al., 2024).

Instead of a single next-token prediction head, MTP adds *N* heads
that predict tokens at positions :math:`t+1, t+2, \ldots, t+N`.
All heads share the backbone's unembedding matrix. Each additional
head gets a lightweight "future transformer" layer that processes the
backbone's hidden states concatenated with the embedding of the
previous head's target token (teacher-forced during training).

During training, the total loss is a weighted sum across heads:

.. math::

    \mathcal{L} = \sum_{k=0}^{N-1} w_k \, \text{CE}(\hat{y}^{(k)}, y_{t+k+1})

At inference, only head 0 is used (or extra heads enable speculative
decoding).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as f

from lmt.models.base import BaseModel


class FutureTransformerLayer(nn.Module):
    """A lightweight transformer layer for future-token prediction.

    Takes the concatenation of backbone hidden states and the previous
    head's token embedding, projects it, and applies a single
    self-attention + FFN block.

    Args:
        embed_dim: Model embedding dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Initialize future transformer layer.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.input_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, hidden: Tensor, prev_embed: Tensor) -> Tensor:
        """Apply future transformer layer.

        Args:
            hidden: Backbone hidden states ``[batch, seq, embed_dim]``.
            prev_embed: Embedding of previous head's target
                ``[batch, seq, embed_dim]``.

        Returns:
            Transformed hidden states ``[batch, seq, embed_dim]``.
        """
        x = self.input_proj(torch.cat([hidden, prev_embed], dim=-1))
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class MultiTokenPredictionHead(nn.Module):
    r"""Multi-Token Prediction wrapper for any BaseModel.

    Wraps an existing language model to produce *N* sets of logits,
    one for each future position. Head 0 is the standard next-token
    prediction; heads 1..N-1 use lightweight ``FutureTransformerLayer``
    modules to predict further ahead.

    Args:
        model: A :class:`BaseModel` (or subclass) to wrap.
        num_heads: Number of prediction heads (N). Default 4.
    """

    def __init__(
        self,
        model: BaseModel,
        num_heads: int = 4,
    ) -> None:
        """Initialize Multi-Token Prediction head.

        Args:
            model: The backbone language model.
            num_heads: Number of future tokens to predict.
        """
        super().__init__()
        self.model = model
        self.num_heads = num_heads
        self.shared_out_head = model.out_head

        embed_dim = model.config.embed_dim
        attn_heads = model.config.num_heads

        self.future_layers = nn.ModuleList(
            [
                FutureTransformerLayer(embed_dim, attn_heads)
                for _ in range(num_heads - 1)
            ]
        )

    def forward(
        self,
        in_idx: Tensor,
        targets: Tensor | None = None,
    ) -> list[Tensor]:
        """Forward pass producing logits for each prediction head.

        During training, ``targets`` provides the ground-truth token
        IDs used for teacher-forcing the future heads. During
        inference (no targets), each future head uses the argmax
        of the previous head's logits.

        Args:
            in_idx: Input token indices ``[batch, seq_len]``.
            targets: Target token indices ``[batch, seq_len]`` for
                teacher forcing. If None, uses greedy predictions
                from prior heads.

        Returns:
            List of N logit tensors, each ``[batch, seq_len, vocab]``.
        """
        hidden = self.model.forward_hidden(in_idx)
        logits_list: list[Tensor] = []

        # Head 0: standard next-token prediction
        logits_0 = self.shared_out_head(hidden)
        logits_list.append(logits_0)

        # Heads 1..N-1: future prediction
        prev_hidden = hidden
        for _k, layer in enumerate(self.future_layers):
            if targets is not None:
                # Teacher forcing: embed the target tokens as
                # context for the next prediction head
                prev_tokens = targets
            else:
                # Inference: use argmax of previous head
                prev_tokens = logits_list[-1].argmax(dim=-1)

            prev_embed = self.model.tok_embed(prev_tokens)
            prev_hidden = layer(prev_hidden, prev_embed)
            logits_k = self.shared_out_head(prev_hidden)
            logits_list.append(logits_k)

        return logits_list

    def compute_loss(
        self,
        logits_list: list[Tensor],
        targets: Tensor,
        head_weights: list[float] | None = None,
    ) -> Tensor:
        """Compute weighted cross-entropy loss across all heads.

        For head k, the target is the token at position t+k+1.
        Since we may not have targets beyond the sequence length,
        the loss for head k is computed only on positions where
        a valid target exists.

        Args:
            logits_list: List of N logit tensors from :meth:`forward`.
            targets: Target token indices ``[batch, seq_len]``.
            head_weights: Optional per-head loss weights. Defaults
                to equal weights (1/N each).

        Returns:
            Scalar loss tensor.
        """
        n = len(logits_list)

        if head_weights is None:
            head_weights = [1.0 / n] * n

        total_loss = torch.tensor(0.0, device=targets.device)

        for k, (logits, weight) in enumerate(
            zip(logits_list, head_weights, strict=True)
        ):
            if weight == 0.0:
                continue

            # For head k, target is shifted by k positions:
            # head 0 predicts targets[t], head 1 predicts targets[t+1], etc.
            # We handle this by truncating logits and targets appropriately
            seq_len = targets.shape[1]
            valid_len = seq_len - k

            if valid_len <= 0:
                continue

            # logits[:, :valid_len] predicts targets[:, k:k+valid_len]
            head_logits = logits[:, :valid_len].contiguous()
            head_targets = targets[:, k : k + valid_len].contiguous()

            loss_k = f.cross_entropy(
                head_logits.view(-1, head_logits.shape[-1]),
                head_targets.view(-1),
            )
            total_loss = total_loss + weight * loss_k

        return total_loss
