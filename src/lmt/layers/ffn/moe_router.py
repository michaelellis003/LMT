r"""Top-K Router for Mixture of Experts.

Implements the gating mechanism that selects which experts
process each token:

.. math::

    G(x) = \text{TopK}\bigl(\text{softmax}(x W_g),\; k\bigr)

The load balancing auxiliary loss encourages even expert usage:

.. math::

    \mathcal{L}_{\text{bal}} = N \sum_{i=1}^{N} f_i \cdot P_i

where :math:`f_i` is the fraction of tokens routed to expert
:math:`i` and :math:`P_i` is the mean routing probability for
expert :math:`i`.

References:
    Shazeer et al., 2017 -- "Outrageously Large Neural Networks:
    The Sparsely-Gated Mixture-of-Experts Layer"
    https://arxiv.org/abs/1701.06538
    Fedus et al., 2022 -- "Switch Transformers: Scaling to Trillion
    Parameter Models with Simple and Efficient Sparsity"
    https://arxiv.org/abs/2101.03961
"""

import torch
import torch.nn as nn
from torch import Tensor


class TopKRouter(nn.Module):
    """Top-K expert router with load balancing loss.

    Args:
        num_experts: Number of experts to route between.
        d_model: Input feature dimension.
        top_k: Number of experts selected per token.
    """

    def __init__(self, num_experts: int, d_model: int, top_k: int) -> None:
        """Initialize TopKRouter.

        Args:
            num_experts: Total number of experts.
            d_model: Input feature dimension.
            top_k: Experts activated per token.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Route tokens to top-k experts.

        Args:
            x: Input ``[batch, seq_len, d_model]``.

        Returns:
            Tuple of:
            - gate_values: ``[batch*seq, top_k]`` renormalized weights
            - expert_indices: ``[batch*seq, top_k]`` selected experts
            - aux_loss: scalar load balancing loss
        """
        b, s, _ = x.shape
        x_flat = x.reshape(b * s, -1)  # [N, d_model]

        # Compute routing probabilities
        logits = self.gate(x_flat)  # [N, num_experts]
        probs = torch.softmax(logits.float(), dim=-1)

        # Select top-k experts
        top_k_values, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Renormalize gate values to sum to 1
        gate_values = top_k_values / top_k_values.sum(dim=-1, keepdim=True)

        # Load balancing auxiliary loss
        # f_i: fraction of tokens assigned to expert i
        num_tokens = x_flat.shape[0]
        expert_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            device=x.device,
            dtype=probs.dtype,
        )
        expert_mask.scatter_(1, top_k_indices, 1.0)
        f = expert_mask.mean(dim=0)  # [num_experts]

        # P_i: mean routing probability for expert i
        p = probs.mean(dim=0)  # [num_experts]

        aux_loss = self.num_experts * (f * p).sum()

        return gate_values.to(x.dtype), top_k_indices, aux_loss
