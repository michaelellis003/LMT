r"""Mixture of Experts (MoE) feed-forward layer.

Each token is routed to the top-k experts out of N total, where
each expert is a SwiGLU FFN. Only the selected experts compute
their output for each token, making the layer "sparsely activated":

.. math::

    \text{MoE}(x) = \sum_{i \in \text{TopK}} g_i \cdot E_i(x)

where :math:`g_i` are the normalized gate weights from the router
and :math:`E_i` are the expert FFNs.

Optionally includes shared experts that process every token
(DeepSeek-style), providing a stable base representation
alongside the routed experts.

The auxiliary load balancing loss is stored as ``self.aux_loss``
and should be added to the training loss (weighted by a small
coefficient like 0.01).

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

from .moe_router import TopKRouter
from .swiglu import SwiGLU


class MoEFeedForward(nn.Module):
    """Mixture of Experts FFN with optional shared experts.

    Args:
        d_model: Input and output feature dimension.
        hidden_dim: Expert hidden dimension. Defaults to SwiGLU
            default if None.
        num_experts: Total number of routed experts.
        top_k: Experts activated per token.
        num_shared_experts: Always-active experts (0 = none).
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        num_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 0,
    ) -> None:
        """Initialize MoE FFN.

        Args:
            d_model: Feature dimension.
            hidden_dim: Expert hidden dim (None = SwiGLU default).
            num_experts: Number of routed experts.
            top_k: Experts per token.
            num_shared_experts: Always-active shared experts.
        """
        super().__init__()
        self.router = TopKRouter(num_experts, d_model, top_k)
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, hidden_dim) for _ in range(num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLU(d_model, hidden_dim) for _ in range(num_shared_experts)]
        )
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        """Apply MoE FFN.

        Routes each token to top-k experts, computes their
        outputs, and combines with gate weights. Stores
        aux_loss as an attribute.

        Args:
            x: Input ``[batch, seq_len, d_model]``.

        Returns:
            Output with same shape as input.
        """
        b, s, d = x.shape
        x_flat = x.reshape(b * s, d)

        gate_values, expert_indices, self.aux_loss = self.router(x)

        # Compute expert outputs for selected tokens
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Find which tokens selected this expert
            mask = (expert_indices == i).any(dim=-1)  # [N]
            if not mask.any():
                continue

            token_indices = mask.nonzero(as_tuple=True)[0]
            expert_input = x_flat[token_indices]
            expert_output = expert(expert_input)

            # Get gate weight for this expert per token
            # expert_indices[token_indices] has shape [n, top_k]
            # Find which position in top_k this expert is at
            expert_mask = expert_indices[token_indices] == i  # [n, top_k]
            weights = (gate_values[token_indices] * expert_mask).sum(
                dim=-1, keepdim=True
            )

            output[token_indices] += weights * expert_output

        # Add shared expert contributions
        for shared_expert in self.shared_experts:
            output = output + shared_expert(x_flat)

        return output.reshape(b, s, d)
