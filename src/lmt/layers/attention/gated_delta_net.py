r"""Gated Delta Networks -- linear attention with delta rule.

Implements the Gated DeltaNet mechanism from Yang et al. (2024),
which combines gated memory decay with the delta rule for
associative memory updates.

**Standard linear attention** accumulates key-value associations
additively:

.. math::

    S_t = S_{t-1} + v_t k_t^\top

This never forgets, so the memory fills up and becomes noisy.

**The delta rule** treats S as associative memory updated via
one-step gradient descent on :math:`\|S k_t - v_t\|^2`:

.. math::

    S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^\top

This "writes minus what's already there" -- a targeted correction
rather than blind accumulation.

**Gated DeltaNet** adds a decay gate :math:`\alpha_t \in (0, 1]`
for fast memory erasure:

.. math::

    S_t = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1} k_t) k_t^\top

The output is :math:`o_t = S_t q_t`, followed by normalization
and output gating with SiLU.

**Complexity:** O(n * d^2) per layer -- linear in sequence length,
quadratic in head dimension. This trades the O(n^2 * d) of softmax
attention for O(n * d^2), which wins when n >> d.

Reference: Yang et al. (2024). "Gated Delta Networks: Improving
Mamba2 with Delta Rule." https://arxiv.org/abs/2412.06464
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor

from lmt.models.config import ModelConfig


class GatedDeltaNet(nn.Module):
    r"""Gated Delta Network attention layer.

    Linear attention variant that maintains a fixed-size state
    matrix :math:`S \in \mathbb{R}^{d_h \times d_h}` per head,
    updated via the gated delta rule. No KV cache or causal mask
    needed -- causality is inherent in the recurrence.

    Args:
        model_config: Model configuration with embed_dim, num_heads.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize Gated DeltaNet.

        Args:
            model_config: Model configuration.
        """
        super().__init__()
        embed_dim = model_config.embed_dim
        num_heads = model_config.num_heads

        assert embed_dim % num_heads == 0, (
            'embed_dim must be divisible by num_heads'
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Decay gate: alpha = exp(-exp(A_log) * softplus(W_alpha(x) + bias))
        self.alpha_proj = nn.Linear(embed_dim, num_heads, bias=False)
        self.A_log = nn.Parameter(
            torch.linspace(math.log(1), math.log(16), num_heads)
        )
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))

        # Update gate: beta = sigmoid(W_beta(x))
        self.beta_proj = nn.Linear(embed_dim, num_heads, bias=False)

        # Output gate: o * silu(gate)
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Per-head output normalization
        self.out_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def _compute_alpha(self, x: Tensor) -> Tensor:
        """Compute decay gate alpha in (0, 1].

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Alpha ``[batch, seq_len, num_heads]``.
        """
        # dt = softplus(W_alpha(x) + dt_bias)
        dt = f.softplus(self.alpha_proj(x) + self.dt_bias)
        # alpha = exp(-exp(A_log) * dt), always in (0, 1]
        a_decay = self.A_log.exp()
        return torch.exp(-a_decay * dt)

    def _compute_beta(self, x: Tensor) -> Tensor:
        """Compute update gate beta in (0, 1).

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Beta ``[batch, seq_len, num_heads]``.
        """
        return torch.sigmoid(self.beta_proj(x))

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project input to Q, K, V with L2 normalization.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Tuple of (q, k, v) each
            ``[batch, num_heads, seq_len, head_dim]``.
        """
        b, seq_len, _ = x.shape

        q = self.q_proj(x).view(b, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, seq_len, self.num_heads, self.head_dim)

        # L2 normalize for training stability
        q = f.normalize(q, dim=-1) / math.sqrt(self.head_dim)
        k = f.normalize(k, dim=-1)

        # Transpose to [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply Gated DeltaNet attention.

        The recurrence (per head):

        .. math::

            S_t = \alpha_t S_{t-1}
                + \beta_t (v_t - \alpha_t S_{t-1} k_t) k_t^\top

            o_t = S_t q_t

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output ``[batch, seq_len, embed_dim]``.
        """
        b, seq_len, _ = x.shape

        q, k, v = self._project_qkv(x)

        # Gates: [batch, seq_len, num_heads]
        alpha = self._compute_alpha(x)
        beta = self._compute_beta(x)

        # Reshape gates to [batch, num_heads, seq_len, 1, 1] and
        # [batch, num_heads, seq_len, 1] for broadcasting
        alpha = alpha.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        beta = beta.transpose(1, 2).unsqueeze(-1)

        # Sequential recurrence over time steps
        # S: [batch, num_heads, head_dim, head_dim]
        state = torch.zeros(
            b,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []
        for t in range(seq_len):
            k_t = k[:, :, t]  # [b, h, d]
            v_t = v[:, :, t]  # [b, h, d]
            q_t = q[:, :, t]  # [b, h, d]
            a_t = alpha[:, :, t]  # [b, h, 1, 1]
            b_t = beta[:, :, t]  # [b, h, 1]

            # Decay old state
            state = a_t * state

            # Delta: what we want (v_t) minus what's already stored
            # S @ k_t: [b, h, d, d] @ [b, h, d, 1] -> [b, h, d]
            stored = torch.einsum('bhij,bhj->bhi', state, k_t)
            delta = b_t * (v_t - stored)

            # Update: S += delta @ k_t^T
            state = state + torch.einsum('bhi,bhj->bhij', delta, k_t)

            # Output: o_t = S @ q_t
            o_t = torch.einsum('bhij,bhj->bhi', state, q_t)
            outputs.append(o_t)

        # Stack: [b, h, seq, d]
        out = torch.stack(outputs, dim=2)

        # Per-head RMSNorm
        out = self.out_norm(out)

        # Reshape to [b, seq, embed_dim]
        out = out.transpose(1, 2).contiguous().view(b, seq_len, self.embed_dim)

        # Output gating: out * silu(gate)
        gate = f.silu(self.gate_proj(x))
        out = out * gate

        return self.out_proj(out)
