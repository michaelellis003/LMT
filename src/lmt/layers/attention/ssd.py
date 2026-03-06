r"""Structured State Space Duality (SSD) -- linear attention with decay.

Implements the SSD layer from Dao & Gu (2024, "Transformers are SSMs"),
which demonstrates that SSMs with scalar decay are mathematically
equivalent to a form of attention.

The key insight: given a selective SSM with scalar A_t:

.. math::

    h_t = a_t \cdot h_{t-1} + B_t x_t, \quad y_t = C_t^\top h_t

the output can be computed in two equivalent ways:

**Recurrent (linear) form** -- O(T) sequential:

.. math::

    h_t = a_t h_{t-1} + k_t v_t^\top, \quad o_t = h_t q_t

**Quadratic (attention-like) form** -- O(T^2) parallel:

.. math::

    Y = (L \odot Q K^\top) V

where L is the lower-triangular decay mask:

.. math::

    L_{ij} = \prod_{s=j+1}^{i} a_s \; (i \geq j), \;
    L_{ij} = 0 \; (i < j)

Both forms produce **identical** outputs. This is the Structured
State Space Duality -- choosing between them is like choosing
different contraction orders for the same tensor network.

Unlike GatedDeltaNet, SSD does NOT use the delta rule. It's
simpler (no state-dependent correction) but demonstrates the
duality more cleanly. GatedDeltaNet adds the delta correction
on top of this foundation.

Reference: Dao & Gu (2024). "Transformers are SSMs: Generalized
Models and Efficient Algorithms Through Structured State Space
Duality." https://arxiv.org/abs/2405.21060
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor

from lmt.models.config import ModelConfig


class SSDAttention(nn.Module):
    r"""SSD attention layer with both recurrent and quadratic forms.

    This layer implements vanilla linear attention with scalar decay,
    exposing both computation modes for verification of the duality.
    The default ``forward`` uses the quadratic form (more efficient
    on GPU for moderate sequence lengths).

    Args:
        model_config: Model configuration with embed_dim, num_heads.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize SSD attention.

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

        # Q, K, V projections (maps to C, B, X in SSM notation)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Scalar decay: a_t = exp(-exp(A_log) * softplus(W_a(x) + bias))
        self.alpha_proj = nn.Linear(embed_dim, num_heads, bias=False)
        self.A_log = nn.Parameter(
            torch.linspace(math.log(1), math.log(16), num_heads)
        )
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))

        # Output gate and projection
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Per-head output normalization
        self.out_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def _compute_alpha(self, x: Tensor) -> Tensor:
        """Compute scalar decay alpha in (0, 1].

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Alpha ``[batch, seq_len, num_heads]``.
        """
        dt = f.softplus(self.alpha_proj(x) + self.dt_bias)
        a_decay = self.A_log.exp()
        return torch.exp(-a_decay * dt)

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

        # L2 normalize for stability (no softmax to bound outputs)
        q = f.normalize(q, dim=-1)
        k = f.normalize(k, dim=-1)

        # Transpose to [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def _build_decay_mask(self, alpha: Tensor) -> Tensor:
        r"""Build the lower-triangular decay mask L.

        .. math::

            L_{ij} = \prod_{s=j+1}^{i} a_s \quad (i \geq j)

        Args:
            alpha: Decay values ``[batch, seq_len, num_heads]``.

        Returns:
            L matrix ``[batch, num_heads, seq_len, seq_len]``.
        """
        b, seq_len, h = alpha.shape

        # Compute cumulative log-decay for efficient product
        # log(alpha): [b, seq, h] -> cumsum gives log(prod(alpha[0:t]))
        log_alpha = torch.log(alpha + 1e-8)  # [b, seq, h]
        cum_log = torch.cumsum(log_alpha, dim=1)  # [b, seq, h]

        # L[i,j] = exp(cum_log[i] - cum_log[j]) for i >= j
        # Transpose to [b, h, seq]
        cum_log = cum_log.transpose(1, 2)  # [b, h, seq]

        # Broadcast: [b, h, seq, 1] - [b, h, 1, seq] = [b, h, seq, seq]
        log_l = cum_log.unsqueeze(-1) - cum_log.unsqueeze(-2)

        # Apply causal mask BEFORE exp to avoid inf * 0 = nan
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=alpha.device)
        )
        # Set upper-triangular entries to -inf so exp gives 0
        log_l = log_l.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf')
        )

        return torch.exp(log_l)

    def forward_recurrent(self, x: Tensor) -> Tensor:
        """Compute output using the recurrent (linear) form.

        O(T) sequential computation via SSM recurrence.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output ``[batch, seq_len, embed_dim]``.
        """
        b, seq_len, _ = x.shape

        q, k, v = self._project_qkv(x)
        alpha = self._compute_alpha(x)
        alpha = alpha.transpose(1, 2)  # [b, h, seq]

        # Sequential recurrence: h_t = a_t * h_{t-1} + v_t @ k_t^T
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
            a_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)  # [b, h, 1, 1]

            # Decay and accumulate
            state = a_t * state + torch.einsum('bhi,bhj->bhij', v_t, k_t)

            # Output: o_t = h_t @ q_t
            o_t = torch.einsum('bhij,bhj->bhi', state, q_t)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2)  # [b, h, seq, d]
        return self._apply_output(out, x, seq_len)

    def forward_quadratic(self, x: Tensor) -> Tensor:
        r"""Compute output using the quadratic (attention-like) form.

        .. math::

            Y = (L \odot Q K^\top) V

        O(T^2) parallel computation via materialized attention matrix.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output ``[batch, seq_len, embed_dim]``.
        """
        b, seq_len, _ = x.shape

        q, k, v = self._project_qkv(x)
        alpha = self._compute_alpha(x)

        # Build decay mask L: [b, h, seq, seq]
        l_mat = self._build_decay_mask(alpha)

        # Attention scores: Q @ K^T
        scores = torch.einsum('bhid,bhjd->bhij', q, k)

        # Apply decay mask: L ⊙ (Q @ K^T)
        scores = l_mat * scores

        # Output: (L ⊙ Q K^T) @ V
        out = torch.einsum('bhij,bhjd->bhid', scores, v)

        return self._apply_output(out, x, seq_len)

    def _apply_output(self, out: Tensor, x: Tensor, seq_len: int) -> Tensor:
        """Apply output normalization, gating, and projection.

        Args:
            out: Raw output ``[batch, num_heads, seq_len, head_dim]``.
            x: Original input for gating ``[batch, seq_len, embed_dim]``.
            seq_len: Sequence length.

        Returns:
            Final output ``[batch, seq_len, embed_dim]``.
        """
        b = out.shape[0]
        out = self.out_norm(out)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, self.embed_dim)
        gate = f.silu(self.gate_proj(x))
        out = out * gate
        return self.out_proj(out)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using the quadratic form.

        For moderate sequence lengths, the quadratic form is faster
        due to GPU parallelism. For very long sequences, the recurrent
        form would be preferred.

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output ``[batch, seq_len, embed_dim]``.
        """
        return self.forward_quadratic(x)
