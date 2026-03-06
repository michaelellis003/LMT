# Copyright 2025 Michael Ellis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Selective State Space Model (S6) from Mamba.

The core SSM computes a linear recurrence with input-dependent parameters:

.. math::

    h_t = \bar{A} h_{t-1} + \bar{B} x_t
    y_t = C h_t + D x_t

where the discretization step converts continuous parameters to discrete:

.. math::

    \bar{A} = \exp(\Delta \cdot A)
    \bar{B} = \Delta \cdot B

The key innovation of Mamba is that B, C, and Delta are all
*input-dependent* (projected from x), making the model "selective"
-- it learns what to remember and what to forget based on content.

Reference: Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling
with Selective State Spaces." https://arxiv.org/abs/2312.00752
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as f


class SelectiveSSM(nn.Module):
    r"""Selective State Space Model (S6).

    Implements the selective scan from Mamba with input-dependent
    B, C, and Delta parameters. Uses a sequential scan for clarity;
    a parallel scan (Blelloch algorithm) would be faster on GPU but
    harder to understand.

    Args:
        d_inner: Input/output dimension (expanded model dimension).
        d_state: SSM state dimension per channel. Default 16.
        dt_rank: Rank of the Delta projection. Default ``ceil(d_inner/16)``.
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int | None = None,
    ) -> None:
        """Initialize SelectiveSSM.

        Args:
            d_inner: Input/output dimension.
            d_state: SSM hidden state dimension per channel.
            dt_rank: Rank of the delta projection. Defaults to
                ceil(d_inner / 16).
        """
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_inner / 16)

        # A is stored in log-space for numerical stability.
        # Initialized as log(arange(1, d_state+1)), so
        # actual A = -exp(A_log) is always negative (stable).
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(d_inner, -1)
            .clone()
        )

        # D is a skip connection scalar per channel
        self.D = nn.Parameter(torch.ones(d_inner))

        # Input-dependent projections: x -> (dt, B, C)
        self.x_proj = nn.Linear(
            d_inner, self.dt_rank + 2 * d_state, bias=False
        )

        # Delta projection (low-rank -> d_inner)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run selective scan over input sequence.

        Args:
            x: Input tensor of shape ``[batch, seq_len, d_inner]``.

        Returns:
            Output tensor of shape ``[batch, seq_len, d_inner]``.
        """
        batch, seq_len, _ = x.shape

        # Project input to get dt, b_mat, c_mat (all input-dependent)
        x_proj = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        dt, b_mat, c_mat = torch.split(
            x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Delta: low-rank projection + softplus (ensures positive)
        dt = f.softplus(self.dt_proj(dt))  # [B, L, d_inner]

        # A is always negative (stable dynamics)
        a_mat = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # Sequential scan (educational clarity over speed)
        y = self._sequential_scan(x, dt, a_mat, b_mat, c_mat)

        # Skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def _sequential_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        a_mat: torch.Tensor,
        b_mat: torch.Tensor,
        c_mat: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential scan for the SSM recurrence.

        Args:
            x: Input ``[batch, seq_len, d_inner]``.
            dt: Time steps ``[batch, seq_len, d_inner]``.
            a_mat: State matrix ``[d_inner, d_state]``.
            b_mat: Input projection ``[batch, seq_len, d_state]``.
            c_mat: Output projection ``[batch, seq_len, d_state]``.

        Returns:
            Output ``[batch, seq_len, d_inner]``.
        """
        batch, seq_len, d_inner = x.shape

        # Discretize: a_bar = exp(dt * A), b_bar = dt * B
        # dt: [B, L, d_inner] -> [B, L, d_inner, 1]
        # a_mat: [d_inner, d_state] -> [1, 1, d_inner, d_state]
        dt_exp = dt.unsqueeze(-1)  # [B, L, d_inner, 1]
        a_exp = a_mat.unsqueeze(0).unsqueeze(0)  # [1, 1, d_inner, N]
        a_bar = torch.exp(dt_exp * a_exp)  # [B, L, d_inner, N]

        # b_mat: [B, L, N] -> [B, L, 1, N] * dt: [B, L, d_inner, 1]
        b_bar = dt_exp * b_mat.unsqueeze(2)  # [B, L, d_inner, N]

        # Initialize hidden state
        h = torch.zeros(
            batch,
            d_inner,
            self.d_state,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []
        for t in range(seq_len):
            # h_t = a_bar * h_{t-1} + b_bar * x_t
            h = a_bar[:, t] * h + b_bar[:, t] * x[:, t].unsqueeze(-1)
            # y_t = C * h_t (sum over state dimension)
            y_t = (h * c_mat[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # [B, L, d_inner]
