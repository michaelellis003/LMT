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
"""Mamba block combining Conv1d, Selective SSM, and gating.

A Mamba block replaces the attention + FFN pattern of transformers
with a single sequence mixer that combines:

1. Input projection + split into x and z (gate) branches
2. Causal depthwise Conv1d on x branch (local context)
3. Selective SSM on x branch (global sequence modeling)
4. Gating: y * SiLU(z)
5. Output projection

The block uses RMSNorm + residual connection following the
pre-norm pattern used by LLaMA and other modern architectures.

Reference: Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling
with Selective State Spaces." https://arxiv.org/abs/2312.00752
"""

import torch
import torch.nn as nn
from torch.nn import functional as f

from lmt.layers.normalization import RMSNorm
from lmt.layers.ssm.selective_ssm import SelectiveSSM


class MambaBlock(nn.Module):
    """A single Mamba block with SSM mixer and residual connection.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension. Default 16.
        expand: Expansion factor for inner dimension. Default 2.
        d_conv: Causal conv1d kernel size. Default 4.
        dt_rank: Rank of delta projection. Default ceil(d_inner/16).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dt_rank: int | None = None,
    ) -> None:
        """Initialize MambaBlock.

        Args:
            d_model: Model embedding dimension.
            d_state: SSM hidden state dimension.
            expand: Expansion factor (d_inner = expand * d_model).
            d_conv: Kernel size for causal depthwise Conv1d.
            dt_rank: Rank of the delta projection.
        """
        super().__init__()
        self.d_model = d_model
        d_inner = expand * d_model

        self.norm = RMSNorm(d_model)

        # Project to 2*d_inner: split into x and z (gate) branches
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise conv1d on x branch
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal padding
            groups=d_inner,  # depthwise
        )

        # Selective SSM
        self.ssm = SelectiveSSM(
            d_inner=d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
        )

        # Output projection back to d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.

        Returns:
            Output tensor ``[batch, seq_len, d_model]``.
        """
        residual = x
        x = self.norm(x)

        # Project and split into x_branch and gate
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_branch, z = xz.chunk(2, dim=-1)  # each [B, L, d_inner]

        # Causal conv1d on x branch
        # Conv1d expects [B, C, L], so transpose
        x_branch = x_branch.transpose(1, 2)  # [B, d_inner, L]
        x_branch = self.conv1d(x_branch)[:, :, : x.shape[1]]  # causal trim
        x_branch = x_branch.transpose(1, 2)  # [B, L, d_inner]
        x_branch = f.silu(x_branch)

        # Selective SSM
        y = self.ssm(x_branch)  # [B, L, d_inner]

        # Gating: y * SiLU(z)
        y = y * f.silu(z)

        # Output projection + residual
        return self.out_proj(y) + residual
