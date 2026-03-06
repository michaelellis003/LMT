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
"""Mamba language model.

A decoder-only language model that replaces attention with Selective
State Space Models (SSMs). Unlike transformers, Mamba processes
sequences in O(n) time with constant-size hidden state, eliminating
the need for a KV cache at inference time.

Architecture:
    Token Embedding -> [MambaBlock x num_layers] -> RMSNorm -> LM Head

Each MambaBlock contains:
    RMSNorm -> in_proj -> Conv1d + SiLU -> SSM -> Gating -> out_proj

Reference: Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling
with Selective State Spaces." https://arxiv.org/abs/2312.00752
"""

import math

import torch
import torch.nn as nn

from lmt.layers.normalization import RMSNorm
from lmt.layers.ssm import MambaBlock
from lmt.models.config import ModelConfig


class Mamba(nn.Module):
    """Mamba language model with Selective SSM blocks.

    Args:
        config: Model configuration (uses embed_dim, vocab_size,
            num_layers, dropout).
        d_state: SSM state dimension per channel. Default 16.
        expand: Expansion factor for inner dimension. Default 2.
        d_conv: Causal conv1d kernel size. Default 4.
    """

    def __init__(
        self,
        config: ModelConfig,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
    ) -> None:
        """Initialize Mamba model.

        Args:
            config: Model configuration.
            d_state: SSM hidden state dimension.
            expand: Expansion factor (d_inner = expand * embed_dim).
            d_conv: Kernel size for causal depthwise Conv1d.
        """
        super().__init__()
        self.config = config

        self.tok_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=config.embed_dim,
                    d_state=d_state,
                    expand=expand,
                    d_conv=d_conv,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.norm_f = RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False
        )

        self._init_weights(config.num_layers)

    def _init_weights(self, num_layers: int) -> None:
        """Initialize weights with scaled residual init.

        Args:
            num_layers: Number of layers for scaling factor.
        """
        scale = 1.0 / math.sqrt(2 * num_layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        # Scale output projections on residual path
        for block in self.blocks:
            mamba_block: MambaBlock = block  # type: ignore[assignment]
            nn.init.normal_(mamba_block.out_proj.weight, std=0.02 * scale)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass: token indices to logits.

        Args:
            in_idx: Input token indices ``[batch, seq_len]``.

        Returns:
            Logits tensor ``[batch, seq_len, vocab_size]``.
        """
        x = self.tok_embed(in_idx)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        return self.lm_head(x)
