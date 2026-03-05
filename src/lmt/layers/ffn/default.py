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

"""Default feedforward network used in Transformer blocks."""

import torch
import torch.nn as nn


class DefaultFeedForward(nn.Module):
    r"""Default feedforward network used in Transformer block.

    Implements a two-layer MLP with GELU activation:

    .. math::

        \text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2

    This is the standard feedforward network from the original Transformer
    and GPT-2 architectures.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        """Initialize the DefaultFeedForward network.

        Args:
            embed_dim: The dimensionality of the input and output
                embedding vectors for each token position.
            hidden_dim: The dimensionality of the intermediate (hidden)
                layer within the feedforward network. This is commonly
                ``4 * embed_dim`` in many Transformer architectures.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feedforward network to the input tensor.

        Args:
            x: Input tensor of shape
                ``(batch_size, sequence_length, embed_dim)``.

        Returns:
            Output tensor of shape
                ``(batch_size, sequence_length, embed_dim)``.
        """
        return self.layers(x)
