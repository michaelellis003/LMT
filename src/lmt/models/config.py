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
"""This module defines the configuration for the model architecture."""

from dataclasses import dataclass

from torch.nn import Module


@dataclass
class ModelConfig:
    """Configuration for the model architecture.

    Attributes:
        context_length (int): Max sequence length (context window).
        vocab_size (int): Vocabulary size, e.g., GPT-2 vocab size.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        embed_dim (int): Embedding dimension.
        dropout (float): Probability of a weight to be zeroed. Regularization
            technique to prevent overfitting. Must be a float between 0.0
            and 1.0.
        qkv_bias (bool, optional): If True, enables bias in the query, key,
                and value projections within the attention mechanism of each
                transformer block. Defaults to False.
        ff_network (nn.Module | None, optional): A custom feed-forward
            network to be used within each transformer block. If None, a
            default feed-forward network will be used. Defaults to None.
    """

    context_length: int = 4
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    dropout: float = 0.1
    qkv_bias: bool = False
    ff_network: Module | None = None
    num_kv_heads: int | None = None  # None = num_heads (standard MHA)
    ffn_hidden_dim: int | None = None  # None = use default for FFN type
    window_size: int | None = None  # None = full attention
    tie_weights: bool = False  # Tie input/output embedding weights
    qk_norm: bool = False  # Apply RMSNorm to Q and K before dot product
    value_residual_mix: float = 0.0  # Blend first layer's V into later layers
    head_dim: int | None = None  # None = embed_dim // num_heads


class ModelConfigPresets:
    """Predefined model configurations for common use cases.

    This class provides a set of static methods to create pre-configured
    `ModelConfig` objects for popular model architectures and sizes,
    as well as a utility method to build a configuration from command-line
    arguments.
    """

    @staticmethod
    def gpt2_124m(
        context_length: int = 1024, dropout: float = 0.1
    ) -> ModelConfig:
        """Creates a configuration for the GPT-2 124M model.

        Args:
            context_length (int): The maximum sequence length for the model.
                Defaults to 1024.
            dropout (float): The dropout rate to apply. Defaults to 0.1.

        Returns:
            ModelConfig: A configured `ModelConfig` object for the GPT-2 124M
                model.
        """
        return ModelConfig(
            context_length=context_length,
            vocab_size=50257,
            num_layers=12,
            num_heads=12,
            embed_dim=768,
            dropout=dropout,
            qkv_bias=True,
            ff_network=None,
        )

    @staticmethod
    def gpt2_355m(
        context_length: int = 1024, dropout: float = 0.1
    ) -> ModelConfig:
        """Creates a configuration for the GPT-2 355M model.

        Args:
            context_length (int): The maximum sequence length for the model.
                Defaults to 1024.
            dropout (float): The dropout rate to apply. Defaults to 0.1.

        Returns:
            ModelConfig: A configured `ModelConfig` object for the GPT-2 355M
            model.
        """
        return ModelConfig(
            context_length=context_length,
            vocab_size=50257,
            num_layers=24,
            num_heads=16,
            embed_dim=1024,
            dropout=dropout,
            qkv_bias=True,
            ff_network=None,
        )

    @staticmethod
    def small_gpt(
        context_length: int = 256, dropout: float = 0.1
    ) -> ModelConfig:
        """Creates a small GPT model configuration for quick experimentation.

        Args:
            context_length (int): The maximum sequence length for the model.
                Defaults to 256.
            dropout (float): The dropout rate to apply. Defaults to 0.1.

        Returns:
            ModelConfig: A configured `ModelConfig` object for a small GPT
            model.
        """
        return ModelConfig(
            context_length=context_length,
            vocab_size=50257,
            num_layers=6,
            num_heads=6,
            embed_dim=384,
            dropout=dropout,
            qkv_bias=False,
            ff_network=None,
        )

    @staticmethod
    def from_args(args) -> ModelConfig:
        """Creates a model config from command line arguments.

        This method expects the `args` object to have attributes corresponding
        to the `ModelConfig` parameters.

        Args:
            args (object): An object, such as an `argparse.Namespace`,
                containing the necessary configuration attributes.

        Returns:
            ModelConfig: A configured `ModelConfig` object built from the
            provided arguments.
        """
        return ModelConfig(
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
            qkv_bias=getattr(args, 'qkv_bias', False),
            ff_network=None,
        )
