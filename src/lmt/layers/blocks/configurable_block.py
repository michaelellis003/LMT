"""Configurable transformer block that composes layers from registries.

Instead of hardcoding specific attention/FFN/norm combinations (like
LlamaBlock or MixtralBlock), this block looks up components by string
key in the layer registries, making it easy to mix and match.
"""

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from lmt.layers.attention import (
    ATTENTION_REGISTRY,
    GroupedQueryAttention,
    MultiHeadAttention,
    SlidingWindowAttention,
)
from lmt.layers.ffn import FFN_REGISTRY
from lmt.layers.ffn.moe import MoEFeedForward
from lmt.layers.normalization import NORM_REGISTRY
from lmt.models.config import ModelConfig


@dataclass
class BlockConfig:
    """Configuration for a single transformer block.

    Args:
        attention: Key into ATTENTION_REGISTRY.
        ffn: Key into FFN_REGISTRY.
        norm: Key into NORM_REGISTRY.
        moe_num_experts: Number of MoE experts (only used if ffn='moe').
        moe_top_k: Experts per token (only used if ffn='moe').
        moe_shared_experts: Always-active experts (only if ffn='moe').
    """

    attention: str = 'gqa'
    ffn: str = 'swiglu'
    norm: str = 'rmsnorm'
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_shared_experts: int = 0


class ConfigurableBlock(nn.Module):
    """A transformer block assembled from registry components.

    Pre-norm: Norm -> Attn -> Residual -> Norm -> FFN -> Residual.

    Args:
        model_config: Model-level configuration.
        block_config: Block-level configuration selecting components.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        block_config: BlockConfig,
    ) -> None:
        """Initialize configurable block.

        Args:
            model_config: Model configuration.
            block_config: Block component selection.
        """
        super().__init__()

        # Build normalization layers
        norm_cls = NORM_REGISTRY[block_config.norm]
        self.attn_norm = norm_cls(model_config.embed_dim)
        self.ffn_norm = norm_cls(model_config.embed_dim)

        # Build attention
        self.attn = self._build_attention(model_config, block_config)

        # Build FFN
        self.ffn = self._build_ffn(model_config, block_config)

    @staticmethod
    def _build_attention(
        model_config: ModelConfig,
        block_config: BlockConfig,
    ) -> nn.Module:
        """Build the attention module from config.

        Args:
            model_config: Model configuration.
            block_config: Block configuration.

        Returns:
            Attention module.
        """
        key = block_config.attention

        if key == 'mha':
            return MultiHeadAttention(model_config)
        elif key == 'gqa':
            return GroupedQueryAttention(model_config)
        elif key == 'sliding_window':
            return SlidingWindowAttention(model_config)
        else:
            attn_cls = ATTENTION_REGISTRY[key]
            return attn_cls(model_config)

    @staticmethod
    def _build_ffn(
        model_config: ModelConfig,
        block_config: BlockConfig,
    ) -> nn.Module:
        """Build the FFN module from config.

        Args:
            model_config: Model configuration.
            block_config: Block configuration.

        Returns:
            FFN module.
        """
        key = block_config.ffn

        if key == 'moe':
            return MoEFeedForward(
                d_model=model_config.embed_dim,
                num_experts=block_config.moe_num_experts,
                top_k=block_config.moe_top_k,
                num_shared_experts=block_config.moe_shared_experts,
            )
        elif key == 'swiglu':
            ffn_cls = FFN_REGISTRY['swiglu']
            return ffn_cls(d_model=model_config.embed_dim)
        else:
            ffn_cls = FFN_REGISTRY[key]
            return ffn_cls(model_config.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the configurable block (pre-norm architecture).

        Args:
            x: Input ``[batch, seq_len, embed_dim]``.

        Returns:
            Output with same shape.
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
