"""Configurable transformer block that composes layers from registries.

Instead of hardcoding specific attention/FFN/norm combinations (like
LlamaBlock or MixtralBlock), this block looks up components by string
key in the layer registries, making it easy to mix and match.

Supports optional RoPE integration and sliding window masking via
``BlockConfig``, so named models like LLaMA and Mixtral can be
expressed as configuration rather than custom code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

TYPE_CHECKING = False
if TYPE_CHECKING:
    from lmt.layers.positional import RoPE


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
        rope: Optional RoPE instance for attention layers that support it.
        window_size: Optional sliding window size for GQA.
    """

    attention: str = 'gqa'
    ffn: str = 'swiglu'
    norm: str = 'rmsnorm'
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_shared_experts: int = 0
    rope: RoPE | None = field(default=None, repr=False)
    window_size: int | None = None


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
            return GroupedQueryAttention(
                model_config,
                rope=block_config.rope,
                window_size=block_config.window_size,
            )
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
        elif key == 'default':
            ffn_cls = FFN_REGISTRY['default']
            return ffn_cls(
                embed_dim=model_config.embed_dim,
                hidden_dim=4 * model_config.embed_dim,
            )
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
