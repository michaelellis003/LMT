r"""Gemma architecture.

Implements the decoder-only transformer from:
    Gemma Team, Google, 2024 -- "Gemma: Open Models Based on Gemini
    Research and Technology"
    Gemma Team, 2025 -- "Gemma 3 Technical Report"

Gemma 3 shares the modern LLaMA-family blueprint with:
- **QK-Norm**: RMSNorm on Q and K per-head (added in Gemma 3)
- **Weight Tying**: shared input/output embedding weights
- **RMSNorm** (pre-norm)
- **RoPE** (rotary positional embeddings)
- **SwiGLU** feed-forward (called GeGLU in some Gemma docs,
  but the gating mechanism is the same)
- **GQA** (grouped query attention)

Gemma also scales the embedding output by ``sqrt(embed_dim)``
before feeding it to the blocks. This is a normalizing factor
that stabilizes training at different model scales.

Implemented as a thin wrapper around ``BaseModel`` that forces
``qk_norm=True`` and ``tie_weights=True``, and applies the
embedding scaling factor.
"""

import math

import torch
from torch import Tensor

from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.layers.positional import RoPE
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig


class Gemma(BaseModel):
    """Gemma decoder-only transformer model.

    LLaMA-family architecture with QK-Norm, weight tying, and
    embedding scaling by ``sqrt(embed_dim)``.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize Gemma model.

        Args:
            config: Model configuration. ``qk_norm`` and
                ``tie_weights`` are forced True regardless of
                the config values.
        """
        # Force Gemma-specific features
        config.qk_norm = True
        config.tie_weights = True

        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(
            d_model=head_dim,
            max_seq_len=config.context_length,
        )

        block_config = BlockConfig(
            attention='gqa',
            ffn='swiglu',
            norm='rmsnorm',
            rope=rope,
        )

        super().__init__(config, block_config=block_config)

        # Gemma scales embeddings by sqrt(embed_dim)
        self._embed_scale = math.sqrt(config.embed_dim)

    def forward_hidden(self, in_idx: Tensor) -> Tensor:
        """Forward pass with Gemma's embedding scaling.

        Args:
            in_idx: Token indices ``[batch, seq_len]``.

        Returns:
            Hidden states ``[batch, seq_len, embed_dim]``.
        """
        x = self.tok_embed(in_idx) * self._embed_scale

        if self.pos_embed is not None:
            seq_len = in_idx.shape[1]
            positions = torch.arange(seq_len, device=in_idx.device)
            x = x + self.pos_embed(positions)

        x = self.drop(x)

        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import (
                checkpoint as grad_checkpoint,
            )

            for block in self.blocks:
                x = grad_checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)

        return self.final_norm(x)
