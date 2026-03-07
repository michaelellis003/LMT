r"""Gemma architecture.

Implements the decoder-only transformer from:
    Gemma Team, Google, 2024 -- "Gemma: Open Models Based on Gemini
    Research and Technology"
    Gemma Team, 2025 -- "Gemma 3 Technical Report"

Gemma 3 key features:
- **Interleaved local/global attention**: most layers use sliding
  window (local) attention for efficiency, every Nth layer uses
  full (global) attention for long-range dependencies
- **QK-Norm**: RMSNorm on Q and K per-head
- **Weight Tying**: shared input/output embedding weights
- **RMSNorm** (pre-norm)
- **RoPE** (rotary positional embeddings)
- **SwiGLU** feed-forward
- **GQA** (grouped query attention)
- **Embedding scaling** by ``sqrt(embed_dim)``

Implemented as a BaseModel wrapper using per-layer block configs
to achieve the interleaved attention pattern.
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

    Uses interleaved local/global attention, QK-Norm, weight
    tying, and embedding scaling by ``sqrt(embed_dim)``.

    Args:
        config: Model configuration.
        local_window: Sliding window size for local attention
            layers. Default: 512.
        global_every: Insert a global attention layer every N
            layers. Default: 4 (i.e., layers 3, 7, 11, ...).
    """

    def __init__(
        self,
        config: ModelConfig,
        local_window: int = 512,
        global_every: int = 4,
    ) -> None:
        """Initialize Gemma model.

        Args:
            config: Model configuration. ``qk_norm`` and
                ``tie_weights`` are forced True.
            local_window: Window size for local attention layers.
            global_every: Global attention every N layers. The
                last layer in each group of N is global.
        """
        config.qk_norm = True
        config.tie_weights = True

        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(
            d_model=head_dim,
            max_seq_len=config.context_length,
        )

        # Build per-layer configs: local with window, global without
        block_configs = []
        for i in range(config.num_layers):
            is_global = (i + 1) % global_every == 0
            block_configs.append(
                BlockConfig(
                    attention='gqa',
                    ffn='swiglu',
                    norm='rmsnorm',
                    rope=rope,
                    window_size=None if is_global else local_window,
                )
            )

        super().__init__(config, block_config=block_configs)

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
