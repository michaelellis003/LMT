"""Base language model scaffold.

Provides the shared structure for all decoder-only language models:
token embedding, a stack of transformer blocks, final normalization,
and an output projection (LM head).

Named models (LLaMA, Mixtral, etc.) can either subclass BaseModel
or be constructed directly by passing a ``BlockConfig`` that selects
the desired attention, FFN, and normalization components.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from lmt.layers.blocks.configurable_block import BlockConfig, ConfigurableBlock
from lmt.layers.ffn.moe import MoEFeedForward
from lmt.layers.normalization import NORM_REGISTRY
from lmt.models.config import ModelConfig


class BaseModel(nn.Module):
    """Decoder-only language model assembled from composable blocks.

    The default configuration (GQA + SwiGLU + RMSNorm) produces a
    LLaMA-like model. Pass a different ``BlockConfig`` to get
    Mixtral (MoE FFN), GPT (MHA + default FFN), etc.

    Args:
        config: Model-level configuration.
        block_config: Per-block component selection. Defaults to
            GQA + SwiGLU + RMSNorm (LLaMA-like).
    """

    def __init__(
        self,
        config: ModelConfig,
        block_config: BlockConfig | None = None,
        learned_pos_embed: bool = False,
    ) -> None:
        """Initialize BaseModel.

        Args:
            config: Model configuration.
            block_config: Block component selection. If None, uses
                defaults (GQA + SwiGLU + RMSNorm).
            learned_pos_embed: If True, adds a learned positional
                embedding (GPT-style). If False, positional info
                comes from RoPE or other mechanisms in the attention
                layer.
        """
        super().__init__()
        if block_config is None:
            block_config = BlockConfig()

        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.embed_dim)

        self.pos_embed: nn.Embedding | None = None
        if learned_pos_embed:
            self.pos_embed = nn.Embedding(
                config.context_length, config.embed_dim
            )

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                ConfigurableBlock(config, block_config)
                for _ in range(config.num_layers)
            ]
        )

        norm_cls = NORM_REGISTRY[block_config.norm]
        self.final_norm = norm_cls(config.embed_dim)
        self.out_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False
        )

        # Tie input/output embedding weights
        if config.tie_weights:
            self.out_head.weight = self.tok_embed.weight

        # Track whether any block has MoE for aux_loss collection
        self._has_moe = block_config.ffn == 'moe'
        self.aux_loss = torch.tensor(0.0)

        # Gradient checkpointing: trade compute for memory
        self.gradient_checkpointing = False

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with scaled residual init.

        All linear layers get N(0, 0.02). Output projections on the
        residual path are scaled by 1/sqrt(2*num_layers) to prevent
        the residual stream variance from growing with depth.
        """
        num_layers = len(self.blocks)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scaled init for residual projections
        std = 0.02 / math.sqrt(2 * num_layers)
        for block in self.blocks:
            cfg_block: ConfigurableBlock = block  # type: ignore[assignment]
            attn = cfg_block.attn
            out_proj = getattr(attn, 'out_proj', None)
            if out_proj is not None and isinstance(out_proj, nn.Linear):
                nn.init.normal_(out_proj.weight, mean=0.0, std=std)

    def forward_hidden(self, in_idx: Tensor) -> Tensor:
        """Forward pass returning hidden states before the LM head.

        Useful for Multi-Token Prediction and other heads that
        need to process the backbone's representations.

        Args:
            in_idx: Token indices ``[batch, seq_len]``.

        Returns:
            Hidden states ``[batch, seq_len, embed_dim]`` after
            final normalization.
        """
        x = self.tok_embed(in_idx)

        if self.pos_embed is not None:
            seq_len = in_idx.shape[1]
            positions = torch.arange(seq_len, device=in_idx.device)
            x = x + self.pos_embed(positions)

        x = self.drop(x)

        if self._has_moe:
            total_aux = torch.tensor(0.0, device=x.device)
            for block in self.blocks:
                x = block(x)
                cfg_block: ConfigurableBlock = block  # type: ignore[assignment]
                if isinstance(cfg_block.ffn, MoEFeedForward):
                    total_aux = total_aux + cfg_block.ffn.aux_loss
            self.aux_loss = total_aux
        elif self.gradient_checkpointing and self.training:
            for block in self.blocks:
                x = grad_checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)

        return self.final_norm(x)

    def forward(self, in_idx: Tensor) -> Tensor:
        """Forward pass: token indices to logits.

        Args:
            in_idx: Token indices ``[batch, seq_len]``.

        Returns:
            Logits tensor ``[batch, seq_len, vocab_size]``.
        """
        return self.out_head(self.forward_hidden(in_idx))
