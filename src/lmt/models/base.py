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

    Supports per-layer configuration via a list of ``BlockConfig``
    objects, enabling interleaved patterns like local/global
    attention (Gemma 3) or mixed FFN types.

    Args:
        config: Model-level configuration.
        block_config: Per-block component selection. Can be a single
            config (applied to all layers), a list of configs (one
            per layer), or None for defaults.
    """

    def __init__(
        self,
        config: ModelConfig,
        block_config: BlockConfig | list[BlockConfig] | None = None,
        learned_pos_embed: bool = False,
    ) -> None:
        """Initialize BaseModel.

        Args:
            config: Model configuration.
            block_config: Block component selection. A single config
                is applied to all layers. A list provides per-layer
                configs (length must match num_layers). None uses
                defaults (GQA + SwiGLU + RMSNorm).
            learned_pos_embed: If True, adds a learned positional
                embedding (GPT-style). If False, positional info
                comes from RoPE or other mechanisms in the attention
                layer.
        """
        super().__init__()

        # Normalize block_config to a list of per-layer configs
        if block_config is None:
            block_configs = [BlockConfig() for _ in range(config.num_layers)]
        elif isinstance(block_config, list):
            if len(block_config) != config.num_layers:
                raise ValueError(
                    f'block_config list length ({len(block_config)}) '
                    f'must match num_layers ({config.num_layers})'
                )
            block_configs = block_config
        else:
            block_configs = [block_config for _ in range(config.num_layers)]

        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.embed_dim)

        self.pos_embed: nn.Embedding | None = None
        if learned_pos_embed:
            self.pos_embed = nn.Embedding(
                config.context_length, config.embed_dim
            )

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [ConfigurableBlock(config, bc) for bc in block_configs]
        )

        # Use the last layer's norm for final norm
        norm_cls = NORM_REGISTRY[block_configs[-1].norm]
        self.final_norm = norm_cls(config.embed_dim)
        self.out_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False
        )

        # Tie input/output embedding weights
        if config.tie_weights:
            self.out_head.weight = self.tok_embed.weight

        # Track whether any block has MoE for aux_loss collection
        self._has_moe = any(bc.ffn == 'moe' for bc in block_configs)
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

        # Scaled init for residual projections (both attention and FFN)
        std = 0.02 / math.sqrt(2 * num_layers)
        for block in self.blocks:
            cfg_block: ConfigurableBlock = block  # type: ignore[assignment]
            # Scale attention output projection
            attn = cfg_block.attn
            out_proj = getattr(attn, 'out_proj', None)
            if out_proj is not None and isinstance(out_proj, nn.Linear):
                nn.init.normal_(out_proj.weight, mean=0.0, std=std)
            # Scale FFN output projection (w2 for SwiGLU, each expert
            # w2 for MoE, or second linear for standard FFN)
            ffn = cfg_block.ffn
            if isinstance(ffn, MoEFeedForward):
                for expert in ffn.experts:
                    nn.init.normal_(expert.w2.weight, mean=0.0, std=std)  # type: ignore[union-attr]
                for expert in ffn.shared_experts:
                    nn.init.normal_(expert.w2.weight, mean=0.0, std=std)  # type: ignore[union-attr]
            else:
                ffn_out = getattr(ffn, 'w2', None)
                if ffn_out is not None and isinstance(ffn_out, nn.Linear):
                    nn.init.normal_(ffn_out.weight, mean=0.0, std=std)

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
                if self.gradient_checkpointing and self.training:
                    x = grad_checkpoint(block, x, use_reentrant=False)
                else:
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
