"""Tests for interleaved local/global attention patterns."""

import torch

from lmt.layers.blocks.configurable_block import BlockConfig
from lmt.layers.positional import RoPE
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig


def _small_config(**kwargs):
    """Create a small test config."""
    defaults = dict(
        embed_dim=64,
        num_heads=4,
        num_kv_heads=4,
        context_length=32,
        vocab_size=256,
        num_layers=4,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


class TestInterleavedAttention:
    """Test per-layer block configs for interleaved patterns."""

    def test_per_layer_block_configs(self):
        """BaseModel accepts a list of BlockConfigs (one per layer)."""
        config = _small_config(num_layers=4)
        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(d_model=head_dim, max_seq_len=config.context_length)

        # 3 local (sliding window) + 1 global
        local = BlockConfig(
            attention='gqa',
            ffn='swiglu',
            norm='rmsnorm',
            rope=rope,
            window_size=8,
        )
        global_ = BlockConfig(
            attention='gqa',
            ffn='swiglu',
            norm='rmsnorm',
            rope=rope,
        )
        configs = [local, local, local, global_]

        model = BaseModel(config, block_config=configs)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_per_layer_length_must_match(self):
        """List length must match num_layers."""
        import pytest

        config = _small_config(num_layers=4)
        configs = [BlockConfig() for _ in range(3)]  # wrong length

        with pytest.raises(ValueError, match='block_config list'):
            BaseModel(config, block_config=configs)

    def test_gradient_flow_interleaved(self):
        """Gradients flow through interleaved architecture."""
        config = _small_config(num_layers=4)
        head_dim = config.embed_dim // config.num_heads
        rope = RoPE(d_model=head_dim, max_seq_len=config.context_length)

        local = BlockConfig(
            attention='gqa',
            ffn='swiglu',
            norm='rmsnorm',
            rope=rope,
            window_size=8,
        )
        global_ = BlockConfig(
            attention='gqa',
            ffn='swiglu',
            norm='rmsnorm',
            rope=rope,
        )
        configs = [local, local, local, global_]

        model = BaseModel(config, block_config=configs)
        x = torch.randint(0, 256, (1, 16))
        logits = model(x)
        logits.sum().backward()
        assert model.tok_embed.weight.grad is not None

    def test_single_block_config_still_works(self):
        """Passing a single BlockConfig applies to all layers."""
        config = _small_config(num_layers=4)
        bc = BlockConfig()
        model = BaseModel(config, block_config=bc)
        x = torch.randint(0, 256, (1, 8))
        logits = model(x)
        assert logits.shape == (1, 8, 256)

    def test_mixed_ffn_types(self):
        """Can mix different FFN types across layers."""
        config = _small_config(num_layers=3)

        configs = [
            BlockConfig(ffn='swiglu'),
            BlockConfig(ffn='default'),
            BlockConfig(ffn='swiglu'),
        ]

        model = BaseModel(config, block_config=configs)
        x = torch.randint(0, 256, (1, 8))
        logits = model(x)
        assert logits.shape == (1, 8, 256)
