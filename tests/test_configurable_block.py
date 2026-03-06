"""Unit tests for Configurable Block Framework."""

import torch

from lmt.layers.blocks import BlockConfig, ConfigurableBlock
from lmt.models.config import ModelConfig


class TestBlockConfig:
    """Test suite for BlockConfig dataclass."""

    def test_default_values(self):
        """Test BlockConfig has sensible defaults."""
        bc = BlockConfig()
        assert bc.attention == 'gqa'
        assert bc.ffn == 'swiglu'
        assert bc.norm == 'rmsnorm'

    def test_custom_values(self):
        """Test BlockConfig accepts custom values."""
        bc = BlockConfig(
            attention='mha',
            ffn='moe',
            norm='rmsnorm',
            moe_num_experts=16,
            moe_top_k=4,
        )
        assert bc.attention == 'mha'
        assert bc.ffn == 'moe'
        assert bc.moe_num_experts == 16
        assert bc.moe_top_k == 4


class TestConfigurableBlock:
    """Test suite for ConfigurableBlock."""

    def _make_config(self, **kwargs):
        """Create a ModelConfig for tests."""
        defaults = dict(
            embed_dim=64,
            num_heads=4,
            context_length=32,
            vocab_size=1000,
            num_layers=1,
            dropout=0.0,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)

    def test_default_block(self):
        """Test block with default config (GQA + SwiGLU + RMSNorm)."""
        config = self._make_config()
        block = ConfigurableBlock(config, BlockConfig())
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_mha_block(self):
        """Test block with standard MHA."""
        config = self._make_config()
        bc = BlockConfig(attention='mha')
        block = ConfigurableBlock(config, bc)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_moe_block(self):
        """Test block with MoE FFN."""
        config = self._make_config()
        bc = BlockConfig(
            ffn='moe',
            moe_num_experts=4,
            moe_top_k=2,
        )
        block = ConfigurableBlock(config, bc)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_sliding_window_block(self):
        """Test block with sliding window attention."""
        config = self._make_config(window_size=8)
        bc = BlockConfig(attention='sliding_window')
        block = ConfigurableBlock(config, bc)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients propagate through configurable block."""
        config = self._make_config()
        block = ConfigurableBlock(config, BlockConfig())
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        for name, p in block.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'No gradient for {name}'

    def test_prenorm_structure(self):
        """Test block uses pre-norm architecture."""
        config = self._make_config()
        block = ConfigurableBlock(config, BlockConfig())
        # Should have two norm layers
        assert hasattr(block, 'attn_norm')
        assert hasattr(block, 'ffn_norm')

    def test_moe_aux_loss(self):
        """Test MoE block stores aux_loss."""
        config = self._make_config()
        bc = BlockConfig(
            ffn='moe',
            moe_num_experts=4,
            moe_top_k=2,
        )
        block = ConfigurableBlock(config, bc)
        x = torch.randn(2, 8, 64)
        block(x)
        assert hasattr(block.ffn, 'aux_loss')

    def test_deterministic_eval(self):
        """Test eval mode produces identical outputs."""
        config = self._make_config()
        block = ConfigurableBlock(config, BlockConfig())
        block.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out1 = block(x)
            out2 = block(x)
        torch.testing.assert_close(out1, out2)
