"""Unit tests for Mixture of Depths."""

import torch

from lmt.layers.blocks.configurable_block import BlockConfig, ConfigurableBlock
from lmt.layers.blocks.mixture_of_depths import DepthRouter, MoDBlock
from lmt.models.config import ModelConfig


class TestDepthRouter:
    """Test suite for the DepthRouter."""

    def _make_router(self, d_model=64, capacity=0.5):
        """Create a DepthRouter for testing."""
        return DepthRouter(d_model=d_model, capacity=capacity)

    def test_initialization(self):
        """Test router initializes correctly."""
        router = self._make_router(d_model=64, capacity=0.5)
        assert router.capacity == 0.5
        assert router.gate.in_features == 64
        assert router.gate.out_features == 1

    def test_output_shapes(self):
        """Test router returns correct shapes."""
        router = self._make_router(d_model=64, capacity=0.5)
        x = torch.randn(2, 16, 64)

        mask, router_weights, aux_loss = router(x)

        # mask: [batch, seq_len, 1] binary
        assert mask.shape == (2, 16, 1)
        # router_weights: [batch, seq_len, 1] continuous
        assert router_weights.shape == (2, 16, 1)
        # aux_loss: scalar
        assert aux_loss.shape == ()

    def test_capacity_fraction(self):
        """Test that capacity controls the fraction of tokens processed."""
        router = self._make_router(d_model=64, capacity=0.5)
        x = torch.randn(2, 16, 64)

        mask, _, _ = router(x)

        # With capacity=0.5, exactly 50% of tokens per batch should be selected
        expected_selected = int(16 * 0.5)
        for b in range(2):
            selected = mask[b].sum().item()
            assert selected == expected_selected

    def test_capacity_one(self):
        """With capacity=1.0, all tokens are processed."""
        router = self._make_router(d_model=64, capacity=1.0)
        x = torch.randn(2, 8, 64)

        mask, _, _ = router(x)

        assert mask.sum().item() == 2 * 8  # All tokens selected

    def test_gradient_flow(self):
        """Test gradients flow through the router."""
        router = self._make_router(d_model=64, capacity=0.5)
        x = torch.randn(2, 8, 64, requires_grad=True)

        mask, weights, aux_loss = router(x)
        # Use weights (continuous) for gradient flow
        loss = (weights * x).sum() + aux_loss
        loss.backward()

        assert x.grad is not None
        for name, param in router.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f'No grad for {name}'


class TestMoDBlock:
    """Test suite for the MoDBlock."""

    def _make_config(self, **kwargs):
        """Create a ModelConfig for testing."""
        defaults = dict(
            embed_dim=64,
            num_heads=4,
            num_kv_heads=2,
            context_length=32,
            vocab_size=256,
            num_layers=2,
            dropout=0.0,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)

    def _make_block(self, capacity=0.5, **kwargs):
        """Create a MoDBlock for testing."""
        config = self._make_config(**kwargs)
        block_config = BlockConfig(
            attention='mha',
            ffn='swiglu',
            norm='rmsnorm',
        )
        return MoDBlock(config, block_config, capacity=capacity)

    def test_initialization(self):
        """Test MoDBlock initializes correctly."""
        block = self._make_block()
        assert hasattr(block, 'router')
        assert hasattr(block, 'block')
        assert isinstance(block.block, ConfigurableBlock)

    def test_forward_shape(self):
        """Test output shape matches input."""
        block = self._make_block()
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_forward_deterministic_eval(self):
        """Test eval mode produces identical outputs."""
        block = self._make_block()
        block.eval()
        x = torch.randn(1, 8, 64)

        with torch.no_grad():
            out1 = block(x)
            out2 = block(x)

        torch.testing.assert_close(out1, out2)

    def test_gradient_flow(self):
        """Test gradients flow through the MoD block."""
        block = self._make_block()
        x = torch.randn(2, 8, 64)

        out = block(x)
        out.sum().backward()

        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f'No grad for {name}'
                assert not torch.isnan(param.grad).any(), (
                    f'NaN grad for {name}'
                )

    def test_capacity_zero_is_identity(self):
        """With capacity=0, no tokens are processed (pure residual)."""
        block = self._make_block(capacity=0.0)
        block.eval()
        x = torch.randn(1, 8, 64)

        with torch.no_grad():
            out = block(x)

        # Output should equal input (no processing)
        torch.testing.assert_close(out, x)

    def test_aux_loss_exposed(self):
        """Test that auxiliary loss is accessible."""
        block = self._make_block()
        x = torch.randn(2, 8, 64)
        block(x)

        assert hasattr(block, 'aux_loss')
        assert block.aux_loss.shape == ()
        assert block.aux_loss.item() >= 0

    def test_different_capacities(self):
        """Test with various capacity values."""
        x = torch.randn(2, 8, 64)

        for cap in [0.0, 0.25, 0.5, 0.75, 1.0]:
            block = self._make_block(capacity=cap)
            out = block(x)
            assert out.shape == x.shape

    def test_single_token(self):
        """Test with seq_len=1."""
        block = self._make_block(capacity=1.0)
        x = torch.randn(1, 1, 64)
        out = block(x)
        assert out.shape == (1, 1, 64)
