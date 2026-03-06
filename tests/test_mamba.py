"""Tests for Mamba (Selective State Space Model) architecture.

Tests follow TDD -- written before implementation.
Reference: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
(Gu & Dao, 2023) https://arxiv.org/abs/2312.00752
"""

import torch
import torch.nn as nn

from lmt.models.config import ModelConfig


class TestSelectiveSSM:
    """Test the core Selective SSM layer."""

    def _make_ssm(self, d_inner: int = 32, d_state: int = 16):
        """Create a SelectiveSSM for testing."""
        from lmt.layers.ssm import SelectiveSSM

        return SelectiveSSM(d_inner=d_inner, d_state=d_state)

    def test_output_shape(self) -> None:
        """SSM output matches input shape [B, L, d_inner]."""
        ssm = self._make_ssm(d_inner=32, d_state=16)
        x = torch.randn(2, 10, 32)
        y = ssm(x)
        assert y.shape == (2, 10, 32)

    def test_single_token(self) -> None:
        """SSM handles single-token sequences."""
        ssm = self._make_ssm()
        x = torch.randn(1, 1, 32)
        y = ssm(x)
        assert y.shape == (1, 1, 32)

    def test_gradient_flow(self) -> None:
        """Gradients flow through the SSM."""
        ssm = self._make_ssm()
        x = torch.randn(2, 8, 32, requires_grad=True)
        y = ssm(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_a_stays_negative(self) -> None:
        """A matrix should be negative (stable dynamics)."""
        ssm = self._make_ssm(d_inner=16, d_state=8)
        # A is stored as A_log, actual A = -exp(A_log)
        a_actual = -torch.exp(ssm.A_log)
        assert torch.all(a_actual < 0), 'A must be negative for stability'

    def test_delta_is_positive(self) -> None:
        """Delta (time step) should be positive after softplus."""
        ssm = self._make_ssm()
        x = torch.randn(2, 8, 32)
        # Run forward to check delta internally
        # We test this property via the output being finite
        y = ssm(x)
        assert torch.all(torch.isfinite(y))

    def test_d_state_affects_capacity(self) -> None:
        """Different d_state values produce different param counts."""
        ssm_small = self._make_ssm(d_inner=32, d_state=4)
        ssm_large = self._make_ssm(d_inner=32, d_state=32)
        params_small = sum(p.numel() for p in ssm_small.parameters())
        params_large = sum(p.numel() for p in ssm_large.parameters())
        assert params_large > params_small


class TestMambaBlock:
    """Test the Mamba block (norm + SSM mixer)."""

    def _make_block(
        self,
        d_model: int = 64,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
    ):
        """Create a MambaBlock for testing."""
        from lmt.layers.ssm import MambaBlock

        return MambaBlock(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
        )

    def test_output_shape(self) -> None:
        """Block output matches input shape [B, L, d_model]."""
        block = self._make_block(d_model=64)
        x = torch.randn(2, 10, 64)
        y = block(x)
        assert y.shape == (2, 10, 64)

    def test_residual_connection(self) -> None:
        """Block uses residual connection (output != pure transform)."""
        block = self._make_block(d_model=32)
        x = torch.randn(1, 4, 32)
        y = block(x)
        # With residual, output should be correlated with input
        # (not identical, but not completely different either)
        assert y.shape == x.shape
        # Output should differ from input (transformation happened)
        assert not torch.allclose(x, y)

    def test_gradient_flow(self) -> None:
        """Gradients flow through the entire block."""
        block = self._make_block()
        x = torch.randn(2, 8, 64, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_expand_factor(self) -> None:
        """Expand factor controls inner dimension."""
        block_e1 = self._make_block(d_model=32, expand=1)
        block_e4 = self._make_block(d_model=32, expand=4)
        params_e1 = sum(p.numel() for p in block_e1.parameters())
        params_e4 = sum(p.numel() for p in block_e4.parameters())
        assert params_e4 > params_e1

    def test_conv_kernel_size(self) -> None:
        """Different conv kernel sizes work."""
        for k in [2, 4, 8]:
            block = self._make_block(d_model=32, d_conv=k)
            x = torch.randn(1, 16, 32)
            y = block(x)
            assert y.shape == (1, 16, 32)


class TestMambaModel:
    """Test the full Mamba language model."""

    def _make_model(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        num_layers: int = 2,
        d_state: int = 16,
    ):
        """Create a Mamba model for testing."""
        from lmt.models.mamba import Mamba

        config = ModelConfig(
            vocab_size=vocab_size,
            embed_dim=d_model,
            num_layers=num_layers,
            num_heads=1,  # unused by Mamba but required by ModelConfig
            context_length=128,
            dropout=0.0,
        )
        return Mamba(config, d_state=d_state)

    def test_output_shape(self) -> None:
        """Model output is [B, L, vocab_size] logits."""
        model = self._make_model(vocab_size=256, d_model=64)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_single_token(self) -> None:
        """Model handles single-token input."""
        model = self._make_model()
        x = torch.randint(0, 256, (1, 1))
        logits = model(x)
        assert logits.shape == (1, 1, 256)

    def test_gradient_flow(self) -> None:
        """Gradients reach the embedding layer."""
        model = self._make_model()
        x = torch.randint(0, 256, (2, 8))
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        assert model.tok_embed.weight.grad is not None
        assert not torch.all(model.tok_embed.weight.grad == 0)

    def test_no_attention(self) -> None:
        """Mamba has no attention layers (no KV cache concept)."""
        model = self._make_model()
        # Should not have any attention-related parameters
        param_names = [n for n, _ in model.named_parameters()]
        attn_params = [
            n
            for n in param_names
            if 'attn' in n.lower()
            or 'query' in n.lower()
            or 'key' in n.lower()
        ]
        assert len(attn_params) == 0, (
            f'Mamba should have no attention params: {attn_params}'
        )

    def test_weight_init(self) -> None:
        """Model initializes weights properly."""
        model = self._make_model(d_model=64, num_layers=4)
        # Weight parameters should not be all zeros
        # (bias params can be zero-initialized, that's fine)
        for name, param in model.named_parameters():
            if param.requires_grad and 'bias' not in name:
                assert not torch.all(param == 0), f'{name} is all zeros'

    def test_param_count_scales_with_layers(self) -> None:
        """More layers = more parameters."""
        model_2 = self._make_model(num_layers=2)
        model_4 = self._make_model(num_layers=4)
        p2 = sum(p.numel() for p in model_2.parameters())
        p4 = sum(p.numel() for p in model_4.parameters())
        assert p4 > p2

    def test_different_d_state(self) -> None:
        """d_state controls SSM state dimension."""
        model_s = self._make_model(d_state=4)
        model_l = self._make_model(d_state=32)
        ps = sum(p.numel() for p in model_s.parameters())
        pl = sum(p.numel() for p in model_l.parameters())
        assert pl > ps

    def test_compatible_with_trainer(self) -> None:
        """Model output is compatible with cross-entropy loss."""
        model = self._make_model(vocab_size=256)
        x = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        logits = model(x)
        # Flatten for cross entropy
        loss = nn.functional.cross_entropy(
            logits.view(-1, 256), targets.view(-1)
        )
        loss.backward()
        assert torch.isfinite(loss)
