"""Tests for mixed precision (AMP) training utility."""

import torch
import torch.nn as nn

from lmt.training.mixed_precision import AMPContext


class TestAMPContext:
    """Test AMPContext for mixed precision training."""

    def test_init_disabled(self):
        """Should create a disabled context by default on CPU."""
        ctx = AMPContext(enabled=False)
        assert not ctx.enabled
        assert ctx.scaler is None

    def test_init_bf16(self):
        """Should not create scaler for bf16 (no scaling needed)."""
        ctx = AMPContext(enabled=True, dtype=torch.bfloat16)
        assert ctx.enabled
        assert ctx.scaler is None

    def test_init_fp16(self):
        """Should create scaler for fp16."""
        ctx = AMPContext(enabled=True, dtype=torch.float16)
        assert ctx.enabled
        assert ctx.scaler is not None

    def test_autocast_disabled(self):
        """Should return a no-op context when disabled."""
        ctx = AMPContext(enabled=False)
        model = nn.Linear(4, 2)
        x = torch.randn(2, 4)
        with ctx.autocast():
            out = model(x)
        # Output should still be float32
        assert out.dtype == torch.float32

    def test_autocast_bf16_on_cpu(self):
        """Should autocast to bf16 on CPU."""
        ctx = AMPContext(enabled=True, dtype=torch.bfloat16, device='cpu')
        model = nn.Linear(4, 2)
        x = torch.randn(2, 4)
        with ctx.autocast():
            out = model(x)
        # Linear on CPU with bf16 autocast produces bf16
        assert out.dtype == torch.bfloat16

    def test_scale_loss_disabled(self):
        """Should return loss unchanged when disabled."""
        ctx = AMPContext(enabled=False)
        loss = torch.tensor(3.0, requires_grad=True)
        scaled = ctx.scale_loss(loss)
        assert torch.allclose(scaled, loss)

    def test_scale_loss_bf16(self):
        """Should return loss unchanged for bf16 (no scaler)."""
        ctx = AMPContext(enabled=True, dtype=torch.bfloat16)
        loss = torch.tensor(3.0, requires_grad=True)
        scaled = ctx.scale_loss(loss)
        assert torch.allclose(scaled, loss)

    def test_unscale_and_step(self):
        """Should unscale gradients and step optimizer."""
        ctx = AMPContext(enabled=False)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        x = torch.randn(2, 4)
        loss = model(x).mean()
        loss.backward()

        old_weight = model.weight.data.clone()
        ctx.unscale_and_step(optimizer)
        # Weight should have changed
        assert not torch.allclose(model.weight.data, old_weight)

    def test_unscale_and_step_with_clip(self):
        """Should clip gradients during unscale_and_step."""
        ctx = AMPContext(enabled=False)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Large input to create big gradients
        x = torch.randn(2, 4) * 100
        loss = model(x).mean()
        loss.backward()

        ctx.unscale_and_step(optimizer, max_grad_norm=0.01, model=model)

    def test_update_scaler_disabled(self):
        """Should be no-op when disabled."""
        ctx = AMPContext(enabled=False)
        ctx.update_scaler()  # Should not raise

    def test_full_training_step_disabled(self):
        """Should work as a complete training step when disabled."""
        ctx = AMPContext(enabled=False)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        x = torch.randn(2, 4)
        with ctx.autocast():
            loss = model(x).mean()

        scaled = ctx.scale_loss(loss)
        scaled.backward()
        ctx.unscale_and_step(optimizer)
        ctx.update_scaler()
        optimizer.zero_grad()

    def test_full_training_step_bf16(self):
        """Should work as a complete bf16 training step on CPU."""
        ctx = AMPContext(enabled=True, dtype=torch.bfloat16, device='cpu')
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        old_weight = model.weight.data.clone()

        x = torch.randn(2, 4)
        with ctx.autocast():
            loss = model(x).mean()

        scaled = ctx.scale_loss(loss)
        scaled.backward()
        ctx.unscale_and_step(optimizer)
        ctx.update_scaler()
        optimizer.zero_grad()

        # Weights should have updated
        assert not torch.allclose(model.weight.data, old_weight)
