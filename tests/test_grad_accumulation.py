"""Tests for gradient accumulation utility."""

import torch
import torch.nn as nn

from lmt.training.grad_accumulation import GradientAccumulator


class TestGradientAccumulator:
    """Test GradientAccumulator class."""

    def test_init(self):
        """Should initialize with accumulation steps."""
        acc = GradientAccumulator(accumulation_steps=4)
        assert acc.accumulation_steps == 4
        assert acc.micro_step == 0

    def test_default_one_step(self):
        """Should default to 1 (no accumulation)."""
        acc = GradientAccumulator()
        assert acc.accumulation_steps == 1

    def test_should_step_every_n(self):
        """Should signal optimizer step every N micro-steps."""
        acc = GradientAccumulator(accumulation_steps=3)
        assert not acc.should_step()  # micro_step=0, not ready
        acc.micro_step = 1
        assert not acc.should_step()
        acc.micro_step = 2
        assert not acc.should_step()  # only 2 of 3 done
        acc.micro_step = 3
        assert acc.should_step()  # all 3 micro-steps done

    def test_scale_loss(self):
        """Should scale loss by 1/accumulation_steps."""
        acc = GradientAccumulator(accumulation_steps=4)
        loss = torch.tensor(8.0)
        scaled = acc.scale_loss(loss)
        assert torch.allclose(scaled, torch.tensor(2.0))

    def test_scale_loss_no_accumulation(self):
        """Should not scale when accumulation_steps=1."""
        acc = GradientAccumulator(accumulation_steps=1)
        loss = torch.tensor(3.0)
        scaled = acc.scale_loss(loss)
        assert torch.allclose(scaled, torch.tensor(3.0))

    def test_step_resets_micro_step(self):
        """Should reset micro_step counter after optimizer step."""
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        acc = GradientAccumulator(accumulation_steps=2)

        # Simulate 2 micro-steps
        for _i in range(2):
            x = torch.randn(1, 4)
            loss = model(x).sum()
            acc.scale_loss(loss).backward()
            acc.micro_step += 1

        acc.step(optimizer)
        assert acc.micro_step == 0

    def test_step_zeros_grad(self):
        """Should zero gradients after optimizer step."""
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        acc = GradientAccumulator(accumulation_steps=1)

        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        acc.step(optimizer)

        # Grads should be zeroed
        for p in model.parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, torch.zeros_like(p.grad))

    def test_step_with_grad_clip(self):
        """Should clip gradients when max_grad_norm is set."""
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        acc = GradientAccumulator(accumulation_steps=1, max_grad_norm=0.1)

        # Create large gradients
        x = torch.randn(1, 4) * 100
        loss = model(x).sum()
        loss.backward()

        # Verify grads are large before clipping
        total_norm_before = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float('inf')
        )
        assert total_norm_before > 0.1

        # Re-create gradients and step (which clips)
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        acc.step(optimizer)

    def test_step_with_scheduler(self):
        """Should step scheduler when provided."""
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.5
        )
        acc = GradientAccumulator(accumulation_steps=1)

        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        acc.step(optimizer, scheduler=scheduler)

        # LR should have decayed
        assert optimizer.param_groups[0]['lr'] == 0.05

    def test_numerical_equivalence(self):
        """Accumulated gradients should match large-batch gradients.

        Uses mean() loss (standard practice) so that scaling by
        1/accumulation_steps produces identical gradients.
        """
        torch.manual_seed(42)

        # Large batch (batch_size=4, mean loss)
        model_large = nn.Linear(4, 2, bias=False)
        opt_large = torch.optim.SGD(model_large.parameters(), lr=0.1)
        x_all = torch.randn(4, 4)
        loss_large = model_large(x_all).mean()
        loss_large.backward()
        opt_large.step()
        weights_large = model_large.weight.data.clone()

        # Accumulated (2 micro-batches of size 2, mean loss each)
        torch.manual_seed(42)
        model_acc = nn.Linear(4, 2, bias=False)
        opt_acc = torch.optim.SGD(model_acc.parameters(), lr=0.1)
        acc = GradientAccumulator(accumulation_steps=2)

        x1 = x_all[:2]
        x2 = x_all[2:]

        loss1 = model_acc(x1).mean()
        acc.scale_loss(loss1).backward()
        acc.micro_step += 1

        loss2 = model_acc(x2).mean()
        acc.scale_loss(loss2).backward()
        acc.micro_step += 1

        acc.step(opt_acc)
        weights_acc = model_acc.weight.data.clone()

        assert torch.allclose(weights_large, weights_acc, atol=1e-6)

    def test_invalid_accumulation_steps(self):
        """Should reject non-positive accumulation steps."""
        import pytest

        with pytest.raises(ValueError):
            GradientAccumulator(accumulation_steps=0)
        with pytest.raises(ValueError):
            GradientAccumulator(accumulation_steps=-1)
