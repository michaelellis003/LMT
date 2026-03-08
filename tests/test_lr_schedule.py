"""Tests for learning rate schedules.

Validates cosine schedule with linear warmup, the standard choice
for modern transformer pretraining (GPT-3, LLaMA, etc.).
"""

import math

import torch


class TestCosineWarmupSchedule:
    """Test the cosine LR schedule with linear warmup."""

    def test_starts_at_zero(self):
        """LR should start near zero during warmup."""
        from lmt.training.lr_schedule import cosine_warmup_lr

        lr = cosine_warmup_lr(
            step=0, max_lr=1e-3, warmup_steps=100, total_steps=1000
        )
        assert lr < 1e-5  # Near zero at step 0

    def test_reaches_max_lr_after_warmup(self):
        """LR should reach max_lr at the end of warmup."""
        from lmt.training.lr_schedule import cosine_warmup_lr

        lr = cosine_warmup_lr(
            step=100,
            max_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )
        assert abs(lr - 1e-3) < 1e-8

    def test_linear_warmup(self):
        """Warmup phase should increase linearly."""
        from lmt.training.lr_schedule import cosine_warmup_lr

        max_lr = 1e-3
        warmup_steps = 100

        # Check midpoint of warmup
        lr_50 = cosine_warmup_lr(
            step=50,
            max_lr=max_lr,
            warmup_steps=warmup_steps,
            total_steps=1000,
        )
        assert abs(lr_50 - 0.5 * max_lr) < 1e-8

    def test_cosine_decay(self):
        """After warmup, LR should follow cosine decay."""
        from lmt.training.lr_schedule import cosine_warmup_lr

        max_lr = 1e-3
        warmup = 100
        total = 1000

        # At midpoint of decay phase
        mid_step = warmup + (total - warmup) // 2
        lr = cosine_warmup_lr(
            step=mid_step,
            max_lr=max_lr,
            warmup_steps=warmup,
            total_steps=total,
        )

        # Cosine at midpoint should be about half of max_lr
        expected = max_lr * 0.5 * (1 + math.cos(math.pi * 0.5))
        assert abs(lr - expected) < 1e-8

    def test_reaches_min_lr_at_end(self):
        """LR should reach min_lr at the final step."""
        from lmt.training.lr_schedule import cosine_warmup_lr

        lr = cosine_warmup_lr(
            step=1000,
            max_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
            min_lr=1e-5,
        )
        assert abs(lr - 1e-5) < 1e-8

    def test_default_min_lr_is_zero(self):
        """Without min_lr, should decay to zero."""
        from lmt.training.lr_schedule import cosine_warmup_lr

        lr = cosine_warmup_lr(
            step=1000,
            max_lr=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )
        assert abs(lr) < 1e-8

    def test_never_negative(self):
        """LR should never be negative, even beyond total_steps."""
        from lmt.training.lr_schedule import cosine_warmup_lr

        for step in range(0, 1200):
            lr = cosine_warmup_lr(
                step=step,
                max_lr=1e-3,
                warmup_steps=100,
                total_steps=1000,
            )
            assert lr >= 0, f'Negative LR at step {step}: {lr}'

    def test_monotonic_during_warmup(self):
        """LR should strictly increase during warmup."""
        from lmt.training.lr_schedule import cosine_warmup_lr

        prev = -1.0
        for step in range(101):
            lr = cosine_warmup_lr(
                step=step,
                max_lr=1e-3,
                warmup_steps=100,
                total_steps=1000,
            )
            assert lr >= prev, f'LR decreased during warmup at step {step}'
            prev = lr

    def test_monotonic_during_decay(self):
        """LR should monotonically decrease during cosine decay."""
        from lmt.training.lr_schedule import cosine_warmup_lr

        prev = float('inf')
        for step in range(100, 1001):
            lr = cosine_warmup_lr(
                step=step,
                max_lr=1e-3,
                warmup_steps=100,
                total_steps=1000,
            )
            assert lr <= prev + 1e-10, (
                f'LR increased during decay at step {step}'
            )
            prev = lr

    def test_works_with_optimizer(self):
        """Should integrate with PyTorch optimizer via LambdaLR."""
        from lmt.training.lr_schedule import get_cosine_warmup_scheduler

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = get_cosine_warmup_scheduler(
            optimizer=optimizer,
            warmup_steps=10,
            total_steps=100,
        )

        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()

        # LR should increase then decrease
        assert lrs[0] < lrs[10]  # warmup
        assert lrs[10] > lrs[99]  # decay
