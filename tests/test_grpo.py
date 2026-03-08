"""Tests for Group Relative Policy Optimization (GRPO) loss."""

import torch

from lmt.training.grpo import compute_group_advantages, grpo_loss


class TestGroupAdvantages:
    """Test group-relative advantage computation."""

    def test_output_shape(self):
        """Advantages have same shape as rewards."""
        rewards = torch.tensor([1.0, 0.5, -0.5, 0.0])
        adv = compute_group_advantages(rewards)
        assert adv.shape == rewards.shape

    def test_zero_mean(self):
        """Group advantages should have zero mean."""
        rewards = torch.tensor([1.0, 0.5, -0.5, 0.0, 0.3])
        adv = compute_group_advantages(rewards)
        assert torch.allclose(adv.mean(), torch.tensor(0.0), atol=1e-6)

    def test_unit_variance(self):
        """Group advantages should have unit variance."""
        rewards = torch.tensor([1.0, 0.5, -0.5, 0.0, 0.3, -0.2])
        adv = compute_group_advantages(rewards)
        assert torch.allclose(adv.std(), torch.tensor(1.0), atol=0.1)

    def test_ordering_preserved(self):
        """Higher reward -> higher advantage."""
        rewards = torch.tensor([1.0, 0.5, -0.5])
        adv = compute_group_advantages(rewards)
        assert adv[0] > adv[1] > adv[2]

    def test_single_element(self):
        """Single element group gives zero advantage."""
        rewards = torch.tensor([5.0])
        adv = compute_group_advantages(rewards)
        assert torch.allclose(adv, torch.tensor([0.0]))

    def test_equal_rewards(self):
        """Equal rewards give zero advantages."""
        rewards = torch.tensor([1.0, 1.0, 1.0])
        adv = compute_group_advantages(rewards)
        assert torch.allclose(adv, torch.zeros(3), atol=1e-6)


class TestGRPOLoss:
    """Test the GRPO loss function."""

    def test_basic_output_is_scalar(self):
        """GRPO loss returns a scalar."""
        group_size = 4
        seq_len = 8

        loss = grpo_loss(
            policy_logps=torch.randn(group_size, seq_len),
            old_logps=torch.randn(group_size, seq_len),
            ref_logps=torch.randn(group_size, seq_len),
            rewards=torch.randn(group_size),
        )
        assert loss.dim() == 0

    def test_loss_is_finite(self):
        """GRPO loss should be finite for normal inputs."""
        loss = grpo_loss(
            policy_logps=torch.randn(4, 8),
            old_logps=torch.randn(4, 8),
            ref_logps=torch.randn(4, 8),
            rewards=torch.randn(4),
        )
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        """Gradients flow through policy log-probs."""
        policy_logps = torch.randn(4, 8, requires_grad=True)

        loss = grpo_loss(
            policy_logps=policy_logps,
            old_logps=torch.randn(4, 8),
            ref_logps=torch.randn(4, 8),
            rewards=torch.randn(4),
        )
        loss.backward()
        assert policy_logps.grad is not None

    def test_clipping_limits_ratio(self):
        """Clipped loss should bound the effective ratio."""
        # Make policy very different from old policy
        old_logps = torch.zeros(4, 8)

        # Very high ratio (policy >> old)
        high_policy = torch.ones(4, 8) * 5.0
        high_policy.requires_grad_(True)

        loss_high = grpo_loss(
            policy_logps=high_policy,
            old_logps=old_logps,
            ref_logps=torch.zeros(4, 8),
            rewards=torch.ones(4),
            clip_eps=0.2,
        )

        # Moderate ratio
        mod_policy = torch.ones(4, 8) * 0.1
        mod_policy.requires_grad_(True)

        loss_mod = grpo_loss(
            policy_logps=mod_policy,
            old_logps=old_logps,
            ref_logps=torch.zeros(4, 8),
            rewards=torch.ones(4),
            clip_eps=0.2,
        )

        # Both should be finite (clipping prevents explosion)
        assert torch.isfinite(loss_high)
        assert torch.isfinite(loss_mod)

    def test_kl_penalty(self):
        """Higher KL coefficient should increase loss when policy != ref."""
        policy_logps = torch.randn(4, 8)
        ref_logps = policy_logps + 1.0  # different from policy

        loss_low_kl = grpo_loss(
            policy_logps=policy_logps,
            old_logps=policy_logps.clone(),
            ref_logps=ref_logps,
            rewards=torch.ones(4),
            kl_coeff=0.01,
        )
        loss_high_kl = grpo_loss(
            policy_logps=policy_logps,
            old_logps=policy_logps.clone(),
            ref_logps=ref_logps,
            rewards=torch.ones(4),
            kl_coeff=1.0,
        )
        # Higher KL penalty should give higher loss
        assert loss_high_kl > loss_low_kl

    def test_kl_is_nonnegative(self):
        """KL penalty should be non-negative for any inputs."""
        for _ in range(10):
            policy_logps = torch.randn(4, 8, requires_grad=True)
            ref_logps = torch.randn(4, 8)

            # Isolate KL term by setting kl_coeff=1 and using equal
            # rewards (so policy_loss component is ~0)
            loss = grpo_loss(
                policy_logps=policy_logps,
                old_logps=policy_logps.detach(),
                ref_logps=ref_logps,
                rewards=torch.ones(4),  # equal rewards -> zero advantages
                kl_coeff=1.0,
            )
            # With zero advantages, loss ≈ KL term, which must be >= 0
            assert loss >= -1e-6, f'KL penalty was negative: {loss.item()}'

    def test_different_group_sizes(self):
        """Works with different group sizes."""
        for g in [1, 2, 8, 16]:
            loss = grpo_loss(
                policy_logps=torch.randn(g, 8),
                old_logps=torch.randn(g, 8),
                ref_logps=torch.randn(g, 8),
                rewards=torch.randn(g),
            )
            assert loss.dim() == 0
            assert torch.isfinite(loss)
