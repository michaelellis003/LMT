"""Tests for Direct Preference Optimization (DPO) loss."""

import torch

from lmt.training.dpo import dpo_loss


class TestDPOLoss:
    """Test the DPO loss function."""

    def test_basic_output_shape(self):
        """DPO loss returns a scalar."""
        # policy prefers chosen, ref is neutral -> low loss
        policy_chosen_logps = torch.tensor([-1.0, -2.0])
        policy_rejected_logps = torch.tensor([-3.0, -4.0])
        ref_chosen_logps = torch.tensor([-1.5, -2.5])
        ref_rejected_logps = torch.tensor([-3.5, -4.5])

        loss = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )
        assert loss.dim() == 0  # scalar

    def test_loss_is_finite(self):
        """DPO loss should be finite for normal inputs."""
        policy_chosen = torch.tensor([-1.0, -2.0])
        policy_rejected = torch.tensor([-3.0, -4.0])
        ref_chosen = torch.tensor([-1.5, -2.5])
        ref_rejected = torch.tensor([-3.5, -4.5])

        loss = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected
        )
        assert torch.isfinite(loss)

    def test_loss_zero_when_perfectly_aligned(self):
        """When policy matches ref exactly, loss depends on margin."""
        # If policy and ref have same log-ratios, the DPO objective
        # reduces to -log(sigma(0)) = log(2) per sample
        logps = torch.tensor([-1.0, -2.0])
        loss = dpo_loss(logps, logps, logps, logps)
        expected = torch.log(torch.tensor(2.0))
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_loss_decreases_when_policy_prefers_chosen(self):
        """Loss should be lower when policy prefers chosen over ref."""
        ref_chosen = torch.tensor([-2.0])
        ref_rejected = torch.tensor([-2.0])

        # Policy strongly prefers chosen
        loss_good = dpo_loss(
            torch.tensor([-0.5]),  # high prob chosen
            torch.tensor([-5.0]),  # low prob rejected
            ref_chosen,
            ref_rejected,
        )

        # Policy is indifferent
        loss_bad = dpo_loss(
            torch.tensor([-2.0]),
            torch.tensor([-2.0]),
            ref_chosen,
            ref_rejected,
        )

        assert loss_good < loss_bad

    def test_beta_scaling(self):
        """Higher beta means stronger preference signal."""
        policy_chosen = torch.tensor([-1.0])
        policy_rejected = torch.tensor([-3.0])
        ref_chosen = torch.tensor([-2.0])
        ref_rejected = torch.tensor([-2.0])

        loss_low_beta = dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            beta=0.05,
        )
        loss_high_beta = dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            beta=0.5,
        )

        # With policy preferring chosen, higher beta -> lower loss
        # (stronger signal through the sigmoid)
        assert loss_high_beta < loss_low_beta

    def test_gradient_flows(self):
        """Gradients should flow through policy log-probs."""
        policy_chosen = torch.tensor([-1.0, -2.0], requires_grad=True)
        policy_rejected = torch.tensor([-3.0, -4.0], requires_grad=True)
        ref_chosen = torch.tensor([-1.5, -2.5])
        ref_rejected = torch.tensor([-3.5, -4.5])

        loss = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected
        )
        loss.backward()
        assert policy_chosen.grad is not None
        assert policy_rejected.grad is not None

    def test_no_gradient_to_ref(self):
        """dpo_loss should internally detach reference log-probs."""
        policy_chosen = torch.tensor([-1.0, -2.0], requires_grad=True)
        policy_rejected = torch.tensor([-3.0, -4.0], requires_grad=True)
        ref_chosen = torch.tensor([-1.5, -2.5], requires_grad=True)
        ref_rejected = torch.tensor([-3.5, -4.5], requires_grad=True)

        # Pass ref tensors WITHOUT detaching -- dpo_loss should
        # handle detachment internally
        loss = dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )
        loss.backward()
        # Policy gets gradients
        assert policy_chosen.grad is not None
        # dpo_loss detaches ref internally, so no grad flows
        assert ref_chosen.grad is None
        assert ref_rejected.grad is None

    def test_batch_dimension(self):
        """Works with different batch sizes."""
        for batch_size in [1, 4, 16]:
            loss = dpo_loss(
                torch.randn(batch_size),
                torch.randn(batch_size),
                torch.randn(batch_size),
                torch.randn(batch_size),
            )
            assert loss.dim() == 0
            assert torch.isfinite(loss)


class TestDPOLogProbs:
    """Test the log-probability computation helper."""

    def test_log_probs_shape(self):
        """Per-sequence log-probs have correct shape."""
        from lmt.training.dpo import sequence_log_probs

        # [batch, seq_len, vocab_size]
        logits = torch.randn(2, 8, 64)
        # [batch, seq_len] - target token ids
        labels = torch.randint(0, 64, (2, 8))

        logps = sequence_log_probs(logits, labels)
        assert logps.shape == (2,)  # one scalar per sequence

    def test_log_probs_are_negative(self):
        """Log-probs should be non-positive."""
        from lmt.training.dpo import sequence_log_probs

        logits = torch.randn(4, 16, 128)
        labels = torch.randint(0, 128, (4, 16))

        logps = sequence_log_probs(logits, labels)
        assert (logps <= 0).all()

    def test_log_probs_with_mask(self):
        """Masked positions should not contribute."""
        from lmt.training.dpo import sequence_log_probs

        logits = torch.randn(2, 8, 64)
        labels = torch.randint(0, 64, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[0, 4:] = False  # mask second half of first sequence

        logps_masked = sequence_log_probs(logits, labels, mask=mask)
        logps_full = sequence_log_probs(logits, labels)

        # Masked version should differ since fewer tokens contribute
        assert not torch.allclose(logps_masked, logps_full)
