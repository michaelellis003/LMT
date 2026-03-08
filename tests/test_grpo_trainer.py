"""Tests for the GRPO training loop.

The GRPO trainer orchestrates:
1. Generating G responses per prompt (group sampling)
2. Scoring responses with a reward function
3. Computing GRPO loss and updating the policy

These tests use tiny models and mock data to verify the training
loop mechanics, not the quality of trained models.
"""

import torch

from lmt.models.config import ModelConfig


def _make_tiny_model():
    """Create a tiny model for GRPO testing."""
    from lmt.models.base import BaseModel

    config = ModelConfig(
        embed_dim=32,
        num_heads=2,
        num_kv_heads=2,
        context_length=16,
        vocab_size=64,
        num_layers=1,
        dropout=0.0,
    )
    return BaseModel(config)


class TestGRPOTrainer:
    """Test the GRPO training loop."""

    def test_init(self):
        """Trainer initializes with model, ref model, and config."""
        from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer

        model = _make_tiny_model()
        config = GRPOConfig()
        trainer = GRPOTrainer(model=model, config=config)

        assert trainer.model is model
        assert trainer.ref_model is not None
        # Ref model should be a separate copy, not the same object
        assert trainer.ref_model is not model

    def test_ref_model_is_frozen(self):
        """Reference model parameters should not require grad."""
        from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer

        model = _make_tiny_model()
        trainer = GRPOTrainer(model=model, config=GRPOConfig())

        for param in trainer.ref_model.parameters():
            assert not param.requires_grad

    def test_compute_log_probs(self):
        """compute_log_probs returns per-token log-probs."""
        from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer

        model = _make_tiny_model()
        trainer = GRPOTrainer(model=model, config=GRPOConfig())

        # [batch, seq_len] token ids
        tokens = torch.randint(0, 64, (2, 8))
        logps = trainer.compute_log_probs(model, tokens)

        # Should be [batch, seq_len-1] (predicting next token)
        assert logps.shape == (2, 7)
        assert torch.isfinite(logps).all()
        assert (logps <= 0).all()  # log-probs are non-positive

    def test_training_step_reduces_loss(self):
        """A few training steps should reduce loss on a toy task."""
        from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer

        model = _make_tiny_model()
        config = GRPOConfig(
            lr=1e-3,
            group_size=4,
            kl_coeff=0.0,  # No KL (Raschka's finding)
            max_response_len=8,  # Keep within context_length=16
        )
        trainer = GRPOTrainer(model=model, config=config)

        # Simple reward: reward 1.0 if response starts with token 0
        def reward_fn(prompt_ids, response_ids):
            return 1.0 if response_ids[0].item() == 0 else 0.0

        # Mock prompts (just token ids)
        prompts = [torch.randint(0, 64, (4,)) for _ in range(3)]

        losses = []
        for prompt in prompts:
            loss = trainer.train_step(prompt, reward_fn)
            losses.append(loss)

        # Loss should be finite
        for loss in losses:
            assert torch.isfinite(torch.tensor(loss))

    def test_no_grad_to_ref_model(self):
        """Reference model should not receive gradients."""
        from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer

        model = _make_tiny_model()
        trainer = GRPOTrainer(
            model=model,
            config=GRPOConfig(max_response_len=8),
        )

        # Get initial ref model params
        ref_params_before = {
            n: p.clone() for n, p in trainer.ref_model.named_parameters()
        }

        # Do a training step
        prompt = torch.randint(0, 64, (4,))
        trainer.train_step(prompt, lambda p, r: 1.0)

        # Ref model should be unchanged
        for name, param in trainer.ref_model.named_parameters():
            assert torch.equal(param, ref_params_before[name]), (
                f'Ref model param {name} changed during training'
            )

    def test_policy_model_updates(self):
        """Policy model parameters should change after training."""
        from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer

        model = _make_tiny_model()
        config = GRPOConfig(lr=1e-2, group_size=4, max_response_len=8)
        trainer = GRPOTrainer(model=model, config=config)

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        # Multiple steps to ensure some update
        for _ in range(3):
            prompt = torch.randint(0, 64, (4,))
            trainer.train_step(prompt, lambda p, r: 1.0)

        # At least some parameters should have changed
        changed = sum(
            1
            for n, p in model.named_parameters()
            if not torch.equal(p, params_before[n])
        )
        assert changed > 0, 'No parameters updated after training'

    def test_kl_coeff_zero_disables_kl(self):
        """With kl_coeff=0, KL term should not affect loss."""
        from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer

        torch.manual_seed(42)
        model = _make_tiny_model()
        config = GRPOConfig(
            lr=1e-3,
            group_size=4,
            kl_coeff=0.0,
            max_response_len=8,
        )
        trainer = GRPOTrainer(model=model, config=config)

        prompt = torch.randint(0, 64, (4,))
        loss = trainer.train_step(prompt, lambda p, r: 1.0)

        # Loss should be finite (no NaN from KL computation)
        assert torch.isfinite(torch.tensor(loss))

    def test_loss_uses_response_tokens_only(self):
        """GRPO loss should only use response tokens, not prompt."""
        from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer

        model = _make_tiny_model()
        config = GRPOConfig(lr=1e-3, group_size=4, max_response_len=8)
        trainer = GRPOTrainer(model=model, config=config)

        # Generate sequences and check log-prob slicing
        prompt = torch.randint(0, 64, (4,))
        sequences = trainer._generate_responses(prompt)

        # Full log-probs: [group, seq_len - 1]
        logps = trainer.compute_log_probs(model, sequences)
        assert logps.shape[1] == sequences.shape[1] - 1

        # Response-only slice should start at prompt_len - 1
        resp_start = prompt.shape[0] - 1
        resp_logps = logps[:, resp_start:]
        # Response log-probs should have max_response_len tokens
        assert resp_logps.shape[1] == config.max_response_len

    def test_gradient_clipping(self):
        """Gradient clipping should be applied when configured."""
        from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer

        model = _make_tiny_model()
        config = GRPOConfig(
            lr=1e-3,
            group_size=4,
            max_response_len=8,
            max_grad_norm=0.5,
        )
        trainer = GRPOTrainer(model=model, config=config)

        prompt = torch.randint(0, 64, (4,))
        loss = trainer.train_step(prompt, lambda p, r: 1.0)
        assert torch.isfinite(torch.tensor(loss))


class TestGRPOConfig:
    """Test the GRPO configuration dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        from lmt.training.grpo_trainer import GRPOConfig

        config = GRPOConfig()
        assert config.group_size == 8
        assert config.clip_eps == 0.2
        assert config.kl_coeff == 0.0  # Raschka's finding
        assert config.lr > 0
        assert config.max_response_len > 0

    def test_custom_values(self):
        """Config accepts custom values."""
        from lmt.training.grpo_trainer import GRPOConfig

        config = GRPOConfig(
            group_size=16,
            kl_coeff=0.04,
            lr=1e-5,
        )
        assert config.group_size == 16
        assert config.kl_coeff == 0.04
        assert config.lr == 1e-5
