"""Tests for the math reasoning GRPO recipe.

Validates the wiring between math data loading, reward functions,
and the GRPO trainer. Uses tiny models and mock data.
"""

import torch

from lmt.models.config import ModelConfig
from lmt.training.grpo_trainer import GRPOConfig, GRPOTrainer
from lmt.training.reproducibility import set_seed


def _tiny_config() -> ModelConfig:
    """Create a tiny model config for testing."""
    return ModelConfig(
        embed_dim=32,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        ffn_hidden_dim=64,
        vocab_size=64,
        context_length=32,
        tie_weights=True,
        dropout=0.0,
    )


class TestRewardBridge:
    """Test bridging string rewards to tensor rewards."""

    def test_tensor_reward_fn(self):
        """A tensor reward fn should accept prompt and response tensors."""
        from lmt.training.rewards import math_reward

        # Simple reward fn that ignores the tensors and uses fixed strings
        def mock_tensor_reward(
            prompt_ids: torch.Tensor,
            response_ids: torch.Tensor,
        ) -> float:
            # In real use, you'd decode these with a tokenizer
            return math_reward(
                prompt='What is 2+2?',
                response=r'\boxed{4}',
                ground_truth='4',
            )

        # Should return 1.0 for correct answer
        prompt = torch.tensor([1, 2, 3])
        response = torch.tensor([4, 5, 6])
        assert mock_tensor_reward(prompt, response) == 1.0

    def test_create_math_reward_fn(self):
        """Create a tensor-level reward fn from math data."""
        from lmt.data.math_data import MathDataItem
        from lmt.recipes.reasoning import create_math_reward_fn

        item = MathDataItem(prompt='What is 2+2?', ground_truth='4')

        # Mock tokenizer that returns fixed strings
        class MockTokenizer:
            """Mock tokenizer for testing."""

            def decode(self, ids: list[int]) -> str:
                """Return a fixed response with correct answer."""
                return r'The answer is \boxed{4}'

        reward_fn = create_math_reward_fn(item, MockTokenizer())
        prompt = torch.tensor([1, 2, 3])
        response = torch.tensor([4, 5, 6])
        assert reward_fn(prompt, response) == 1.0

    def test_wrong_answer_reward(self):
        """Wrong answer should get 0.0 reward."""
        from lmt.data.math_data import MathDataItem
        from lmt.recipes.reasoning import create_math_reward_fn

        item = MathDataItem(prompt='What is 2+2?', ground_truth='4')

        class MockTokenizer:
            """Mock tokenizer returning wrong answer."""

            def decode(self, ids: list[int]) -> str:
                """Return wrong answer."""
                return r'The answer is \boxed{5}'

        reward_fn = create_math_reward_fn(item, MockTokenizer())
        prompt = torch.tensor([1, 2, 3])
        response = torch.tensor([4, 5, 6])
        assert reward_fn(prompt, response) == 0.0


class TestGRPOWithMathReward:
    """Test GRPO trainer with math reward function."""

    def test_single_train_step(self):
        """GRPO train_step should complete with a math reward fn."""
        from lmt.models.qwen3.qwen3 import Qwen3

        set_seed(42)
        config = _tiny_config()
        model = Qwen3(config)

        grpo_config = GRPOConfig(
            group_size=2,
            max_response_len=8,
            lr=1e-4,
            temperature=1.0,
            device='cpu',
        )

        trainer = GRPOTrainer(model, grpo_config)

        # Simple reward fn: reward=1 if response contains token 5
        def simple_reward(
            prompt_ids: torch.Tensor,
            response_ids: torch.Tensor,
        ) -> float:
            return 1.0 if 5 in response_ids else 0.0

        prompt = torch.randint(0, config.vocab_size, (4,))
        loss = trainer.train_step(prompt, simple_reward)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_multiple_train_steps(self):
        """Multiple GRPO steps should not error."""
        from lmt.models.qwen3.qwen3 import Qwen3

        set_seed(42)
        config = _tiny_config()
        model = Qwen3(config)

        grpo_config = GRPOConfig(
            group_size=2,
            max_response_len=8,
            lr=1e-4,
            temperature=1.0,
            device='cpu',
        )

        trainer = GRPOTrainer(model, grpo_config)

        def const_reward(
            prompt_ids: torch.Tensor,
            response_ids: torch.Tensor,
        ) -> float:
            return 0.5

        losses = []
        for _ in range(3):
            prompt = torch.randint(0, config.vocab_size, (4,))
            loss = trainer.train_step(prompt, const_reward)
            losses.append(loss)

        assert len(losses) == 3
        assert all(isinstance(v, float) for v in losses)
