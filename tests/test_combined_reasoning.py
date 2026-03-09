"""Tests for combined math + code GRPO training."""

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


class TestMultiTaskGRPO:
    """Test alternating between math and code rewards in GRPO."""

    def test_alternating_reward_functions(self):
        """Should handle alternating between different reward types."""
        from lmt.models.qwen3.qwen3 import Qwen3
        from lmt.recipes.multi_task import create_multi_task_iterator

        set_seed(42)
        model = Qwen3(_tiny_config())

        grpo_config = GRPOConfig(
            group_size=2,
            max_response_len=8,
            lr=1e-4,
            temperature=1.0,
            device='cpu',
        )
        trainer = GRPOTrainer(model, grpo_config)

        # Create alternating math/code tasks
        def math_reward(
            prompt_ids: torch.Tensor,
            response_ids: torch.Tensor,
        ) -> float:
            return 0.5

        def code_reward(
            prompt_ids: torch.Tensor,
            response_ids: torch.Tensor,
        ) -> float:
            return 0.7

        tasks = create_multi_task_iterator(
            task_reward_fns={
                'math': [(torch.randint(0, 64, (4,)), math_reward)],
                'code': [(torch.randint(0, 64, (4,)), code_reward)],
            },
            strategy='alternate',
        )

        losses = {}
        for task_name, prompt, reward_fn in tasks:
            loss = trainer.train_step(prompt, reward_fn)
            losses[task_name] = loss

        assert 'math' in losses
        assert 'code' in losses
        assert all(isinstance(v, float) for v in losses.values())

    def test_round_robin_strategy(self):
        """Round-robin should cycle through all tasks."""
        from lmt.recipes.multi_task import create_multi_task_iterator

        tasks_a = [(torch.tensor([1, 2]), lambda p, r: 0.5)] * 2
        tasks_b = [(torch.tensor([3, 4]), lambda p, r: 0.7)] * 2

        iterator = create_multi_task_iterator(
            task_reward_fns={'a': tasks_a, 'b': tasks_b},
            strategy='alternate',
        )

        names = [name for name, _, _ in iterator]
        assert names == ['a', 'b', 'a', 'b']

    def test_proportional_strategy(self):
        """Proportional should respect task weights."""
        from lmt.recipes.multi_task import create_multi_task_iterator

        tasks_a = [(torch.tensor([1]), lambda p, r: 0.5)] * 3
        tasks_b = [(torch.tensor([2]), lambda p, r: 0.7)] * 1

        iterator = create_multi_task_iterator(
            task_reward_fns={'a': tasks_a, 'b': tasks_b},
            strategy='sequential',
        )

        names = [name for name, _, _ in iterator]
        # Sequential: all of 'a' first, then all of 'b'
        assert names == ['a', 'a', 'a', 'b']
