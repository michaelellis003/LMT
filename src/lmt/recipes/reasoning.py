"""Utilities for math reasoning GRPO training recipes.

Bridges string-based reward functions (which compare text answers)
to tensor-based reward functions (which the GRPO trainer expects).
This requires a tokenizer to decode generated token sequences.

Usage::

    from lmt.data.math_data import MathDataItem, load_math_items
    from lmt.recipes.reasoning import create_math_reward_fn

    items = load_math_items(rows, dataset_format='gsm8k')
    reward_fn = create_math_reward_fn(items[0], tokenizer)
    loss = trainer.train_step(prompt_ids, reward_fn)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from lmt.data.math_data import MathDataItem
from lmt.training.rewards import math_reward


def create_math_reward_fn(
    item: MathDataItem,
    tokenizer: Any,
) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """Create a tensor-level reward function from a MathDataItem.

    Wraps the string-based ``math_reward`` function by decoding
    the response tensor with the provided tokenizer, then comparing
    against the ground truth answer.

    Args:
        item: Math problem with ground truth answer.
        tokenizer: Object with a ``decode(ids) -> str`` method.

    Returns:
        Callable(prompt_ids, response_ids) -> float reward.
    """

    def reward_fn(
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> float:
        response_text = tokenizer.decode(response_ids.tolist())
        return math_reward(
            prompt=item.prompt,
            response=response_text,
            ground_truth=item.ground_truth,
        )

    return reward_fn
