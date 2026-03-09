r"""Code reasoning utilities for GRPO training.

Provides the reward bridge between LMT's tensor-based GRPO trainer
and code execution rewards. Given a coding problem with test cases,
creates a reward function that:

1. Decodes model-generated tokens into text
2. Extracts code from markdown code blocks
3. Runs the code against test assertions
4. Returns graded reward (0.0 for syntax errors, partial credit
   for passing some tests, 1.0 for all tests passing)

Usage::

    from lmt.recipes.code_reasoning import CodeProblem, create_code_reward_fn

    problem = CodeProblem(
        prompt='Write add(a, b) that returns the sum.',
        tests='assert add(1, 2) == 3\\nassert add(0, 0) == 0',
    )
    reward_fn = create_code_reward_fn(problem, tokenizer)
    loss = trainer.train_step(prompt_ids, reward_fn)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from lmt.training.rewards import code_reward


@dataclass
class CodeProblem:
    """A coding problem with test assertions.

    Attributes:
        prompt: The problem description / instructions.
        tests: Newline-separated test assertions (e.g. ``assert f(1) == 2``).
        language: Programming language (default ``'python'``).
    """

    prompt: str
    tests: str
    language: str = 'python'


def extract_code_block(text: str) -> str:
    """Extract code from markdown-style triple-backtick blocks.

    If multiple code blocks are present, returns the last one
    (models often refine their answer in later blocks).
    If no code block is found, returns the full text stripped.

    Args:
        text: Model-generated text potentially containing code blocks.

    Returns:
        Extracted code string.
    """
    # Match ```python\n...\n``` or ```\n...\n```
    pattern = r'```(?:\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[-1].strip()
    return text.strip()


def create_code_reward_fn(
    problem: CodeProblem,
    tokenizer: Any,
    timeout: int = 10,
) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """Create a tensor-level reward function from a coding problem.

    Bridges between the GRPO trainer's tensor interface and the
    string-based code execution reward.

    Args:
        problem: The coding problem with tests.
        tokenizer: Tokenizer with a ``decode(ids) -> str`` method.
        timeout: Maximum execution time in seconds.

    Returns:
        Reward function accepting (prompt_ids, response_ids) tensors
        and returning a float reward.
    """

    def reward_fn(
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> float:
        response_text = tokenizer.decode(response_ids.tolist())
        extracted_code = extract_code_block(response_text)
        return code_reward(
            code=extracted_code,
            tests=problem.tests,
            language=problem.language,
            timeout=timeout,
        )

    return reward_fn
