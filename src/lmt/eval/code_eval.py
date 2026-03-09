"""Code reasoning evaluation for LMT models.

Evaluates a model's ability to solve coding problems by generating
responses, extracting code, and running test assertions.

Usage::

    from lmt.eval.code_eval import evaluate_code_accuracy

    result = evaluate_code_accuracy(
        model=model,
        problems=problems,
        tokenizer=tokenizer,
        max_new_tokens=256,
    )
    print(f'Pass@1: {result["accuracy"]:.1%}')
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lmt.eval.math_eval import _greedy_generate
from lmt.recipes.code_reasoning import CodeProblem, extract_code_block
from lmt.training.rewards import code_reward


def evaluate_code_accuracy(
    model: nn.Module,
    problems: list[CodeProblem],
    tokenizer: Any,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    timeout: int = 10,
) -> dict[str, Any]:
    """Evaluate code generation accuracy.

    For each coding problem, generates a response, extracts code,
    and runs it against the test assertions.

    Args:
        model: The language model to evaluate.
        problems: List of coding problems with test assertions.
        tokenizer: Tokenizer with ``encode`` and ``decode`` methods.
        max_new_tokens: Maximum tokens to generate per problem.
        temperature: Sampling temperature (0.0 = greedy).
        timeout: Maximum execution time per problem in seconds.

    Returns:
        Dict with ``accuracy`` (pass@1), ``num_correct``,
        ``num_total``, ``avg_reward``, and ``details``.
    """
    if not problems:
        return {
            'accuracy': 0.0,
            'num_correct': 0,
            'num_total': 0,
            'avg_reward': 0.0,
            'details': [],
        }

    model.eval()
    device = next(model.parameters()).device

    num_correct = 0
    total_reward = 0.0
    details: list[dict[str, Any]] = []

    for problem in problems:
        prompt_ids = tokenizer.encode(problem.prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            generated = _greedy_generate(
                model, input_ids, max_new_tokens, temperature
            )

        response_ids = generated[0, len(prompt_ids) :].tolist()
        response_text = tokenizer.decode(response_ids)

        extracted = extract_code_block(response_text)
        reward = code_reward(
            code=extracted,
            tests=problem.tests,
            language=problem.language,
            timeout=timeout,
        )

        correct = reward == 1.0
        if correct:
            num_correct += 1
        total_reward += reward

        details.append(
            {
                'prompt': problem.prompt,
                'response': response_text,
                'extracted_code': extracted,
                'reward': reward,
                'correct': correct,
            }
        )

    accuracy = num_correct / len(problems)
    avg_reward = total_reward / len(problems)

    return {
        'accuracy': accuracy,
        'num_correct': num_correct,
        'num_total': len(problems),
        'avg_reward': avg_reward,
        'details': details,
    }
