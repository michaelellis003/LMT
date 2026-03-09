"""Best-of-N inference-time scaling for code generation.

Generates N code samples from a model, executes each against test
assertions, and selects the sample with the highest reward. This is
the simplest and most effective inference-time scaling strategy for
code: the Codex paper (Chen et al., 2021) showed that pass@100 from
a 300M model can match pass@1 from a model 20x larger.

The key insight is that code has a free, perfect verifier — just run
the tests. No learned reward model needed.

Usage::

    from lmt.inference.best_of_n import best_of_n, BestOfNConfig

    config = BestOfNConfig(n_samples=50, temperature=0.8)
    result = best_of_n(model, prompt, tests, tokenizer, config)
    print(f'Best reward: {result.best_reward}')
    print(f'{result.n_correct}/{result.n_samples} samples correct')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from lmt.eval.code_execution import ExecutionResult, execute_with_tests
from lmt.recipes.code_reasoning import extract_code_block
from lmt.training.generation import GenerationConfig, generate_responses


@dataclass
class BestOfNConfig:
    """Configuration for best-of-N sampling.

    Attributes:
        n_samples: Number of candidate samples to generate.
        temperature: Sampling temperature (higher = more diverse).
        max_new_tokens: Maximum tokens to generate per sample.
        timeout: Maximum execution time per sample in seconds.
    """

    n_samples: int = 16
    temperature: float = 0.8
    max_new_tokens: int = 256
    timeout: int = 10


@dataclass
class BestOfNResult:
    """Result of best-of-N selection.

    Attributes:
        best_code: The selected code sample (highest reward).
        best_reward: Reward of the selected sample (0.0 to 1.0).
        all_codes: All generated code samples.
        all_rewards: Rewards for each sample.
        n_samples: Total number of samples generated.
        n_correct: Number of samples that passed all tests.
    """

    best_code: str
    best_reward: float
    all_codes: list[str]
    all_rewards: list[float]
    n_samples: int
    n_correct: int


def best_of_n_select(
    codes: list[str],
    execution_results: list[ExecutionResult],
) -> BestOfNResult:
    """Select the best code sample by execution reward.

    Given a list of code samples and their execution results, picks
    the sample with the highest reward. Ties are broken by shortest
    code length (Occam's razor heuristic — shorter correct code is
    usually simpler and more likely correct on unseen tests).

    Args:
        codes: Generated code samples.
        execution_results: Execution result for each sample.

    Returns:
        BestOfNResult with the selected sample and statistics.

    Raises:
        ValueError: If inputs are empty or mismatched.
    """
    if not codes:
        raise ValueError('best_of_n_select requires at least one sample.')
    if len(codes) != len(execution_results):
        raise ValueError(
            f'codes and execution_results must have same length, '
            f'got {len(codes)} and {len(execution_results)}.'
        )

    rewards = [r.reward for r in execution_results]

    # Find best: highest reward, then shortest code as tiebreak
    best_idx = 0
    for i in range(1, len(codes)):
        if rewards[i] > rewards[best_idx] or (
            rewards[i] == rewards[best_idx]
            and len(codes[i]) < len(codes[best_idx])
        ):
            best_idx = i

    n_correct = sum(1 for r in execution_results if r.all_passed)

    return BestOfNResult(
        best_code=codes[best_idx],
        best_reward=rewards[best_idx],
        all_codes=codes,
        all_rewards=rewards,
        n_samples=len(codes),
        n_correct=n_correct,
    )


def best_of_n(
    model: nn.Module,
    prompt: str,
    tests: str,
    tokenizer: Any,
    config: BestOfNConfig | None = None,
) -> BestOfNResult:
    """Generate N samples and select the best by execution reward.

    End-to-end pipeline: generates N completions from the model,
    extracts code blocks, executes each against the given tests,
    and returns the best sample.

    Args:
        model: Language model for generation.
        prompt: The coding problem prompt.
        tests: Test assertions (newline-separated).
        tokenizer: Tokenizer with ``encode`` and ``decode`` methods.
        config: Sampling configuration. Uses defaults if ``None``.

    Returns:
        BestOfNResult with the best sample and all candidates.
    """
    if config is None:
        config = BestOfNConfig()

    model.eval()
    device = next(model.parameters()).device

    # Generate N samples using the batch generation pipeline
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    gen_config = GenerationConfig(
        group_size=config.n_samples,
        max_response_len=config.max_new_tokens,
        temperature=config.temperature,
    )

    with torch.no_grad():
        sequences = generate_responses(model, prompt_tensor, gen_config)

    # Extract code from each sample
    codes = []
    for i in range(sequences.shape[0]):
        response_ids = sequences[i, len(prompt_ids) :].tolist()
        response_text = tokenizer.decode(response_ids)
        codes.append(extract_code_block(response_text))

    # Execute each against tests
    execution_results = [
        execute_with_tests(code, tests, timeout=config.timeout)
        for code in codes
    ]

    return best_of_n_select(codes, execution_results)
