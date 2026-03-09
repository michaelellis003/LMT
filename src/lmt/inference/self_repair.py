"""Self-repair loop for code generation.

When generated code fails tests, constructs a structured error prompt
with the error type, failing test, and previous code, then retries
generation. Inspired by Self-Debugging (Chen et al., ICLR 2024) but
simplified for small models: instead of asking the model to interpret
raw tracebacks, we provide structured feedback that's easier to act on.

The key insight for small models: use execution feedback as structured
signal (error type + failing test) rather than raw tracebacks. This
reduces the "self-debug" to "informed re-sampling" — the model sees
what went wrong and gets another chance, without needing the language
understanding to parse Python tracebacks.

Usage::

    from lmt.inference.self_repair import self_repair, RepairConfig

    config = RepairConfig(max_retries=3)
    result = self_repair(model, prompt, tests, tokenizer, config)
    print(f'Succeeded: {result.succeeded} after {result.attempts} attempts')
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
class RepairConfig:
    """Configuration for self-repair loop.

    Attributes:
        max_retries: Maximum number of retry attempts after initial
            generation.
        temperature: Sampling temperature for generation.
        max_new_tokens: Maximum tokens per generation attempt.
        timeout: Maximum execution time per attempt in seconds.
    """

    max_retries: int = 3
    temperature: float = 0.7
    max_new_tokens: int = 256
    timeout: int = 10


@dataclass
class RepairResult:
    """Result of the self-repair loop.

    Attributes:
        final_code: The best code produced (last successful or best
            partial).
        final_reward: Reward of the final code.
        attempts: Total number of generation attempts (1 = no retries).
        succeeded: Whether all tests passed.
    """

    final_code: str
    final_reward: float
    attempts: int
    succeeded: bool


def build_error_prompt(
    original_prompt: str,
    code: str,
    execution: ExecutionResult,
) -> str:
    """Construct a structured error prompt for retry.

    Provides the model with actionable feedback: the original problem,
    the failed code, and what went wrong. Uses simple, structured
    format rather than raw tracebacks.

    Args:
        original_prompt: The original coding problem prompt.
        code: The code that failed.
        execution: Execution result with error details.

    Returns:
        Prompt string for the retry attempt.
    """
    parts = [original_prompt.rstrip(), '', '# Previous attempt:', '']
    parts.append(f'```python\n{code}\n```')
    parts.append('')

    if execution.timed_out:
        parts.append('# Error: Your code timed out (possible infinite loop).')
        parts.append('# Fix the infinite loop and try again.')
    else:
        # Collect failing tests
        failures = [tr for tr in execution.test_results if not tr.passed]
        if failures:
            parts.append(f'# Error: {len(failures)} test(s) failed:')
            for tr in failures[:3]:  # Limit to 3 to save context
                parts.append(f'#   FAIL: {tr.name}')
                if tr.error:
                    parts.append(f'#     {tr.error}')
        elif execution.stderr:
            parts.append(f'# Error: {execution.stderr[:200]}')
        else:
            parts.append('# Error: Code did not produce correct output.')

    parts.append('')
    parts.append('# Write the corrected code:')
    return '\n'.join(parts)


def self_repair(
    model: nn.Module,
    prompt: str,
    tests: str,
    tokenizer: Any,
    config: RepairConfig | None = None,
) -> RepairResult:
    """Generate code with self-repair on failure.

    Generates an initial solution, executes it against tests, and
    if it fails, constructs an error prompt and retries up to
    ``max_retries`` times.

    Args:
        model: Language model for generation.
        prompt: The coding problem prompt.
        tests: Test assertions (newline-separated).
        tokenizer: Tokenizer with ``encode`` and ``decode`` methods.
        config: Repair configuration. Uses defaults if ``None``.

    Returns:
        RepairResult with the best code found.
    """
    if config is None:
        config = RepairConfig()

    model.eval()
    device = next(model.parameters()).device

    best_code = ''
    best_reward = -1.0
    current_prompt = prompt

    for attempt in range(1 + config.max_retries):
        # Generate one sample
        code = _generate_one(model, current_prompt, tokenizer, device, config)
        execution = execute_with_tests(code, tests, timeout=config.timeout)

        if execution.reward > best_reward:
            best_code = code
            best_reward = execution.reward

        if execution.all_passed:
            return RepairResult(
                final_code=code,
                final_reward=1.0,
                attempts=attempt + 1,
                succeeded=True,
            )

        # Build error prompt for retry
        current_prompt = build_error_prompt(prompt, code, execution)

    return RepairResult(
        final_code=best_code,
        final_reward=best_reward,
        attempts=1 + config.max_retries,
        succeeded=False,
    )


def _generate_one(
    model: nn.Module,
    prompt: str,
    tokenizer: Any,
    device: torch.device,
    config: RepairConfig,
) -> str:
    """Generate a single code sample from a prompt."""
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    gen_config = GenerationConfig(
        group_size=1,
        max_response_len=config.max_new_tokens,
        temperature=config.temperature,
    )

    with torch.no_grad():
        sequences = generate_responses(model, prompt_tensor, gen_config)

    response_ids = sequences[0, len(prompt_ids) :].tolist()
    response_text = tokenizer.decode(response_ids)
    return extract_code_block(response_text)
