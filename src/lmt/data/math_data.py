"""Math dataset utilities for GRPO training.

Provides loading and formatting utilities for math datasets used
in RLVR (Reinforcement Learning with Verifiable Rewards). Supports
GSM8K and MATH dataset formats.

The workflow:
1. Load dataset rows (e.g. from HuggingFace ``datasets``)
2. Call ``load_math_items(rows, dataset_format='gsm8k')``
3. Each item has ``.prompt`` and ``.ground_truth``
4. Use ``math_reward`` from ``lmt.training.rewards`` to score responses

References:
    Cobbe et al. (2021). "Training Verifiers to Solve Math Word
    Problems" (GSM8K dataset).
    Hendrycks et al. (2021). "Measuring Mathematical Problem Solving
    with the MATH Dataset."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class MathDataItem:
    """A single math problem with its ground truth answer.

    Attributes:
        prompt: The math problem text (raw or formatted).
        ground_truth: The correct answer string.
    """

    prompt: str
    ground_truth: str


def format_math_prompt(problem: str) -> str:
    r"""Format a math problem as a prompt for the model.

    Wraps the problem in an instruction format that encourages
    the model to show reasoning and provide a final answer in
    ``\\boxed{}`` notation.

    Args:
        problem: The raw math problem text.

    Returns:
        Formatted prompt string.
    """
    return (
        'Solve the following math problem. '
        'Show your reasoning step by step, '
        'then give your final answer in \\boxed{}.\n\n'
        f'Problem: {problem}\n\n'
        'Solution:'
    )


def extract_gsm8k_answer(solution: str) -> str | None:
    """Extract the answer from GSM8K format.

    GSM8K solutions end with ``#### <answer>`` where the answer
    is a number (possibly with commas for thousands).

    Args:
        solution: The full GSM8K solution string.

    Returns:
        The extracted answer with commas removed, or None.
    """
    match = re.search(r'####\s*(.+)', solution)
    if match is None:
        return None

    answer = match.group(1).strip()
    # Remove commas from numbers (e.g., "1,234" -> "1234")
    answer = answer.replace(',', '')
    return answer


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract the last ``\boxed{...}`` content from text.

    Handles nested braces correctly using a brace-matching parser.
    If multiple ``\boxed{}`` are present, returns the last one
    (typically the final answer in MATH solutions).

    Args:
        text: Text containing ``\boxed{answer}``.

    Returns:
        The content inside the last ``\boxed{}``, or None.
    """
    # Find all \boxed{ positions
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # Use the last match (final answer)
    last_match = matches[-1]
    start = last_match.end()  # position after \boxed{

    # Brace-matching parser to handle nesting
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1

    if depth != 0:
        return None

    return text[start : pos - 1]


def load_math_items(
    rows: list[dict[str, Any]] | Any,
    dataset_format: str = 'gsm8k',
    format_prompts: bool = False,
) -> list[MathDataItem]:
    """Load math problems from dataset rows into MathDataItems.

    Supports GSM8K and MATH dataset formats. Rows with invalid
    or missing answers are silently skipped.

    Args:
        rows: Iterable of dicts (e.g. HuggingFace dataset rows).
        dataset_format: ``'gsm8k'`` or ``'math'``.
        format_prompts: If True, wrap prompts in instruction format
            via ``format_math_prompt()``. Default False (raw prompts).

    Returns:
        List of MathDataItem with prompt and ground_truth.

    Raises:
        ValueError: If dataset_format is not recognized.
    """
    if dataset_format == 'gsm8k':
        return _load_gsm8k(rows, format_prompts)
    elif dataset_format == 'math':
        return _load_math(rows, format_prompts)
    else:
        msg = f"Unknown dataset format: '{dataset_format}'"
        raise ValueError(msg)


def _load_gsm8k(
    rows: list[dict[str, Any]] | Any,
    format_prompts: bool,
) -> list[MathDataItem]:
    """Load GSM8K format: 'question' + 'answer' with #### delimiter."""
    items: list[MathDataItem] = []
    for row in rows:
        question = row['question']
        answer_text = row['answer']
        ground_truth = extract_gsm8k_answer(answer_text)
        if ground_truth is None:
            continue
        prompt = format_math_prompt(question) if format_prompts else question
        items.append(MathDataItem(prompt=prompt, ground_truth=ground_truth))
    return items


def _load_math(
    rows: list[dict[str, Any]] | Any,
    format_prompts: bool,
) -> list[MathDataItem]:
    r"""Load MATH format: 'problem' + 'solution' with \boxed{}."""
    items: list[MathDataItem] = []
    for row in rows:
        problem = row['problem']
        solution = row['solution']
        ground_truth = extract_boxed_answer(solution)
        if ground_truth is None:
            continue
        prompt = format_math_prompt(problem) if format_prompts else problem
        items.append(MathDataItem(prompt=prompt, ground_truth=ground_truth))
    return items
