"""Math dataset utilities for GRPO training.

Provides loading and formatting utilities for math datasets used
in RLVR (Reinforcement Learning with Verifiable Rewards). Supports
GSM8K and MATH dataset formats.

The workflow:
1. Load dataset → list of ``MathDataItem`` (prompt + ground_truth)
2. Format prompts for the model
3. Use ``math_reward`` from ``lmt.training.rewards`` to score responses

References:
    Cobbe et al. (2021). "Training Verifiers to Solve Math Word
    Problems" (GSM8K dataset).
    Hendrycks et al. (2021). "Measuring Mathematical Problem Solving
    with the MATH Dataset."
"""

import re
from dataclasses import dataclass


@dataclass
class MathDataItem:
    """A single math problem with its ground truth answer.

    Attributes:
        prompt: The math problem text.
        ground_truth: The correct answer string.
    """

    prompt: str
    ground_truth: str


def format_math_prompt(problem: str) -> str:
    """Format a math problem as a prompt for the model.

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
