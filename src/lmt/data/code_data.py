"""Code dataset utilities for GRPO training.

Provides loading and formatting utilities for coding problem datasets
used in RLVR (Reinforcement Learning with Verifiable Rewards). Supports
HumanEval and simple (prompt, tests) formats.

The workflow:
1. Load dataset rows (e.g. from HuggingFace ``datasets``)
2. Call ``load_code_items(rows, dataset_format='humaneval')``
3. Each item has ``.prompt``, ``.tests``, and optional
   ``.canonical_solution``
4. Convert to ``CodeProblem`` via ``.to_code_problem()``
5. Use ``code_reward`` from ``lmt.training.rewards`` to score

References:
    Chen et al. (2021). "Evaluating Large Language Models Trained on
    Code" (HumanEval benchmark).
    Austin et al. (2021). "Program Synthesis with Large Language
    Models" (MBPP benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lmt.recipes.code_reasoning import CodeProblem


@dataclass
class CodeDataItem:
    """A coding problem with test assertions and metadata.

    Attributes:
        prompt: The problem description or function signature.
        tests: Newline-separated test assertions.
        canonical_solution: Reference solution (if available).
        language: Programming language (default ``'python'``).
        entry_point: Function name to test (e.g. ``'add'``).
    """

    prompt: str
    tests: str
    canonical_solution: str | None = field(default=None)
    language: str = field(default='python')
    entry_point: str | None = field(default=None)

    def to_code_problem(self) -> CodeProblem:
        """Convert to a CodeProblem for reward function creation.

        Returns:
            CodeProblem with prompt, tests, and language.
        """
        return CodeProblem(
            prompt=self.prompt,
            tests=self.tests,
            language=self.language,
        )


def format_code_prompt(
    problem: str,
    entry_point: str | None = None,
    language: str = 'python',
) -> str:
    """Format a coding problem as a prompt for the model.

    Wraps the problem in an instruction format that encourages
    the model to write code in a markdown code block.

    Args:
        problem: The raw problem description or function signature.
        entry_point: Optional function name to implement.
        language: Programming language name.

    Returns:
        Formatted prompt string.
    """
    parts = [
        f'Write a {language} solution for the following problem.',
    ]
    if entry_point:
        parts.append(f'Implement the function `{entry_point}`.')
    parts.append('Put your code in a markdown code block.\n')
    parts.append(f'Problem: {problem}\n')
    parts.append('Solution:')
    return ' '.join(parts[:2]) + '\n' + '\n'.join(parts[2:])


def load_code_items(
    rows: list[dict[str, Any]] | Any,
    dataset_format: str = 'humaneval',
    format_prompts: bool = False,
) -> list[CodeDataItem]:
    """Load coding problems from dataset rows into CodeDataItems.

    Supports HumanEval and simple dataset formats. Rows with
    missing required fields are silently skipped.

    Args:
        rows: Iterable of dicts (e.g. HuggingFace dataset rows).
        dataset_format: ``'humaneval'`` or ``'simple'``.
        format_prompts: If True, wrap prompts in instruction format
            via ``format_code_prompt()``. Default False.

    Returns:
        List of CodeDataItem with prompt and tests.

    Raises:
        ValueError: If dataset_format is not recognized.
    """
    if dataset_format == 'humaneval':
        return _load_humaneval(rows, format_prompts)
    elif dataset_format == 'simple':
        return _load_simple(rows, format_prompts)
    else:
        msg = f"Unknown dataset format: '{dataset_format}'"
        raise ValueError(msg)


def _load_humaneval(
    rows: list[dict[str, Any]] | Any,
    format_prompts: bool,
) -> list[CodeDataItem]:
    """Load HumanEval format: prompt + test + canonical_solution."""
    items: list[CodeDataItem] = []
    for row in rows:
        prompt = row.get('prompt')
        tests = row.get('test')
        if prompt is None or tests is None:
            continue
        canonical = row.get('canonical_solution')
        entry_point = row.get('entry_point')

        if format_prompts:
            prompt = format_code_prompt(prompt, entry_point=entry_point)

        items.append(
            CodeDataItem(
                prompt=prompt,
                tests=tests,
                canonical_solution=canonical,
                entry_point=entry_point,
            )
        )
    return items


def _load_simple(
    rows: list[dict[str, Any]] | Any,
    format_prompts: bool,
) -> list[CodeDataItem]:
    """Load simple format: prompt + tests."""
    items: list[CodeDataItem] = []
    for row in rows:
        prompt = row.get('prompt')
        tests = row.get('tests')
        if prompt is None or tests is None:
            continue

        if format_prompts:
            prompt = format_code_prompt(prompt)

        items.append(
            CodeDataItem(
                prompt=prompt,
                tests=tests,
            )
        )
    return items
