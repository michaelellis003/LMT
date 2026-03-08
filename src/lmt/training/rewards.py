r"""Reward functions for Reinforcement Learning with Verifiable Rewards.

Provides deterministic reward signals for GRPO training. Unlike
learned reward models, these use ground-truth verification (math
answer checking, compiler output, format validation) for reliable,
reproducible training signals.

Three reward types:

1. **Math reward**: Binary (0/1) based on exact answer matching.
   Extracts answers from ``\boxed{}`` or final number.

2. **Code reward**: Graded (0.0–1.0) based on compilation and
   test passage. Partial credit for code that compiles but fails
   some tests.

3. **Format reward**: Checks for required structural elements
   (e.g., ``<think>`` tags for chain-of-thought).

References:
    Raschka, S. (2025). "Reasoning from Scratch" -- binary math
    reward for GRPO training on Qwen3-0.6B.
"""

import re
import subprocess


def extract_math_answer(text: str) -> str | None:
    r"""Extract a math answer from model output.

    Looks for ``\boxed{...}`` first, then falls back to the last
    number in the text.

    Args:
        text: Model response text.

    Returns:
        Extracted answer string, or None if no answer found.
    """
    # Try \boxed{} first (LaTeX convention)
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()

    # Fall back to last number (integer or decimal)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]

    return None


def _normalize_number(s: str) -> float | None:
    """Try to parse a string as a float for comparison."""
    try:
        return float(s)
    except ValueError:
        return None


def math_reward(
    prompt: str,
    response: str,
    ground_truth: str,
) -> float:
    r"""Binary reward based on math answer correctness.

    Extracts the answer from the response and compares it to the
    ground truth. Returns 1.0 for correct, 0.0 for incorrect.

    Args:
        prompt: The math problem (unused but kept for API consistency).
        response: The model's response text.
        ground_truth: The correct answer string.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    extracted = extract_math_answer(response)
    if extracted is None:
        return 0.0

    # Exact string match
    if extracted.strip() == ground_truth.strip():
        return 1.0

    # Numeric equivalence
    ext_num = _normalize_number(extracted)
    gt_num = _normalize_number(ground_truth)
    if (
        ext_num is not None
        and gt_num is not None
        and abs(ext_num - gt_num) < 1e-6
    ):
        return 1.0

    return 0.0


def code_reward(
    code: str,
    tests: str,
    language: str = 'python',
    timeout: int = 10,
) -> float:
    """Reward based on code compilation and test results.

    Executes code in a subprocess with the given tests. Returns
    graded reward: 0.0 for syntax/runtime errors, 0.3 for code
    that runs but fails tests, 1.0 for all tests passing.

    Args:
        code: The code solution.
        tests: Test assertions to run after the code.
        language: Programming language (currently ``'python'``).
        timeout: Maximum execution time in seconds.

    Returns:
        Float reward between 0.0 and 1.0.
    """
    if not code.strip():
        return 0.0

    if language != 'python':
        raise NotImplementedError(f'Language {language!r} not yet supported')

    # Combine code and tests into a single script
    full_script = f'{code}\n\n{tests}'

    try:
        result = subprocess.run(
            ['python', '-c', full_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return 1.0

        # Check if it's a syntax error vs runtime error
        stderr = result.stderr
        if 'SyntaxError' in stderr:
            return 0.0

        # Code ran but tests failed -> partial credit
        return 0.3

    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0


def format_reward(
    response: str,
    required_tags: list[str],
) -> float:
    """Reward based on response format compliance.

    Checks for required XML-style tags in the response. Returns
    the fraction of required tags that are present.

    Args:
        response: The model's response text.
        required_tags: List of tag names to check for (e.g.,
            ``['think', 'solution']``). Checks for both opening
            ``<tag>`` and closing ``</tag>``.

    Returns:
        Float between 0.0 and 1.0 (fraction of tags present).
    """
    if not required_tags:
        return 1.0

    found = 0
    for tag in required_tags:
        opening = f'<{tag}>'
        closing = f'</{tag}>'
        if opening in response and closing in response:
            found += 1

    return found / len(required_tags)
