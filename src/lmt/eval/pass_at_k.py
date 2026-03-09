r"""Unbiased pass@k estimator for code generation evaluation.

Implements the pass@k metric from Chen et al. (2021) "Evaluating Large
Language Models Trained on Code" (Codex paper). Given *n* total samples
with *c* correct, the unbiased estimator for the probability that at
least one of *k* random samples is correct is:

.. math::

    \text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}

This avoids the biased estimator ``1 - (1 - c/n)^k`` which
overestimates pass@k when c is small.

Usage::

    from lmt.eval.pass_at_k import pass_at_k, pass_at_k_problems

    # Single problem: 10 samples, 3 correct
    print(pass_at_k(n=10, c=3, k=1))  # 0.3

    # Average across problems
    results = [(10, 3), (10, 7), (10, 0)]  # (n, c) per problem
    print(pass_at_k_problems(results, k=1))  # 0.333...
"""

from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    r"""Compute pass@k for a single problem.

    Uses the unbiased estimator from Chen et al. (2021):

    .. math::

        \text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}

    For large *n*, uses log-space computation to avoid overflow.

    Args:
        n: Total number of samples generated.
        c: Number of correct samples.
        k: Number of samples to consider.

    Returns:
        Probability that at least one of k samples is correct.

    Raises:
        ValueError: If inputs are invalid (negative, k > n, c > n).
    """
    if n < 0 or c < 0 or k <= 0:
        raise ValueError(
            f'Invalid inputs: n={n}, c={c}, k={k}. '
            f'Require n >= 0, c >= 0, k > 0.'
        )
    if k > n:
        raise ValueError(f'k={k} cannot exceed n={n}.')
    if c > n:
        raise ValueError(f'c={c} cannot exceed n={n}.')

    # Edge cases
    if c == 0:
        return 0.0
    if c >= k and n - c < k:
        # C(n-c, k) = 0 when n-c < k, so pass@k = 1.0
        return 1.0

    # For large n, use log-space to avoid overflow
    if n > 500:
        return _pass_at_k_log(n, c, k)

    # Direct computation using math.comb
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_problems(
    results: list[tuple[int, int]],
    k: int,
) -> float:
    """Average pass@k across multiple problems.

    Args:
        results: List of ``(n, c)`` tuples — total samples and correct
            count per problem.
        k: Number of samples to consider.

    Returns:
        Mean pass@k across all problems. Returns 0.0 for empty list.
    """
    if not results:
        return 0.0
    return sum(pass_at_k(n, c, k) for n, c in results) / len(results)


def _pass_at_k_log(n: int, c: int, k: int) -> float:
    """Log-space pass@k for numerical stability with large n.

    Computes log(C(n-c, k)) - log(C(n, k)) to avoid overflow,
    then converts back: pass@k = 1 - exp(log_ratio).
    """
    # log(C(n-c, k)) - log(C(n, k))
    # = sum_{i=0}^{k-1} [log(n-c-i) - log(n-i)]
    log_ratio = 0.0
    for i in range(k):
        if n - c - i <= 0:
            # C(n-c, k) = 0, so pass@k = 1.0
            return 1.0
        log_ratio += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_ratio)
