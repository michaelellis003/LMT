"""CodeT-style dual execution scoring for code generation.

Implements the key insight from Chen et al. (2022) "CodeT: Code
Generation with Generated Tests": score code candidates by their
agreement with independently generated tests.

The algorithm:
1. Given N code candidates and M generated tests
2. Build an N×M agreement matrix (pass/fail for each pair)
3. For each test, determine the majority vote (pass or fail)
4. A code's score = fraction of tests where it agrees with majority

This works because:
- Correct code tends to pass correct tests (true positive)
- Correct code tends to fail incorrect tests (true negative by majority)
- Wrong code disagrees with the majority on correct tests

The scoring function (``codet_score``) is separated from generation
so it can be used with any code/test generation strategy.

Usage::

    from lmt.inference.codet import codet_score

    codes = ['def f(x): return x+1', 'def f(x): return x*2']
    tests = ['assert f(1) == 2', 'assert f(0) == 1']
    result = codet_score(codes, tests)
    print(result.best_code)  # 'def f(x): return x+1'
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class CodeTResult:
    """Result of CodeT dual execution scoring.

    Attributes:
        best_code: The top-scoring code candidate.
        best_score: Agreement score of the best candidate (0.0-1.0).
        scores: Per-candidate agreement scores.
        agreement_matrix: N×M matrix of pass/fail results.
        n_candidates: Number of code candidates.
        n_tests: Number of generated tests.
        all_codes: All code candidates.
    """

    best_code: str
    best_score: float
    scores: list[float]
    agreement_matrix: list[list[bool]]
    n_candidates: int
    n_tests: int
    all_codes: list[str] = field(default_factory=list)


def codet_score(
    codes: list[str],
    gen_tests: list[str],
    timeout: int = 5,
) -> CodeTResult:
    """Score code candidates by agreement with generated tests.

    Builds an agreement matrix by executing each (code, test) pair,
    then scores each code by how often it agrees with the majority
    vote on each test.

    Args:
        codes: Code candidates to evaluate.
        gen_tests: Generated test assertions to score against.
        timeout: Max execution time per (code, test) pair in seconds.

    Returns:
        CodeTResult with scores and selected best code.

    Raises:
        ValueError: If codes list is empty.
    """
    if not codes:
        raise ValueError('codes list must not be empty')

    n = len(codes)
    m = len(gen_tests)

    # Edge case: no tests → all codes tied, pick first
    if m == 0:
        return CodeTResult(
            best_code=codes[0],
            best_score=1.0,
            scores=[1.0] * n,
            agreement_matrix=[[] for _ in range(n)],
            n_candidates=n,
            n_tests=0,
            all_codes=list(codes),
        )

    # Build N×M agreement matrix
    matrix = _build_agreement_matrix(codes, gen_tests, timeout)

    # For each test, compute majority vote
    majority = _compute_majority(matrix, n, m)

    # Score each code by agreement with majority
    scores = _compute_scores(matrix, majority, n, m)

    # Select best (highest score, tiebreak by shortest code)
    best_idx = 0
    for i in range(1, n):
        if scores[i] > scores[best_idx] or (
            scores[i] == scores[best_idx]
            and len(codes[i]) < len(codes[best_idx])
        ):
            best_idx = i

    return CodeTResult(
        best_code=codes[best_idx],
        best_score=scores[best_idx],
        scores=scores,
        agreement_matrix=matrix,
        n_candidates=n,
        n_tests=m,
        all_codes=list(codes),
    )


def _build_agreement_matrix(
    codes: list[str],
    tests: list[str],
    timeout: int,
) -> list[list[bool]]:
    """Execute each (code, test) pair and record pass/fail."""
    matrix = []
    for code in codes:
        row = []
        for test in tests:
            passed = _execute_pair(code, test, timeout)
            row.append(passed)
        matrix.append(row)
    return matrix


def _execute_pair(code: str, test: str, timeout: int) -> bool:
    """Execute a single (code, test) pair and return pass/fail."""
    script = f'{code}\n{test}'
    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _compute_majority(
    matrix: list[list[bool]],
    n: int,
    m: int,
) -> list[bool]:
    """For each test, determine if majority of codes pass it."""
    majority = []
    for j in range(m):
        pass_count = sum(matrix[i][j] for i in range(n))
        majority.append(pass_count > n / 2)
    return majority


def _compute_scores(
    matrix: list[list[bool]],
    majority: list[bool],
    n: int,
    m: int,
) -> list[float]:
    """Score each code by fraction of tests agreeing with majority."""
    scores = []
    for i in range(n):
        agreements = sum(1 for j in range(m) if matrix[i][j] == majority[j])
        scores.append(agreements / m)
    return scores
