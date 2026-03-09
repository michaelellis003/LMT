r"""Sandboxed code execution with structured results.

Provides structured execution results for code generation evaluation,
including per-test pass/fail tracking with error messages. This enables
inference-time scaling strategies (best-of-N, self-repair) that need
to inspect *why* code failed, not just whether it passed.

Usage::

    from lmt.eval.code_execution import execute_with_tests

    result = execute_with_tests(
        code='def add(a, b): return a + b',
        tests='assert add(1, 2) == 3\\nassert add(0, 0) == 0',
    )
    print(f'{result.passed}/{result.total} tests passed')
    for tr in result.test_results:
        if not tr.passed:
            print(f'  FAIL: {tr.name} — {tr.error}')

.. warning::

    This module executes arbitrary code in a subprocess with no
    sandboxing beyond process isolation and timeouts. Only use with
    trusted inputs or inside an isolated container.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class SingleTestResult:
    """Result of a single test assertion.

    Attributes:
        name: The test assertion string (e.g., ``'assert add(1, 2) == 3'``).
        passed: Whether the test passed.
        error: Error type and message if the test failed, or ``None``.
    """

    name: str
    passed: bool
    error: str | None = None


@dataclass
class ExecutionResult:
    """Structured result of code execution against tests.

    Attributes:
        stdout: Captured standard output from execution.
        stderr: Captured standard error from execution.
        passed: Number of tests that passed.
        failed: Number of tests that failed.
        total: Total number of tests.
        timed_out: Whether execution hit the timeout limit.
        test_results: Per-test results with error details.
    """

    stdout: str
    stderr: str
    passed: int
    failed: int
    total: int
    timed_out: bool
    test_results: list[SingleTestResult] = field(default_factory=list)

    @property
    def reward(self) -> float:
        """Graded reward: fraction of tests passed (0.0 to 1.0)."""
        if self.timed_out or self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def all_passed(self) -> bool:
        """Whether all tests passed."""
        return self.total > 0 and self.passed == self.total


def execute_code(
    code: str,
    timeout: int = 10,
) -> ExecutionResult:
    """Execute Python code in a subprocess.

    Runs the code and captures stdout/stderr. Does not run any tests;
    use :func:`execute_with_tests` for test-based evaluation.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        ExecutionResult with stdout/stderr and timeout status.
    """
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            passed=0,
            failed=0,
            total=0,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            stdout='',
            stderr='TimeoutExpired',
            passed=0,
            failed=0,
            total=0,
            timed_out=True,
        )


def execute_with_tests(
    code: str,
    tests: str,
    timeout: int = 10,
) -> ExecutionResult:
    """Execute code and run test assertions with structured results.

    Each test assertion is wrapped individually so that failures in one
    test don't prevent others from running. Returns per-test results
    with error messages for use by self-repair strategies.

    Args:
        code: Python source code defining the function(s) to test.
        tests: Newline-separated test assertions
            (e.g., ``'assert add(1, 2) == 3'``). Comments and blank
            lines are skipped.
        timeout: Maximum execution time in seconds.

    Returns:
        ExecutionResult with per-test pass/fail details.
    """
    test_lines = _parse_test_lines(tests)
    if not test_lines:
        return ExecutionResult(
            stdout='',
            stderr='',
            passed=0,
            failed=0,
            total=0,
            timed_out=False,
        )

    # Check syntax before running
    try:
        compile(code, '<code>', 'exec')
    except SyntaxError as e:
        return ExecutionResult(
            stdout='',
            stderr=f'SyntaxError: {e}',
            passed=0,
            failed=len(test_lines),
            total=len(test_lines),
            timed_out=False,
            test_results=[
                SingleTestResult(name=t, passed=False, error='SyntaxError')
                for t in test_lines
            ],
        )

    runner_script = _build_runner_script(code, test_lines)

    try:
        result = subprocess.run(
            [sys.executable, '-c', runner_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            stdout='',
            stderr='TimeoutExpired',
            passed=0,
            failed=len(test_lines),
            total=len(test_lines),
            timed_out=True,
            test_results=[
                SingleTestResult(name=t, passed=False, error='TimeoutExpired')
                for t in test_lines
            ],
        )

    return _parse_results(result.stdout, result.stderr, test_lines)


def _parse_test_lines(tests: str) -> list[str]:
    """Extract test assertion lines, skipping comments and blanks."""
    return [
        line.strip()
        for line in tests.strip().splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]


def _build_runner_script(code: str, test_lines: list[str]) -> str:
    """Build a Python script that runs each test and reports JSON results."""
    # The script runs each test individually and prints JSON results
    script = f'{code}\n\nimport json as _json\n_results = []\n'
    for i, test in enumerate(test_lines):
        script += (
            f'try:\n'
            f'    {test}\n'
            f'    _results.append({{"i": {i}, "ok": True}})\n'
            f'except Exception as _e:\n'
            f'    _results.append({{"i": {i}, "ok": False, '
            f'"err": type(_e).__name__ + ": " + str(_e)}})\n'
        )
    script += 'print(_json.dumps(_results))\n'
    return script


def _parse_results(
    stdout: str,
    stderr: str,
    test_lines: list[str],
) -> ExecutionResult:
    """Parse JSON test results from runner script output."""
    # Try to find JSON in the last line of stdout
    json_line = ''
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith('['):
            json_line = line
            break

    if not json_line:
        # Script crashed before printing results
        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            passed=0,
            failed=len(test_lines),
            total=len(test_lines),
            timed_out=False,
            test_results=[
                SingleTestResult(
                    name=t,
                    passed=False,
                    error=stderr.strip()[:200] if stderr else 'Unknown error',
                )
                for t in test_lines
            ],
        )

    try:
        raw_results = json.loads(json_line)
    except json.JSONDecodeError:
        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            passed=0,
            failed=len(test_lines),
            total=len(test_lines),
            timed_out=False,
            test_results=[
                SingleTestResult(name=t, passed=False, error='Parse error')
                for t in test_lines
            ],
        )

    test_results = []
    passed = 0
    for entry in raw_results:
        idx = entry['i']
        name = test_lines[idx] if idx < len(test_lines) else f'test_{idx}'
        ok = entry['ok']
        error = entry.get('err')
        test_results.append(
            SingleTestResult(name=name, passed=ok, error=error)
        )
        if ok:
            passed += 1

    failed = len(test_lines) - passed

    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        passed=passed,
        failed=failed,
        total=len(test_lines),
        timed_out=False,
        test_results=test_results,
    )
