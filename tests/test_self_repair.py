"""Tests for self-repair inference-time scaling.

Self-repair retries failed code generation with structured error
feedback: the error type, failing test, and expected vs actual output
are fed back to the model as context for the retry.

Run with: uv run pytest tests/test_self_repair.py -v
"""

from lmt.eval.code_execution import ExecutionResult, SingleTestResult
from lmt.inference.self_repair import (
    RepairConfig,
    RepairResult,
    build_error_prompt,
)


class TestRepairConfig:
    """Tests for RepairConfig defaults."""

    def test_defaults(self):
        """Default config has sensible values."""
        config = RepairConfig()
        assert config.max_retries >= 1
        assert config.temperature > 0.0


class TestBuildErrorPrompt:
    """Tests for structured error prompt construction."""

    def test_includes_original_prompt(self):
        """Error prompt includes the original problem."""
        prompt = 'Write a function add(a, b) that returns a + b.'
        code = 'def add(a, b): return a - b'
        execution = _make_failed_result(
            'assert add(1, 2) == 3',
            'AssertionError: assert -1 == 3',
        )
        error_prompt = build_error_prompt(prompt, code, execution)
        assert 'Write a function' in error_prompt

    def test_includes_previous_code(self):
        """Error prompt includes the failed code."""
        prompt = 'Write add.'
        code = 'def add(a, b): return a - b'
        execution = _make_failed_result(
            'assert add(1, 2) == 3',
            'AssertionError',
        )
        error_prompt = build_error_prompt(prompt, code, execution)
        assert 'def add(a, b): return a - b' in error_prompt

    def test_includes_error_info(self):
        """Error prompt includes the error type and failing test."""
        prompt = 'Write add.'
        code = 'def add(a, b): return a - b'
        execution = _make_failed_result(
            'assert add(1, 2) == 3',
            'AssertionError: assert -1 == 3',
        )
        error_prompt = build_error_prompt(prompt, code, execution)
        assert 'AssertionError' in error_prompt
        assert 'add(1, 2)' in error_prompt

    def test_multiple_failures(self):
        """Error prompt includes all failing tests."""
        prompt = 'Write add.'
        code = 'def add(a, b): return 0'
        execution = ExecutionResult(
            stdout='',
            stderr='',
            passed=0,
            failed=2,
            total=2,
            timed_out=False,
            test_results=[
                SingleTestResult(
                    name='assert add(1, 2) == 3',
                    passed=False,
                    error='AssertionError',
                ),
                SingleTestResult(
                    name='assert add(0, 0) == 0',
                    passed=True,
                    error=None,
                ),
            ],
        )
        error_prompt = build_error_prompt(prompt, code, execution)
        assert 'add(1, 2)' in error_prompt

    def test_syntax_error(self):
        """Error prompt handles syntax errors."""
        prompt = 'Write add.'
        code = 'def add(a, b) return a + b'
        execution = ExecutionResult(
            stdout='',
            stderr='SyntaxError: invalid syntax',
            passed=0,
            failed=1,
            total=1,
            timed_out=False,
            test_results=[
                SingleTestResult(
                    name='assert add(1, 2) == 3',
                    passed=False,
                    error='SyntaxError',
                ),
            ],
        )
        error_prompt = build_error_prompt(prompt, code, execution)
        assert 'SyntaxError' in error_prompt

    def test_timeout_error(self):
        """Error prompt handles timeout."""
        prompt = 'Write add.'
        code = 'def add(a, b):\n    while True: pass'
        execution = ExecutionResult(
            stdout='',
            stderr='TimeoutExpired',
            passed=0,
            failed=1,
            total=1,
            timed_out=True,
            test_results=[],
        )
        error_prompt = build_error_prompt(prompt, code, execution)
        assert 'timed out' in error_prompt.lower()


class TestRepairResult:
    """Tests for RepairResult dataclass."""

    def test_successful_repair(self):
        """Successful repair stores correct final code."""
        result = RepairResult(
            final_code='def add(a, b): return a + b',
            final_reward=1.0,
            attempts=2,
            succeeded=True,
        )
        assert result.succeeded is True
        assert result.attempts == 2

    def test_failed_repair(self):
        """Failed repair stores best attempt."""
        result = RepairResult(
            final_code='def add(a, b): return 0',
            final_reward=0.5,
            attempts=3,
            succeeded=False,
        )
        assert result.succeeded is False
        assert result.final_reward == 0.5


def _make_failed_result(test_name: str, error: str) -> ExecutionResult:
    """Helper to create a single-failure ExecutionResult."""
    return ExecutionResult(
        stdout='',
        stderr='',
        passed=0,
        failed=1,
        total=1,
        timed_out=False,
        test_results=[
            SingleTestResult(
                name=test_name,
                passed=False,
                error=error,
            ),
        ],
    )
