"""Tests for sandboxed code execution with structured results.

Run with: uv run pytest tests/test_code_execution.py -v
"""

from lmt.eval.code_execution import (
    ExecutionResult,
    SingleTestResult,
    execute_code,
    execute_with_tests,
)


class TestExecutionResult:
    """Tests for the ExecutionResult dataclass."""

    def test_basic_fields(self):
        """Fields and reward property work correctly."""
        result = ExecutionResult(
            stdout='hello',
            stderr='',
            passed=1,
            failed=0,
            total=1,
            timed_out=False,
        )
        assert result.stdout == 'hello'
        assert result.passed == 1
        assert result.reward == 1.0

    def test_reward_graded(self):
        """Reward is fraction of tests passed."""
        result = ExecutionResult(
            stdout='',
            stderr='',
            passed=2,
            failed=1,
            total=3,
            timed_out=False,
        )
        assert abs(result.reward - 2 / 3) < 1e-6

    def test_reward_zero_total(self):
        """Zero total tests gives zero reward."""
        result = ExecutionResult(
            stdout='',
            stderr='',
            passed=0,
            failed=0,
            total=0,
            timed_out=False,
        )
        assert result.reward == 0.0

    def test_all_passed_flag(self):
        """The all_passed property reflects pass/total."""
        r1 = ExecutionResult('', '', 3, 0, 3, False)
        r2 = ExecutionResult('', '', 2, 1, 3, False)
        assert r1.all_passed is True
        assert r2.all_passed is False

    def test_timed_out_gives_zero_reward(self):
        """Timeout always yields zero reward."""
        result = ExecutionResult('', '', 0, 0, 1, timed_out=True)
        assert result.reward == 0.0


class TestSingleTestResult:
    """Tests for the SingleTestResult dataclass."""

    def test_passed_test(self):
        """Passing test has no error."""
        tr = SingleTestResult(name='assert add(1, 2) == 3', passed=True)
        assert tr.passed is True
        assert tr.error is None

    def test_failed_test(self):
        """Failed test records error message."""
        tr = SingleTestResult(
            name='assert add(1, 2) == 3',
            passed=False,
            error='AssertionError',
        )
        assert tr.passed is False
        assert tr.error == 'AssertionError'


class TestExecuteCode:
    """Tests for raw code execution."""

    def test_simple_print(self):
        """Captures stdout from print statement."""
        result = execute_code('print("hello")')
        assert result.stdout.strip() == 'hello'
        assert result.timed_out is False

    def test_syntax_error(self):
        """Syntax errors produce stderr output."""
        result = execute_code('def foo(')
        assert result.stderr != ''
        assert result.passed == 0

    def test_runtime_error(self):
        """Runtime errors appear in stderr."""
        result = execute_code('1/0')
        assert 'ZeroDivisionError' in result.stderr

    def test_timeout(self):
        """Infinite loops are caught by timeout."""
        result = execute_code('while True: pass', timeout=1)
        assert result.timed_out is True

    def test_empty_code(self):
        """Empty code runs without error."""
        result = execute_code('')
        assert result.timed_out is False


class TestExecuteWithTests:
    """Tests for code execution with test assertions."""

    def test_all_tests_pass(self):
        """Correct code passes all assertions."""
        code = 'def add(a, b): return a + b'
        tests = 'assert add(1, 2) == 3\nassert add(0, 0) == 0'
        result = execute_with_tests(code, tests)
        assert result.passed == 2
        assert result.failed == 0
        assert result.total == 2
        assert result.all_passed is True
        assert result.reward == 1.0

    def test_partial_pass(self):
        """Buggy code fails all tests when off-by-one."""
        code = 'def add(a, b): return a + b + 1'
        tests = (
            'assert add(0, 0) == 0\n'
            'assert add(-1, 1) == 0\n'
            'assert add(0, -1) == -1'
        )
        result = execute_with_tests(code, tests)
        assert result.passed == 0
        assert result.failed == 3
        assert result.total == 3

    def test_all_tests_fail(self):
        """Completely wrong code gets zero reward."""
        code = 'def add(a, b): return 999'
        tests = 'assert add(1, 2) == 3\nassert add(0, 0) == 0'
        result = execute_with_tests(code, tests)
        assert result.passed == 0
        assert result.failed == 2
        assert result.reward == 0.0

    def test_syntax_error_in_code(self):
        """Syntax errors are caught before execution."""
        code = 'def add(a, b) return a + b'
        tests = 'assert add(1, 2) == 3'
        result = execute_with_tests(code, tests)
        assert result.reward == 0.0

    def test_test_results_populated(self):
        """Per-test results include pass/fail and error info."""
        code = 'def add(a, b): return a + b'
        tests = 'assert add(1, 2) == 3\nassert add(1, 1) == 99'
        result = execute_with_tests(code, tests)
        assert len(result.test_results) == 2
        assert result.test_results[0].passed is True
        assert result.test_results[1].passed is False
        assert result.test_results[1].error is not None

    def test_empty_tests(self):
        """Empty test string gives zero total and zero reward."""
        code = 'x = 1'
        result = execute_with_tests(code, '')
        assert result.total == 0
        assert result.reward == 0.0

    def test_comments_and_blank_lines_in_tests(self):
        """Comments and blank lines should be skipped."""
        code = 'def add(a, b): return a + b'
        tests = (
            '# This is a comment\n'
            '\n'
            'assert add(1, 2) == 3\n'
            '\n'
            '# Another comment\n'
            'assert add(0, 0) == 0\n'
        )
        result = execute_with_tests(code, tests)
        assert result.total == 2
        assert result.passed == 2

    def test_timeout_in_code(self):
        """Slow code is killed by timeout."""
        code = 'import time; time.sleep(10)'
        tests = 'assert True'
        result = execute_with_tests(code, tests, timeout=1)
        assert result.timed_out is True
        assert result.reward == 0.0

    def test_multiline_function(self):
        """Complex multi-line functions work correctly."""
        code = (
            'def fibonacci(n):\n'
            '    if n <= 1:\n'
            '        return n\n'
            '    return fibonacci(n - 1) + fibonacci(n - 2)\n'
        )
        tests = (
            'assert fibonacci(0) == 0\n'
            'assert fibonacci(1) == 1\n'
            'assert fibonacci(5) == 5\n'
            'assert fibonacci(10) == 55\n'
        )
        result = execute_with_tests(code, tests)
        assert result.all_passed is True
        assert result.passed == 4
