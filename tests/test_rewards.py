"""Tests for reward functions used in RLVR (GRPO training).

Reward functions score model responses for GRPO training. They must
return float rewards in a predictable range and handle edge cases.
"""


class TestMathReward:
    """Test the math answer extraction and reward function."""

    def test_extract_boxed_answer(self):
        r"""Extract answer from LaTeX \boxed{} notation."""
        from lmt.training.rewards import extract_math_answer

        assert extract_math_answer(r'The answer is \boxed{42}') == '42'
        assert extract_math_answer(r'\boxed{x^2 + 1}') == 'x^2 + 1'
        assert extract_math_answer(r'So \boxed{3/4}.') == '3/4'

    def test_extract_nested_braces(self):
        r"""Handle nested braces like \boxed{\frac{1}{2}}."""
        from lmt.training.rewards import extract_math_answer

        assert extract_math_answer(r'\boxed{\frac{1}{2}}') == r'\frac{1}{2}'
        assert extract_math_answer(r'\boxed{x^{2}+1}') == 'x^{2}+1'

    def test_extract_final_number(self):
        r"""Extract last number when no \boxed{} is present."""
        from lmt.training.rewards import extract_math_answer

        assert extract_math_answer('The answer is 42.') == '42'
        assert extract_math_answer('Result: -3.14') == '-3.14'

    def test_extract_decimal_without_leading_digit(self):
        """Extract .5 style numbers (no leading digit)."""
        from lmt.training.rewards import extract_math_answer

        assert extract_math_answer('The probability is .5') == '.5'
        assert extract_math_answer('It equals .75 exactly') == '.75'

    def test_extract_returns_none_for_no_answer(self):
        """Return None when no answer can be extracted."""
        from lmt.training.rewards import extract_math_answer

        assert extract_math_answer('I cannot solve this') is None
        assert extract_math_answer('') is None

    def test_math_reward_correct(self):
        """Correct answer gets reward 1.0."""
        from lmt.training.rewards import math_reward

        reward = math_reward(
            prompt='What is 2+2?',
            response=r'Let me think... \boxed{4}',
            ground_truth='4',
        )
        assert reward == 1.0

    def test_math_reward_incorrect(self):
        """Wrong answer gets reward 0.0."""
        from lmt.training.rewards import math_reward

        reward = math_reward(
            prompt='What is 2+2?',
            response=r'\boxed{5}',
            ground_truth='4',
        )
        assert reward == 0.0

    def test_math_reward_no_answer(self):
        """No parseable answer gets reward 0.0."""
        from lmt.training.rewards import math_reward

        reward = math_reward(
            prompt='What is 2+2?',
            response='I am not sure about this question.',
            ground_truth='4',
        )
        assert reward == 0.0

    def test_math_reward_numeric_equivalence(self):
        """Numerically equivalent answers should match."""
        from lmt.training.rewards import math_reward

        # 0.5 == 1/2
        assert math_reward('', r'\boxed{0.5}', '0.5') == 1.0
        # Integer with trailing zero
        assert math_reward('', r'\boxed{42}', '42.0') == 1.0


class TestCodeReward:
    """Test the code compilation/execution reward function."""

    def test_code_reward_correct_python(self):
        """Python code that passes tests gets reward 1.0."""
        from lmt.training.rewards import code_reward

        code = 'def add(a, b):\n    return a + b'
        tests = 'assert add(2, 3) == 5\nassert add(-1, 1) == 0'

        reward = code_reward(code=code, tests=tests, language='python')
        assert reward == 1.0

    def test_code_reward_syntax_error(self):
        """Code with syntax error gets partial reward."""
        from lmt.training.rewards import code_reward

        code = 'def add(a, b)\n    return a + b'  # missing colon
        tests = 'assert add(2, 3) == 5'

        reward = code_reward(code=code, tests=tests, language='python')
        assert reward == 0.0

    def test_code_reward_wrong_output(self):
        """Code that compiles but fails tests gets partial reward."""
        from lmt.training.rewards import code_reward

        code = 'def add(a, b):\n    return a - b'  # wrong operation
        tests = 'assert add(2, 3) == 5'

        reward = code_reward(code=code, tests=tests, language='python')
        assert 0.0 < reward < 1.0  # partial credit for compiling

    def test_code_reward_timeout(self):
        """Infinite loop should timeout and get 0.0."""
        from lmt.training.rewards import code_reward

        code = 'def solve():\n    while True: pass'
        tests = 'assert solve() is None'

        reward = code_reward(
            code=code, tests=tests, language='python', timeout=2
        )
        assert reward == 0.0

    def test_code_reward_empty(self):
        """Empty code gets 0.0."""
        from lmt.training.rewards import code_reward

        reward = code_reward(code='', tests='pass', language='python')
        assert reward == 0.0


class TestFormatReward:
    """Test the format checking reward function."""

    def test_has_think_tags(self):
        """Response with think tags gets format reward."""
        from lmt.training.rewards import format_reward

        response = '<think>\nLet me reason...\n</think>\nAnswer: 42'
        reward = format_reward(response, required_tags=['think'])
        assert reward == 1.0

    def test_missing_tags(self):
        """Response missing required tags gets 0.0."""
        from lmt.training.rewards import format_reward

        response = 'Answer: 42'
        reward = format_reward(response, required_tags=['think'])
        assert reward == 0.0

    def test_partial_tags(self):
        """Response with some but not all tags gets partial credit."""
        from lmt.training.rewards import format_reward

        response = '<think>\nReasoning\n</think>\nAnswer: 42'
        reward = format_reward(response, required_tags=['think', 'solution'])
        assert 0.0 < reward < 1.0
