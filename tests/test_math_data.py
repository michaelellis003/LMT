"""Tests for math dataset loading for GRPO training.

Math datasets provide (prompt, ground_truth_answer) pairs for
RLVR training. The loader handles common formats (GSM8K, MATH)
and extracts clean prompts and answers.
"""

import pytest

from lmt.data.math_data import (
    MathDataItem,
    extract_boxed_answer,
    extract_gsm8k_answer,
    format_math_prompt,
    load_math_items,
)


class TestExtractGsm8kAnswer:
    """Tests for GSM8K answer extraction."""

    def test_simple_number(self):
        """Extract a plain number after ####."""
        assert extract_gsm8k_answer('blah blah #### 42') == '42'

    def test_number_with_commas(self):
        """Remove commas from numbers."""
        assert extract_gsm8k_answer('#### 1,234') == '1234'

    def test_whitespace_around_answer(self):
        """Strip whitespace around the answer."""
        assert extract_gsm8k_answer('####   56  ') == '56'

    def test_no_answer_marker(self):
        """Return None when #### is missing."""
        assert extract_gsm8k_answer('no answer here') is None

    def test_negative_number(self):
        """Handle negative numbers."""
        assert extract_gsm8k_answer('#### -7') == '-7'

    def test_decimal_number(self):
        """Handle decimal numbers."""
        assert extract_gsm8k_answer('#### 3.14') == '3.14'

    def test_multiline_solution(self):
        """Find #### in multiline text."""
        solution = 'Step 1\nStep 2\n#### 3'
        assert extract_gsm8k_answer(solution) == '3'


class TestExtractBoxedAnswer:
    r"""Tests for ``\boxed{}`` answer extraction."""

    def test_simple_boxed(self):
        r"""Extract simple ``\boxed{42}``."""
        assert extract_boxed_answer(r'The answer is \boxed{42}') == '42'

    def test_nested_braces(self):
        r"""Handle nested braces like ``\frac{1}{2}``."""
        text = r'The answer is \boxed{\frac{1}{2}}'
        assert extract_boxed_answer(text) == r'\frac{1}{2}'

    def test_no_boxed(self):
        """Return None when no boxed answer found."""
        assert extract_boxed_answer('no boxed answer') is None

    def test_deeply_nested(self):
        r"""Handle deeply nested braces like ``\sqrt{\frac{a}{b}}``."""
        text = r'\boxed{\sqrt{\frac{a}{b}}}'
        assert extract_boxed_answer(text) == r'\sqrt{\frac{a}{b}}'

    def test_last_boxed_wins(self):
        """If multiple boxed, take the last one."""
        text = r'First \boxed{1}, then \boxed{2}'
        assert extract_boxed_answer(text) == '2'

    def test_empty_boxed(self):
        r"""Handle empty ``\boxed{}``."""
        assert extract_boxed_answer(r'\boxed{}') == ''


class TestFormatMathPrompt:
    """Tests for math prompt formatting."""

    def test_contains_problem(self):
        """Problem text should appear in the formatted prompt."""
        prompt = format_math_prompt('What is 2+2?')
        assert 'What is 2+2?' in prompt

    def test_contains_boxed_instruction(self):
        r"""Prompt should instruct using ``\boxed{}`` notation."""
        prompt = format_math_prompt('Solve x=1')
        assert '\\boxed{}' in prompt

    def test_ends_with_solution(self):
        """Prompt should end with 'Solution:' marker."""
        prompt = format_math_prompt('Any problem')
        assert prompt.endswith('Solution:')


class TestLoadMathItems:
    """Tests for loading math items from dict rows."""

    def test_gsm8k_format(self):
        """GSM8K rows have 'question' and 'answer' fields."""
        rows = [
            {
                'question': 'What is 2+2?',
                'answer': 'Two plus two is four. #### 4',
            },
            {
                'question': 'What is 3*5?',
                'answer': 'Three times five. #### 15',
            },
        ]
        items = load_math_items(rows, dataset_format='gsm8k')
        assert len(items) == 2
        assert items[0].prompt == 'What is 2+2?'
        assert items[0].ground_truth == '4'
        assert items[1].ground_truth == '15'

    def test_gsm8k_skips_bad_answers(self):
        """Rows without valid #### answer markers are skipped."""
        rows = [
            {
                'question': 'Good question',
                'answer': 'Good answer #### 42',
            },
            {
                'question': 'Bad question',
                'answer': 'No answer marker here',
            },
        ]
        items = load_math_items(rows, dataset_format='gsm8k')
        assert len(items) == 1
        assert items[0].ground_truth == '42'

    def test_math_format(self):
        r"""MATH dataset rows have 'problem' and 'solution' fields."""
        rows = [
            {
                'problem': 'Find x if x^2 = 9',
                'solution': r'x = \boxed{3}',
            },
        ]
        items = load_math_items(rows, dataset_format='math')
        assert len(items) == 1
        assert items[0].prompt == 'Find x if x^2 = 9'
        assert items[0].ground_truth == '3'

    def test_math_format_nested_braces(self):
        r"""MATH answers can have nested braces like ``\frac{1}{2}``."""
        rows = [
            {
                'problem': 'What is 1/2?',
                'solution': r'The answer is \boxed{\frac{1}{2}}',
            },
        ]
        items = load_math_items(rows, dataset_format='math')
        assert len(items) == 1
        assert items[0].ground_truth == r'\frac{1}{2}'

    def test_unknown_format_raises(self):
        """Raise ValueError for unrecognized format."""
        with pytest.raises(ValueError, match='Unknown dataset format'):
            load_math_items([], dataset_format='unknown')

    def test_empty_rows(self):
        """Empty input returns empty list."""
        items = load_math_items([], dataset_format='gsm8k')
        assert items == []

    def test_items_are_math_data_items(self):
        """Items should be MathDataItem instances."""
        rows = [{'question': 'Q', 'answer': '#### 1'}]
        items = load_math_items(rows, dataset_format='gsm8k')
        assert isinstance(items[0], MathDataItem)

    def test_format_prompts_flag(self):
        """With format_prompts=True, prompts get instruction wrapping."""
        rows = [
            {
                'question': 'What is 1+1?',
                'answer': '#### 2',
            },
        ]
        items = load_math_items(
            rows, dataset_format='gsm8k', format_prompts=True
        )
        assert '\\boxed{}' in items[0].prompt
        assert 'What is 1+1?' in items[0].prompt

    def test_format_prompts_default_false(self):
        """By default, prompts are raw (not formatted)."""
        rows = [{'question': 'Q?', 'answer': '#### 1'}]
        items = load_math_items(rows, dataset_format='gsm8k')
        assert items[0].prompt == 'Q?'

    def test_reward_integration(self):
        """Loaded items work with the math_reward function."""
        from lmt.training.rewards import math_reward

        rows = [
            {
                'question': 'What is 2+2?',
                'answer': '#### 4',
            },
        ]
        items = load_math_items(rows, dataset_format='gsm8k')
        reward = math_reward(
            prompt=items[0].prompt,
            response=r'The answer is \boxed{4}',
            ground_truth=items[0].ground_truth,
        )
        assert reward == 1.0
