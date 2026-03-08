"""Tests for math dataset loading for GRPO training.

Math datasets provide (prompt, ground_truth_answer) pairs for
RLVR training. The loader handles common formats (GSM8K, MATH)
and extracts clean prompts and answers.
"""


class TestMathDataset:
    """Test math dataset loading and formatting."""

    def test_format_gsm8k_prompt(self):
        """Format a GSM8K-style problem for GRPO training."""
        from lmt.data.math_data import format_math_prompt

        problem = 'If John has 5 apples and gives 2 away, how many?'
        prompt = format_math_prompt(problem)

        # Should wrap in instruction format
        assert problem in prompt
        assert len(prompt) > len(problem)

    def test_extract_gsm8k_answer(self):
        """Extract answer from GSM8K format (#### delimiter)."""
        from lmt.data.math_data import extract_gsm8k_answer

        solution = (
            'John starts with 5 apples.\nHe gives away 2.\n5 - 2 = 3\n#### 3'
        )
        assert extract_gsm8k_answer(solution) == '3'

    def test_extract_gsm8k_answer_with_comma(self):
        """Handle numbers with commas in GSM8K answers."""
        from lmt.data.math_data import extract_gsm8k_answer

        solution = 'Total cost is $1,234\n#### 1,234'
        assert extract_gsm8k_answer(solution) == '1234'

    def test_extract_gsm8k_answer_missing(self):
        """Return None when no #### delimiter found."""
        from lmt.data.math_data import extract_gsm8k_answer

        assert extract_gsm8k_answer('No answer here') is None

    def test_math_dataset_item_structure(self):
        """Each dataset item should have prompt and ground_truth."""
        from lmt.data.math_data import MathDataItem

        item = MathDataItem(
            prompt='What is 2+2?',
            ground_truth='4',
        )
        assert item.prompt == 'What is 2+2?'
        assert item.ground_truth == '4'

    def test_create_grpo_reward_fn(self):
        """Create a reward function from a math data item."""
        from lmt.data.math_data import MathDataItem

        item = MathDataItem(
            prompt='What is 2+2?',
            ground_truth='4',
        )

        # The reward fn should work with the math_reward function
        from lmt.training.rewards import math_reward

        reward = math_reward(
            prompt=item.prompt,
            response=r'The answer is \boxed{4}',
            ground_truth=item.ground_truth,
        )
        assert reward == 1.0
