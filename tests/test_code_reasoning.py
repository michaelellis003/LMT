"""Tests for code reasoning reward bridge and utilities."""

import torch

from lmt.models.config import ModelConfig


def _tiny_config() -> ModelConfig:
    """Create a tiny model config for testing."""
    return ModelConfig(
        embed_dim=32,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        ffn_hidden_dim=64,
        vocab_size=64,
        context_length=32,
        tie_weights=True,
        dropout=0.0,
    )


class TestExtractCodeBlock:
    """Test extracting code from markdown-style responses."""

    def test_extract_python_code_block(self):
        """Should extract code from triple-backtick blocks."""
        from lmt.recipes.code_reasoning import extract_code_block

        text = 'Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```'
        code = extract_code_block(text)
        assert code == 'def add(a, b):\n    return a + b'

    def test_extract_plain_code_block(self):
        """Should extract code from untagged code blocks."""
        from lmt.recipes.code_reasoning import extract_code_block

        text = 'Solution:\n```\nx = 42\n```'
        code = extract_code_block(text)
        assert code == 'x = 42'

    def test_no_code_block_returns_full_text(self):
        """If no code block, return the full text stripped."""
        from lmt.recipes.code_reasoning import extract_code_block

        text = 'def add(a, b): return a + b'
        code = extract_code_block(text)
        assert code == 'def add(a, b): return a + b'

    def test_multiple_blocks_returns_last(self):
        """Multiple code blocks should return the last one."""
        from lmt.recipes.code_reasoning import extract_code_block

        text = '```python\nx = 1\n```\n\n```python\ndef solve():\n    return 42\n```'
        code = extract_code_block(text)
        assert code == 'def solve():\n    return 42'

    def test_empty_code_block(self):
        """Empty code block returns empty string."""
        from lmt.recipes.code_reasoning import extract_code_block

        text = '```python\n```'
        code = extract_code_block(text)
        assert code == ''


class TestCodeProblem:
    """Test CodeProblem dataclass."""

    def test_create_code_problem(self):
        """Should create a CodeProblem with prompt and tests."""
        from lmt.recipes.code_reasoning import CodeProblem

        problem = CodeProblem(
            prompt='Write a function add(a, b) that returns a + b.',
            tests='assert add(1, 2) == 3\nassert add(0, 0) == 0',
        )
        assert (
            problem.prompt == 'Write a function add(a, b) that returns a + b.'
        )
        assert 'assert add(1, 2) == 3' in problem.tests


class TestCreateCodeRewardFn:
    """Test bridging code problems to tensor-level reward functions."""

    def test_correct_code_gets_reward(self):
        """Correct code should get reward 1.0."""
        from lmt.recipes.code_reasoning import (
            CodeProblem,
            create_code_reward_fn,
        )

        problem = CodeProblem(
            prompt='Write add(a, b)',
            tests='assert add(1, 2) == 3\nassert add(0, 0) == 0',
        )

        class MockTokenizer:
            """Returns correct code."""

            def decode(self, ids: list[int]) -> str:
                """Return correct solution."""
                return '```python\ndef add(a, b):\n    return a + b\n```'

        reward_fn = create_code_reward_fn(problem, MockTokenizer())
        prompt = torch.tensor([1, 2, 3])
        response = torch.tensor([4, 5, 6])
        assert reward_fn(prompt, response) == 1.0

    def test_wrong_code_gets_partial_reward(self):
        """Code that passes some tests gets partial reward."""
        from lmt.recipes.code_reasoning import (
            CodeProblem,
            create_code_reward_fn,
        )

        problem = CodeProblem(
            prompt='Write add(a, b)',
            tests='assert add(1, 2) == 3\nassert add(-1, 1) == 0\nassert add(0, 0) == 0',
        )

        class MockTokenizer:
            """Returns code that only works for positive numbers."""

            def decode(self, ids: list[int]) -> str:
                """Return partially correct solution."""
                return '```python\ndef add(a, b):\n    return abs(a) + abs(b)\n```'

        reward_fn = create_code_reward_fn(problem, MockTokenizer())
        prompt = torch.tensor([1, 2, 3])
        response = torch.tensor([4, 5, 6])
        reward = reward_fn(prompt, response)
        # abs(1)+abs(2)=3 ✓, abs(-1)+abs(1)=2≠0 ✗, abs(0)+abs(0)=0 ✓
        assert 0.0 < reward < 1.0

    def test_syntax_error_gets_zero(self):
        """Syntax error code should get 0.0."""
        from lmt.recipes.code_reasoning import (
            CodeProblem,
            create_code_reward_fn,
        )

        problem = CodeProblem(
            prompt='Write add(a, b)',
            tests='assert add(1, 2) == 3',
        )

        class MockTokenizer:
            """Returns invalid Python."""

            def decode(self, ids: list[int]) -> str:
                """Return broken code."""
                return '```python\ndef add(a b):\n    return a + b\n```'

        reward_fn = create_code_reward_fn(problem, MockTokenizer())
        prompt = torch.tensor([1, 2, 3])
        response = torch.tensor([4, 5, 6])
        assert reward_fn(prompt, response) == 0.0
