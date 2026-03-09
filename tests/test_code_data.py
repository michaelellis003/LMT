"""Tests for code dataset loading utilities."""

from lmt.data.code_data import (
    CodeDataItem,
    format_code_prompt,
    load_code_items,
)
from lmt.recipes.code_reasoning import CodeProblem


class TestCodeDataItem:
    """Test CodeDataItem dataclass."""

    def test_basic_fields(self):
        """Should store prompt, tests, and optional fields."""
        item = CodeDataItem(
            prompt='Write a function add(a, b).',
            tests='assert add(1, 2) == 3',
            canonical_solution='def add(a, b): return a + b',
        )
        assert item.prompt == 'Write a function add(a, b).'
        assert item.tests == 'assert add(1, 2) == 3'
        assert item.canonical_solution == 'def add(a, b): return a + b'
        assert item.language == 'python'
        assert item.entry_point is None

    def test_defaults(self):
        """Should have sensible defaults for optional fields."""
        item = CodeDataItem(
            prompt='Write code.',
            tests='assert True',
        )
        assert item.canonical_solution is None
        assert item.language == 'python'
        assert item.entry_point is None

    def test_to_code_problem(self):
        """Should convert to CodeProblem for reward functions."""
        item = CodeDataItem(
            prompt='Write add(a, b).',
            tests='assert add(1, 2) == 3',
            language='python',
        )
        problem = item.to_code_problem()
        assert isinstance(problem, CodeProblem)
        assert problem.prompt == 'Write add(a, b).'
        assert problem.tests == 'assert add(1, 2) == 3'
        assert problem.language == 'python'


class TestFormatCodePrompt:
    """Test code prompt formatting."""

    def test_basic_formatting(self):
        """Should wrap problem in instruction format."""
        prompt = format_code_prompt(
            'Write a function that adds two numbers.',
            entry_point='add',
        )
        assert 'Write a function that adds two numbers.' in prompt
        assert 'add' in prompt

    def test_without_entry_point(self):
        """Should work without entry point."""
        prompt = format_code_prompt('Write a sorting function.')
        assert 'Write a sorting function.' in prompt

    def test_with_language(self):
        """Should include language hint."""
        prompt = format_code_prompt(
            'Implement binary search.',
            language='python',
        )
        assert 'python' in prompt.lower() or 'Python' in prompt


class TestLoadCodeItems:
    """Test loading code items from dataset rows."""

    def test_humaneval_format(self):
        """Should load HumanEval-style rows."""
        rows = [
            {
                'prompt': 'def add(a, b):\n    """Add two numbers."""\n',
                'test': (
                    'def check():\n'
                    '    assert add(1, 2) == 3\n'
                    '    assert add(0, 0) == 0\n'
                    'check()'
                ),
                'canonical_solution': '    return a + b\n',
                'entry_point': 'add',
            },
            {
                'prompt': 'def double(x):\n    """Double a number."""\n',
                'test': ('def check():\n    assert double(3) == 6\ncheck()'),
                'canonical_solution': '    return x * 2\n',
                'entry_point': 'double',
            },
        ]
        items = load_code_items(rows, dataset_format='humaneval')
        assert len(items) == 2
        assert items[0].entry_point == 'add'
        assert items[1].entry_point == 'double'
        assert items[0].canonical_solution is not None

    def test_simple_format(self):
        """Should load simple (prompt, tests) rows."""
        rows = [
            {
                'prompt': 'Write a function add(a, b).',
                'tests': 'assert add(1, 2) == 3',
            },
            {
                'prompt': 'Write a function mul(a, b).',
                'tests': 'assert mul(2, 3) == 6',
            },
        ]
        items = load_code_items(rows, dataset_format='simple')
        assert len(items) == 2
        assert items[0].prompt == 'Write a function add(a, b).'

    def test_unknown_format_raises(self):
        """Should raise ValueError for unknown format."""
        import pytest

        with pytest.raises(ValueError, match='Unknown'):
            load_code_items([], dataset_format='unknown')

    def test_skip_rows_with_missing_fields(self):
        """Should skip rows missing required fields."""
        rows = [
            {'prompt': 'Write add.', 'tests': 'assert True'},
            {'prompt': 'Missing tests field'},
            {'prompt': 'Write mul.', 'tests': 'assert mul(2, 3) == 6'},
        ]
        items = load_code_items(rows, dataset_format='simple')
        assert len(items) == 2

    def test_format_prompts_flag(self):
        """Should format prompts when requested."""
        rows = [
            {
                'prompt': 'Write add(a, b).',
                'tests': 'assert add(1, 2) == 3',
            },
        ]
        items = load_code_items(
            rows, dataset_format='simple', format_prompts=True
        )
        assert len(items) == 1
        # Formatted prompt should be longer than raw
        assert len(items[0].prompt) > len('Write add(a, b).')

    def test_to_code_problems(self):
        """Should convert all items to CodeProblems."""
        rows = [
            {
                'prompt': 'Write add.',
                'tests': 'assert add(1, 2) == 3',
            },
        ]
        items = load_code_items(rows, dataset_format='simple')
        problems = [item.to_code_problem() for item in items]
        assert len(problems) == 1
        assert isinstance(problems[0], CodeProblem)
