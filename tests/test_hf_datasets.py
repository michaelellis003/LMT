"""Tests for HuggingFace dataset download helpers."""

import pytest

from lmt.data.hf_datasets import (
    download_gsm8k,
    download_humaneval,
    download_math,
)
from lmt.data.math_data import MathDataItem


class TestDownloadGSM8K:
    """Test GSM8K dataset download."""

    @pytest.mark.slow
    def test_download_gsm8k_train(self):
        """Should download and format GSM8K train split."""
        items = download_gsm8k(split='train', max_items=5)
        assert len(items) == 5
        assert all(isinstance(i, MathDataItem) for i in items)
        assert all(i.prompt for i in items)
        assert all(i.ground_truth for i in items)

    @pytest.mark.slow
    def test_download_gsm8k_test(self):
        """Should download and format GSM8K test split."""
        items = download_gsm8k(split='test', max_items=3)
        assert len(items) == 3

    def test_download_gsm8k_mock(self):
        """Should work with mock data (no network)."""
        mock_rows = [
            {
                'question': 'What is 2+2?',
                'answer': 'The answer is 4. #### 4',
            },
            {
                'question': 'What is 3*3?',
                'answer': '3 times 3 is 9. #### 9',
            },
        ]
        items = download_gsm8k(rows=mock_rows)
        assert len(items) == 2
        assert items[0].ground_truth == '4'
        assert items[1].ground_truth == '9'


class TestDownloadMATH:
    """Test MATH dataset download."""

    @pytest.mark.slow
    def test_download_math(self):
        """Should download and format MATH dataset."""
        items = download_math(split='test', max_items=3)
        assert len(items) <= 3
        assert all(isinstance(i, MathDataItem) for i in items)

    def test_download_math_mock(self):
        """Should work with mock data."""
        mock_rows = [
            {
                'problem': 'Find x if 2x = 6.',
                'solution': 'Dividing by 2: $x = \\boxed{3}$.',
            },
        ]
        items = download_math(rows=mock_rows)
        assert len(items) == 1
        assert items[0].ground_truth == '3'


class TestDownloadHumanEval:
    """Test HumanEval dataset download."""

    @pytest.mark.slow
    def test_download_humaneval(self):
        """Should download and format HumanEval."""
        from lmt.data.code_data import CodeDataItem

        items = download_humaneval(max_items=3)
        assert len(items) == 3
        assert all(isinstance(i, CodeDataItem) for i in items)
        assert all(i.tests for i in items)

    def test_download_humaneval_mock(self):
        """Should work with mock data."""
        from lmt.data.code_data import CodeDataItem

        mock_rows = [
            {
                'prompt': 'def add(a, b):\n',
                'test': 'assert add(1, 2) == 3',
                'canonical_solution': '    return a + b',
                'entry_point': 'add',
            },
        ]
        items = download_humaneval(rows=mock_rows)
        assert len(items) == 1
        assert isinstance(items[0], CodeDataItem)
        assert items[0].entry_point == 'add'
