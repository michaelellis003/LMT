"""Tests for HuggingFace dataset download helpers."""

import pytest

from lmt.data.hf_datasets import (
    download_gsm8k,
    download_humaneval,
    download_math,
    download_wikitext2,
)
from lmt.data.math_data import MathDataItem


class TestDownloadGSM8K:
    """Test GSM8K dataset download."""

    @pytest.mark.slow
    def test_download_gsm8k_train(self):
        """Should download and format GSM8K train split."""
        datasets = pytest.importorskip('datasets')
        try:
            items = download_gsm8k(split='train', max_items=5)
        except (
            datasets.exceptions.DatasetNotFoundError,
            ConnectionError,
            OSError,
        ):
            pytest.skip('GSM8K dataset not accessible')
        assert len(items) == 5
        assert all(isinstance(i, MathDataItem) for i in items)
        assert all(i.prompt for i in items)
        assert all(i.ground_truth for i in items)

    @pytest.mark.slow
    def test_download_gsm8k_test(self):
        """Should download and format GSM8K test split."""
        datasets = pytest.importorskip('datasets')
        try:
            items = download_gsm8k(split='test', max_items=3)
        except (
            datasets.exceptions.DatasetNotFoundError,
            ConnectionError,
            OSError,
        ):
            pytest.skip('GSM8K dataset not accessible')
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
        datasets = pytest.importorskip('datasets')
        try:
            items = download_math(split='test', max_items=3)
        except datasets.exceptions.DatasetNotFoundError:
            pytest.skip('MATH dataset not accessible')
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
        datasets = pytest.importorskip('datasets')
        from lmt.data.code_data import CodeDataItem

        try:
            items = download_humaneval(max_items=3)
        except (
            datasets.exceptions.DatasetNotFoundError,
            ConnectionError,
            OSError,
        ):
            pytest.skip('HumanEval dataset not accessible')
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


class TestDownloadWikitext2:
    """Test WikiText-2 dataset download."""

    @pytest.mark.slow
    def test_download_wikitext2_train(self):
        """Should download WikiText-2 train split as text."""
        datasets = pytest.importorskip('datasets')
        try:
            texts = download_wikitext2(split='train', max_items=10)
        except (
            datasets.exceptions.DatasetNotFoundError,
            ConnectionError,
            OSError,
        ):
            pytest.skip('WikiText-2 dataset not accessible')
        assert len(texts) == 10
        assert all(isinstance(t, str) for t in texts)
        assert all(len(t) > 0 for t in texts)

    @pytest.mark.slow
    def test_download_wikitext2_validation(self):
        """Should download WikiText-2 validation split."""
        datasets = pytest.importorskip('datasets')
        try:
            texts = download_wikitext2(split='validation', max_items=5)
        except (
            datasets.exceptions.DatasetNotFoundError,
            ConnectionError,
            OSError,
        ):
            pytest.skip('WikiText-2 dataset not accessible')
        assert len(texts) <= 5
        assert all(isinstance(t, str) for t in texts)

    def test_download_wikitext2_mock(self):
        """Should work with mock data (no network)."""
        mock_rows = [
            {'text': 'The quick brown fox jumps.'},
            {'text': ''},
            {'text': 'Another paragraph here.'},
            {'text': ''},
            {'text': ' = Section Title = '},
        ]
        texts = download_wikitext2(rows=mock_rows)
        # Should filter out empty lines
        assert len(texts) == 3
        assert texts[0] == 'The quick brown fox jumps.'
        assert texts[1] == 'Another paragraph here.'
        assert texts[2] == ' = Section Title = '

    def test_download_wikitext2_mock_no_filter(self):
        """Should keep empty lines when filter_empty=False."""
        mock_rows = [
            {'text': 'Hello'},
            {'text': ''},
            {'text': 'World'},
        ]
        texts = download_wikitext2(rows=mock_rows, filter_empty=False)
        assert len(texts) == 3

    def test_download_wikitext2_max_items(self):
        """Should respect max_items after filtering."""
        mock_rows = [
            {'text': 'A'},
            {'text': ''},
            {'text': 'B'},
            {'text': ''},
            {'text': 'C'},
            {'text': 'D'},
        ]
        texts = download_wikitext2(rows=mock_rows, max_items=2)
        assert len(texts) == 2
        assert texts == ['A', 'B']
