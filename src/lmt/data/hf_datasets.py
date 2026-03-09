"""HuggingFace dataset download helpers for GRPO training.

Thin wrappers that download and format standard benchmarks into
LMT's data types (MathDataItem, CodeDataItem). Supports offline
use via pre-loaded row lists.

Usage::

    from lmt.data.hf_datasets import download_gsm8k, download_humaneval

    # Download from HuggingFace Hub
    math_items = download_gsm8k(split='train', max_items=1000)
    code_items = download_humaneval(max_items=50)

    # Or use pre-loaded rows (no network needed)
    items = download_gsm8k(rows=my_rows)
"""

from __future__ import annotations

from typing import Any

from lmt.data.code_data import CodeDataItem, load_code_items
from lmt.data.math_data import (
    MathDataItem,
    load_math_items,
)


def download_gsm8k(
    split: str = 'train',
    max_items: int | None = None,
    format_prompts: bool = False,
    rows: list[dict[str, Any]] | None = None,
) -> list[MathDataItem]:
    """Download and format GSM8K dataset.

    Args:
        split: Dataset split (``'train'`` or ``'test'``).
        max_items: Maximum items to load. None = all.
        format_prompts: If True, wrap in instruction format.
        rows: Pre-loaded rows (skips download if provided).

    Returns:
        List of MathDataItem with prompt and ground_truth.
    """
    if rows is None:
        rows = _download_hf_rows('openai/gsm8k', 'main', split, max_items)

    return load_math_items(
        rows, dataset_format='gsm8k', format_prompts=format_prompts
    )


def download_math(
    split: str = 'test',
    max_items: int | None = None,
    format_prompts: bool = False,
    rows: list[dict[str, Any]] | None = None,
) -> list[MathDataItem]:
    r"""Download and format MATH dataset.

    Args:
        split: Dataset split (``'test'``, ``'train'``).
        max_items: Maximum items to load. None = all.
        format_prompts: If True, wrap in instruction format.
        rows: Pre-loaded rows (skips download if provided).

    Returns:
        List of MathDataItem with prompt and ground_truth.
        Items with unparseable ``\boxed{}`` answers are skipped.
    """
    if rows is None:
        rows = _download_hf_rows(
            'hendrycks/competition_math', None, split, max_items
        )

    return load_math_items(
        rows, dataset_format='math', format_prompts=format_prompts
    )


def download_humaneval(
    max_items: int | None = None,
    format_prompts: bool = False,
    rows: list[dict[str, Any]] | None = None,
) -> list[CodeDataItem]:
    """Download and format HumanEval dataset.

    HumanEval has only a ``'test'`` split with 164 problems.

    Args:
        max_items: Maximum items to load. None = all.
        format_prompts: If True, wrap in instruction format.
        rows: Pre-loaded rows (skips download if provided).

    Returns:
        List of CodeDataItem with prompt, tests, and metadata.
    """
    if rows is None:
        rows = _download_hf_rows(
            'openai/openai_humaneval', None, 'test', max_items
        )

    return load_code_items(
        rows,
        dataset_format='humaneval',
        format_prompts=format_prompts,
    )


_BABYLM_TRACKS: dict[str, str] = {
    'strict': 'nilq/babylm-100M',
    'strict-small': 'nilq/babylm-10M',
}


def download_babylm(
    track: str = 'strict',
    split: str = 'train',
    max_items: int | None = None,
    filter_empty: bool = True,
    rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Download BabyLM Challenge dataset as text lines.

    The BabyLM corpus contains child-directed speech, subtitles,
    children's books, and other developmentally plausible text.
    Used for the BabyLM Challenge (sample-efficient pretraining).

    The datasets are large (10M+ rows for strict, 1M+ for
    strict-small), so ``max_items`` is recommended to limit
    download size. Uses streaming to avoid downloading the
    full corpus when ``max_items`` is set.

    Args:
        track: Competition track. ``'strict'`` (100M words) or
            ``'strict-small'`` (10M words).
        split: Dataset split (``'train'``, ``'test'``,
            ``'validation'``).
        max_items: Maximum text lines to return. Strongly
            recommended — the full dataset is very large.
        filter_empty: If True (default), filter out empty lines.
        rows: Pre-loaded rows (skips download if provided).

    Returns:
        List of text strings from the BabyLM corpus.

    Raises:
        ValueError: If ``track`` is not recognized.
    """
    if rows is None:
        if track not in _BABYLM_TRACKS:
            available = ', '.join(sorted(_BABYLM_TRACKS.keys()))
            raise ValueError(
                f'Unknown track: {track!r}. Available: {available}'
            )
        repo_id = _BABYLM_TRACKS[track]
        rows = _download_hf_rows_streaming(repo_id, split, max_items)

    texts: list[str] = []
    for row in rows:
        text = row.get('text', '')
        if filter_empty and not text.strip():
            continue
        texts.append(text)
        if max_items is not None and len(texts) >= max_items:
            break

    return texts


def download_wikitext2(
    split: str = 'train',
    max_items: int | None = None,
    filter_empty: bool = True,
    rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Download WikiText-2 dataset as raw text paragraphs.

    WikiText-2 is a standard language modeling benchmark with ~2M tokens
    from Wikipedia articles. Useful for pretraining evaluation and
    architecture comparison experiments.

    Args:
        split: Dataset split (``'train'``, ``'test'``, ``'validation'``).
        max_items: Maximum text lines to return. None = all.
        filter_empty: If True (default), filter out empty/whitespace-only
            lines. WikiText-2 has many blank separator lines.
        rows: Pre-loaded rows (skips download if provided).

    Returns:
        List of non-empty text strings from WikiText-2.
    """
    if rows is None:
        rows = _download_hf_rows(
            'Salesforce/wikitext', 'wikitext-2-raw-v1', split, None
        )

    texts: list[str] = []
    for row in rows:
        text = row.get('text', '')
        if filter_empty and not text.strip():
            continue
        texts.append(text)
        if max_items is not None and len(texts) >= max_items:
            break

    return texts


def _download_hf_rows(
    repo_id: str,
    config_name: str | None,
    split: str,
    max_items: int | None,
) -> list[dict[str, Any]]:
    """Download rows from a HuggingFace dataset.

    Args:
        repo_id: HuggingFace dataset repository ID.
        config_name: Dataset configuration name (e.g. ``'main'``).
        split: Dataset split name.
        max_items: Maximum rows to return.

    Returns:
        List of row dicts.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            'The `datasets` package is required to download '
            'HuggingFace datasets. Install with: '
            'uv pip install datasets'
        ) from e

    kwargs: dict[str, Any] = {'split': split}
    if config_name is not None:
        kwargs['name'] = config_name

    ds = load_dataset(repo_id, **kwargs)

    if max_items is not None:
        ds = ds.select(range(min(max_items, len(ds))))

    return list(ds)


def _download_hf_rows_streaming(
    repo_id: str,
    split: str,
    max_items: int | None,
) -> list[dict[str, Any]]:
    """Download rows from a HuggingFace dataset using streaming.

    More memory-efficient than :func:`_download_hf_rows` for large
    datasets. Streams rows one at a time and stops after ``max_items``.

    Args:
        repo_id: HuggingFace dataset repository ID.
        split: Dataset split name.
        max_items: Maximum rows to return. None downloads all
            (not recommended for large datasets).

    Returns:
        List of row dicts.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            'The `datasets` package is required to download '
            'HuggingFace datasets. Install with: '
            'uv pip install datasets'
        ) from e

    ds = load_dataset(repo_id, split=split, streaming=True)

    rows: list[dict[str, Any]] = []
    for row in ds:
        rows.append(dict(row))
        if max_items is not None and len(rows) >= max_items:
            break

    return rows
