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
