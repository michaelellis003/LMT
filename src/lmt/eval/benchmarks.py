r"""High-level benchmarking API for LMT models.

Wraps ``lm-evaluation-harness`` and LMT's BPB metric into a single
``evaluate_model`` function with task presets for common benchmark
suites.

Usage::

    from lmt.eval.benchmarks import evaluate_model

    results = evaluate_model(
        model=my_model,
        tokenizer=my_tokenizer,
        tasks=['leaderboard'],  # preset expands to 6 benchmarks
        device='cpu',
    )
    print(results)
    # {'ifeval': 0.32, 'bbh': 0.28, 'math_hard': 0.05, ...}

Requires the ``lm-eval`` package: ``uv sync --group eval``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lmt.eval.bpb import compute_bpb
from lmt.tokenizer.base import BaseTokenizer
from lmt.training.lm_eval_adapter import LMTEvalModel

try:
    from lm_eval import simple_evaluate
except ImportError as e:
    raise ImportError(
        'lm-evaluation-harness is required for benchmarks. '
        'Install with: uv sync --group eval'
    ) from e

# -- Task presets ------------------------------------------------------------

TASK_PRESETS: dict[str, list[str]] = {
    'leaderboard': [
        'leaderboard_ifeval',
        'leaderboard_bbh',
        'leaderboard_math_hard',
        'leaderboard_gpqa',
        'leaderboard_musr',
        'leaderboard_mmlu_pro',
    ],
    'math': [
        'minerva_math',
        'gsm8k',
    ],
    'code': [
        'humaneval',
    ],
    'common_sense': [
        'hellaswag',
        'piqa',
        'arc_easy',
        'arc_challenge',
        'winogrande',
    ],
}


def resolve_tasks(tasks: list[str]) -> list[str]:
    """Expand preset names into individual task names.

    Args:
        tasks: List of task names and/or preset names from
            ``TASK_PRESETS``.

    Returns:
        Flat list of individual task names with presets expanded.
    """
    resolved: list[str] = []
    for t in tasks:
        if t in TASK_PRESETS:
            resolved.extend(TASK_PRESETS[t])
        else:
            resolved.append(t)
    return resolved


def _extract_scores(
    results: dict[str, Any],
    task_names: list[str],
) -> dict[str, float]:
    """Extract per-task accuracy scores from lm-eval results.

    Looks for the primary metric in each task's results. Prefers
    ``acc_norm`` over ``acc``, falling back to the first numeric
    metric found.

    Args:
        results: Raw output from ``lm_eval.simple_evaluate``.
        task_names: Task names to extract scores for.

    Returns:
        Dict mapping task name to score (float between 0 and 1).
    """
    scores: dict[str, float] = {}
    task_results = results.get('results', {})

    for task in task_names:
        task_data = task_results.get(task, {})
        if not task_data:
            continue

        # Prefer normalized accuracy, then accuracy, then first metric
        score: float | None = None
        for key in ('acc_norm,none', 'acc,none', 'exact_match,none'):
            if key in task_data:
                val = task_data[key]
                if isinstance(val, (int, float)):
                    score = float(val)
                    break

        # Fallback: first numeric value that looks like a score
        if score is None:
            for key, val in task_data.items():
                if isinstance(val, (int, float)) and not key.startswith(
                    'alias'
                ):
                    score = float(val)
                    break

        if score is not None:
            scores[task] = score

    return scores


def evaluate_model(
    model: nn.Module,
    tokenizer: BaseTokenizer,
    tasks: list[str],
    device: str = 'cpu',
    num_fewshot: int = 0,
    batch_size: int = 1,
    limit: int | None = None,
    include_bpb: bool = False,
    bpb_texts: list[str] | None = None,
    bpb_bytes_per_token: float | None = None,
) -> dict[str, float]:
    """Evaluate an LMT model on standard benchmarks.

    Args:
        model: Any LMT model (GPT, LLaMA, etc.).
        tokenizer: Tokenizer with ``encode``/``decode`` methods.
        tasks: Task names or preset names (e.g. ``['leaderboard']``).
            Presets are expanded via ``TASK_PRESETS``.
        device: Device string (``'cpu'``, ``'cuda'``, ``'mps'``).
        num_fewshot: Number of few-shot examples per task.
        batch_size: Batch size for evaluation.
        limit: Max examples per task (useful for quick testing).
        include_bpb: If True, compute BPB alongside benchmarks.
        bpb_texts: Texts to compute BPB on. Required if
            ``include_bpb=True``.
        bpb_bytes_per_token: Average UTF-8 bytes per token. If None,
            estimated from ``bpb_texts``.

    Returns:
        Dict mapping task names to scores. Includes ``'bpb'`` key
        if ``include_bpb=True``.
    """
    resolved = resolve_tasks(tasks)

    # Build the lm-eval adapter
    adapter = LMTEvalModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    # Run lm-eval benchmarks
    eval_results = simple_evaluate(
        model=adapter,
        tasks=resolved,
        num_fewshot=num_fewshot,
        limit=limit,
    )

    scores = _extract_scores(eval_results, resolved)

    # Compute BPB if requested
    if include_bpb and bpb_texts:
        token_seqs = []
        total_bytes = 0
        total_tokens = 0
        for text in bpb_texts:
            ids = tokenizer.encode(text)
            token_seqs.append(ids)
            total_bytes += len(text.encode('utf-8'))
            total_tokens += len(ids)

        if bpb_bytes_per_token is None:
            bpb_bytes_per_token = total_bytes / max(total_tokens, 1)

        # Pad sequences to same length for batching
        max_len = max(len(s) for s in token_seqs)
        padded = torch.zeros(len(token_seqs), max_len, dtype=torch.long)
        for i, s in enumerate(token_seqs):
            padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)

        bpb_score = compute_bpb(
            model=model,
            token_sequences=padded,
            bytes_per_token=bpb_bytes_per_token,
        )
        scores['bpb'] = bpb_score

    return scores
